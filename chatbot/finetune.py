"""Affinage (fine-tuning) de notre GPT sur des scènes de théâtre mises en répliques.

    uv run python -m chatbot.finetune --init submissions/2026-11-11_153839/ckpt.pt \\
        --tokenizer submissions/2026-11-11_153839/tokenizer.json \\
        --data data/stages/chatbot_20261112/theatre/theatre_train.jsonl.zst \\
               data/stages/chatbot_20261112/theatre/moliere_train.jsonl.zst --weights 1 3 \\
        --val data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst --out runs/chat_etape1

Idée : le modèle sait déjà le français (100 M de tokens). On continue son entraînement, avec un
taux d'apprentissage plus petit, sur des textes qui ont exactement la forme du chat
(« NOM.\\nréplique\\n\\n »). Il apprend ainsi le jeu « un personnage parle, l'autre répond ».

Mesure : la loss sur des fenêtres FIXES de pièces de Molière jamais vues (moliere_val), aussi
exprimée en BPB (bits par octet, comme la compétition). Le pas 0 mesure le modèle non affiné :
c'est le point de comparaison. Toute la validation est notée, quelle que soit la taille du lot.

--persona-augment P : le persona par défaut du chat, MOLIÈRE, n'a presque pas de répliques dans
les données. Avec la probabilité P, chaque scène de Molière est ajoutée une seconde fois, avec son
personnage principal renommé MOLIÈRE et le deuxième renommé LA GRANGE (son interlocuteur dans le
chat) : le persona MOLIÈRE devient la voix moyenne des personnages de Molière.

--note TEXTE : recopié dans report.json (p. ex. « validation non tenue à l'écart » pour le run
final ré-entraîné sur tout Molière, dont les pièces de validation).

--format {repliques,tirets} (défaut repliques) : forme du texte d'entraînement ET de la validation.
« tirets » = chatbot en français moderne : chaque échange (contrat : speaker A/B en alternance)
devient « — texte\n\n — texte\n\n… », exactement ce que verra le chat (chatbot.format) :

    uv run python -m chatbot.finetune --format tirets --init runs/demo_pretrain_30m/best.pt \\
        --tokenizer submissions/2026-11-11_153839/tokenizer.json \\
        --data data/stages/chatbot_20261113/dialogues/romans_train.jsonl.zst \\
               data/stages/chatbot_20261113/dialogues/theatre_moderne_train.jsonl.zst \\
        --val data/stages/chatbot_20261113/dialogues/romans_val.jsonl.zst --out runs/chatbot_francais

Le format entre dans la clé du cache des tokens, dans report.json et dans la méta du checkpoint
(« format » : repliques-v2 ou tirets-v1) : l'export (chatbot.generate) le relit.

Carte NVIDIA (chatbot/accel.py) : --precision {auto,bf16,fp16,fp32} (auto : bf16 sur CUDA si la
carte le permet, sinon fp16 + GradScaler ; fp32 sur Mac et processeur, calcul inchangé),
--compile (torch.compile, repli automatique s'il échoue), --micro-batch (accumulation de gradient :
même pas qu'un lot plein, moins de mémoire). Reprise : state.pt est écrit à chaque évaluation ;
après une coupure, relancer la MÊME commande avec --resume (refusé si réglages ou données ont
changé). --stop-at-step N : s'arrêter proprement après le pas N (reprenable).
Code de sortie : 0 terminé, 1 divergence (loss non finie), 3 interrompu (reprendre avec --resume).
"""

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import chatbot.format as fmt
from chatbot.accel import PRECISIONS, Accel, StopRequest, pick_device, replace_file, setup_console
from chatbot.format import DEFAULT_FORMAT, FORMATS, PERSONAS, get_format, normalize_speaker
from ftb.model import build_model, load_checkpoint
from ftb.tokenizer import file_sha256, load_tokenizer
from ftb.train import lr_at, make_optimizer
from pipeline.io import read_docs

AUGMENT_PERSONA = "MOLIÈRE"
AUGMENT_INTERLOCUTOR = next(p["interlocutor"] for p in PERSONAS if p["name"] == AUGMENT_PERSONA)
# Textes de contrôle du tokenizer : on évite volontairement data/validation.jsonl (val de la
# compétition), que load_tokenizer lirait par défaut.
TOKENIZER_CHECK = ["HARPAGON.\nAu voleur ! au voleur ! à l'assassin !\n\nCLÉANTE.\nQu'avez-vous ?\n\n",
                   "Bonjour, ça va ? « Très bien » — merci ! L'œuvre d'Ève, 3,5 %…"]


def open_tokenizer(path):
    return load_tokenizer(path, check_texts=TOKENIZER_CHECK)


# ---------------------------------------------------------------------------------------------
# Données : scènes -> texte au format du chat -> tokens


def read_scene_docs(path, limit=None):
    """Documents de scènes (avec leurs champs is_moliere, play...), au plus `limit`."""
    docs = []
    for doc in read_docs(path):
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break
    return docs


def read_scenes(path, limit=None):
    return [doc["turns"] for doc in read_scene_docs(path, limit)]


def persona_copies(docs, p, rng):
    """Copies de scènes de Molière où le personnage principal devient MOLIÈRE.

    Pour chaque scène de Molière, avec la probabilité p : le personnage qui a le plus de répliques
    est renommé MOLIÈRE, le deuxième LA GRANGE (l'interlocuteur du persona MOLIÈRE au chat). La
    scène d'origine reste dans les données ; la copie s'ajoute. Idée : le persona MOLIÈRE devient la
    voix moyenne des personnages de Molière. On saute une scène à moins de 2 personnages, ou où
    MOLIÈRE / LA GRANGE parlent déjà (on ne fusionne pas deux personnages en un). Seuls les noms de
    rôle changent : le texte des répliques est recopié tel quel.
    """
    copies = []
    if p <= 0:
        return copies
    for doc in docs:
        if not doc.get("is_moliere"):
            continue
        # Tirage pour chaque scène de Molière, même sautée : le résultat ne dépend que de la graine.
        if rng.random() >= p:
            continue
        speakers = [normalize_speaker(t["speaker"]) for t in doc["turns"]]
        counts = Counter(speakers).most_common()  # à égalité : ordre d'apparition
        if len(counts) < 2 or {AUGMENT_PERSONA, AUGMENT_INTERLOCUTOR} & set(speakers):
            continue
        rename = {counts[0][0]: AUGMENT_PERSONA, counts[1][0]: AUGMENT_INTERLOCUTOR}
        copies.append([{"speaker": rename.get(s, s), "text": t["text"]}
                       for s, t in zip(speakers, doc["turns"])])
    return copies


def scenes_to_tokens(scenes, tok, seed=None, fmt_name=DEFAULT_FORMAT):
    """Concatène les scènes en un seul flux de tokens : <|endoftext|> scène <|endoftext|> scène...

    Chaque scène commence après <|endoftext|>, exactement comme le prompt du chat. `seed` mélange
    l'ordre des scènes (entraînement) ; None garde l'ordre du fichier (validation stable).
    `fmt_name` : "repliques" (scènes de théâtre) ou "tirets" (échanges A/B), voir chatbot.format.
    """
    render = get_format(fmt_name)["render_exchange"]
    order = list(range(len(scenes)))
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    texts = [render(scenes[i]) for i in order]
    ids = []
    for start in range(0, len(texts), 2000):  # encodage par lots : rapide, mémoire bornée
        for enc in tok.encode_batch(texts[start:start + 2000]):
            ids.append(tok.eot_id)
            ids.extend(enc)
    return np.array(ids, dtype=np.uint16)  # 8 192 tokens < 65 536 : 2 octets suffisent


def cached_tokens(path, tok, out_dir, index, seed, limit, persona_augment=0.0,
                  fmt_name=DEFAULT_FORMAT):
    """Tokenise un fichier une seule fois : le résultat est gardé dans --out (clé = empreintes).

    La clé contient l'empreinte du fichier chatbot/format.py : le cache dépend du contenu exact du
    code qui fabrique le texte ; changer ce code, c'est changer la clé (donc retokeniser). Elle
    contient aussi la version du format (repliques-v2 / tirets-v1) : un même fichier rendu dans
    deux formats donne deux caches distincts.
    Renvoie (tokens, clé) ; clé["persona_copies"] = nombre de scènes MOLIÈRE ajoutées.
    """
    key = {"file_sha256": file_sha256(path), "tokenizer_sha256": tok.sha256, "seed": seed,
           "limit": limit, "format": get_format(fmt_name)["version"],
           "format_sha256": file_sha256(fmt.__file__), "persona_augment": persona_augment}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(out_dir) / f"tokens_{index}_{digest}.npy"
    info = cache.with_name(cache.stem + ".json")  # à côté : nombre de copies MOLIÈRE
    if cache.exists():
        copies = (json.loads(info.read_text(encoding="utf-8"))["persona_copies"]
                  if info.exists() else None)
        return np.load(cache), dict(key, persona_copies=copies)
    docs = read_scene_docs(path, limit)
    # Générateur local : dépend de --seed et du rang du fichier, rejouable à l'identique.
    extra = persona_copies(docs, persona_augment, np.random.default_rng([seed, index]))
    arr = scenes_to_tokens([d["turns"] for d in docs] + extra, tok, seed, fmt_name)
    info.write_text(json.dumps({"key": key, "persona_copies": len(extra)}) + "\n", encoding="utf-8")
    tmp = cache.with_name(cache.name + ".tmp")
    with open(tmp, "wb") as f:  # un fichier ouvert empêche np.save d'ajouter son propre « .npy »
        np.save(f, arr)
    # Écriture atomique (le cache final est soit absent, soit complet). Le fichier est FERMÉ avant
    # le remplacement : Windows refuse de remplacer un fichier ouvert.
    replace_file(tmp, cache)
    return arr, dict(key, persona_copies=len(extra))


def token_byte_lengths(tok):
    """Nombre d'octets UTF-8 de chaque token (0 pour <|endoftext|>).

    Le tokenizer est un BPE « au niveau des octets » : chaque caractère de la chaîne interne d'un
    token représente exactement un octet. len(chaîne) = nombre d'octets, sans souci des caractères
    accentués coupés entre deux tokens.
    """
    lengths = np.array([len(tok.tk.id_to_token(i) or "") for i in range(tok.vocab_size)],
                       dtype=np.int64)
    lengths[tok.eot_id] = 0
    return lengths


# ---------------------------------------------------------------------------------------------
# Validation : fenêtres fixes, loss et BPB


def fixed_windows(tokens, seq_len, max_windows=None):
    """Fenêtres consécutives sans recouvrement : chaque token est noté une seule fois.

    Fenêtre k : entrée tokens[k*T : k*T+T], cibles tokens[k*T+1 : k*T+T+1]. La dernière peut être
    plus courte. Mêmes fenêtres à chaque mesure : les courbes sont comparables d'un pas à l'autre.
    max_windows=None : toutes les fenêtres (la taille du lot ne change que la vitesse de calcul).
    """
    out = []
    for start in range(0, len(tokens) - 1, seq_len):
        if max_windows is not None and len(out) >= max_windows:
            break
        out.append(np.asarray(tokens[start:start + seq_len + 1], dtype=np.int64))
    return out


@torch.no_grad()
def evaluate_windows(model, windows, byte_lengths, device, batch_size, accel=None):
    """Loss moyenne par token (nats) et BPB = bits totaux / octets UTF-8 du texte noté.

    accel (chatbot.accel.Accel) : même précision que l'entraînement (autocast). La loss, elle,
    est toujours calculée en float32 (logits.float()). Sans accel : float32, comme avant.
    """
    was_training = model.training
    model.eval()
    nats, ntok, nbytes = 0.0, 0, 0
    # Les fenêtres de même longueur passent ensemble (seule la dernière peut être plus courte).
    groups = {}
    for w in windows:
        groups.setdefault(len(w), []).append(w)
    for length, ws in groups.items():
        for b in range(0, len(ws), batch_size):
            xy = torch.from_numpy(np.stack(ws[b:b + batch_size]))
            x, y = xy[:, :-1].to(device), xy[:, 1:].to(device)
            with (accel.autocast() if accel is not None else contextlib.nullcontext()):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                   reduction="sum")
            nats += loss.item()
            ntok += y.numel()
            nbytes += int(byte_lengths[xy[:, 1:].numpy()].sum())
    if was_training:
        model.train()
    return {"val_loss": nats / max(ntok, 1), "val_bpb": nats / (math.log(2) * max(nbytes, 1)),
            "val_tokens": ntok, "val_bytes": nbytes}


def load_val_windows(path, tok, seq_len, max_windows=None, fmt_name=DEFAULT_FORMAT):
    """Fenêtres fixes de validation, rendues dans le MÊME format que l'entraînement."""
    tokens = scenes_to_tokens(read_scenes(path), tok, seed=None, fmt_name=fmt_name)
    return fixed_windows(tokens, seq_len, max_windows)


# ---------------------------------------------------------------------------------------------
# Échantillonnage des fenêtres d'entraînement


class WindowSampler:
    """Tire des fenêtres de seq_len+1 tokens dans plusieurs fichiers.

    Probabilité de choisir un fichier ∝ poids × nombre de tokens : poids 1 partout = toutes les
    données au même rythme ; poids 3 sur Molière = Molière vu 3 fois plus souvent par token.
    """

    def __init__(self, arrays, weights, seq_len, seed):
        self.arrays, self.seq_len = arrays, seq_len
        for i, a in enumerate(arrays):
            if len(a) < seq_len + 1:
                raise ValueError(f"fichier {i} : {len(a)} tokens, moins qu'une fenêtre ({seq_len + 1})")
        mass = np.array([w * len(a) for w, a in zip(weights, arrays)], dtype=np.float64)
        self.probs = mass / mass.sum()
        self.rng = np.random.default_rng(seed)
        self.windows_per_file = [0] * len(arrays)

    def batch(self, batch_size):
        files = self.rng.choice(len(self.arrays), size=batch_size, p=self.probs)
        rows = []
        for f in files:
            a = self.arrays[f]
            start = int(self.rng.integers(0, len(a) - self.seq_len))
            rows.append(a[start:start + self.seq_len + 1].astype(np.int64))
            self.windows_per_file[f] += 1
        return torch.from_numpy(np.stack(rows))

    def state(self):
        """État du tirage (pour state.pt) : la reprise retire exactement les mêmes fenêtres."""
        return {"rng": json.dumps(self.rng.bit_generator.state),
                "windows_per_file": list(self.windows_per_file)}

    def load_state(self, st):
        self.rng.bit_generator.state = json.loads(st["rng"])
        self.windows_per_file = list(st["windows_per_file"])


# ---------------------------------------------------------------------------------------------
# Entraînement


def save_checkpoint(path, model, model_config, meta):
    """Même format que ftb.train : relisible par ftb.model.load_checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": model_config, "meta": meta}, tmp)
    replace_file(tmp, path)  # écriture atomique : une coupure ne laisse pas de fichier à moitié écrit




def train_step(model, opt, sampler, sched, step, batch_size, micro, device, grad_clip,
               accel=None, fwd=None):
    """Un pas d'optimisation : batch_size séquences, en lots de `micro` (accumulation de gradient).

    Partagé par finetune et pretrain. accel : précision, GradScaler, transfert (chatbot.accel) ;
    None = float32 sans rien d'autre (le calcul d'avant). fwd : passe avant (modèle compilé ou non).
    Renvoie (loss moyenne, norme du gradient, taux d'apprentissage). Loss non finie : AUCUNE mise
    à jour. Gradient non fini : aucune mise à jour non plus, sauf avec GradScaler (fp16), qui saute
    lui-même ce pas et baisse son facteur : c'est son fonctionnement normal, pas une divergence.
    """
    accel = accel if accel is not None else Accel(device, "fp32")
    fwd = fwd if fwd is not None else model
    lr = lr_at(step, sched)
    for g in opt.param_groups:
        g["lr"] = lr
    xy = accel.host(sampler.batch(batch_size))
    accum = batch_size // micro
    parts = []
    for m in range(accum):
        chunk = accel.to_device(xy[m * micro:(m + 1) * micro])
        with accel.autocast():
            _, loss = fwd(chunk[:, :-1], chunk[:, 1:])
        accel.backward(loss / accum)  # gradients additionnés : même pas qu'un lot entier
        parts.append(loss.detach())
    loss_sum = 0.0
    for part in parts:  # .item() attend le GPU : une fois par pas, après tous les micro-lots
        loss_sum += part.item() / accum
    accel.unscale(opt)  # avant de couper la norme : gradients remis à leur vraie échelle
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    # Avec GradScaler (fp16), on appelle TOUJOURS accel.step : scaler.step saute lui-même le pas
    # si un gradient n'est pas fini (une loss NaN donne des gradients NaN), puis update() baisse
    # le facteur et « referme » le pas. Sans cet appel, le pas suivant planterait (« unscale_()
    # has already been called »). Sans GradScaler : pas de mise à jour si loss ou norme non finie.
    if accel.scaler is not None or (math.isfinite(loss_sum) and math.isfinite(gnorm)):
        accel.step(opt)
    opt.zero_grad(set_to_none=True)
    return loss_sum, gnorm, lr


MAX_SCALER_SKIPS = 100  # fp16 : au-delà de 100 pas sautés d'affilée, ce n'est plus normal


def diverged(loss, gnorm, accel, skips):
    """(divergence ?, nombre de pas sautés d'affilée par le GradScaler)."""
    bad_grad = not math.isfinite(gnorm)
    skips = skips + 1 if bad_grad and accel.scaler is not None else 0
    bad = (not math.isfinite(loss) or (bad_grad and accel.scaler is None)
           or skips > MAX_SCALER_SKIPS)
    return bad, skips


def exit_code(result):
    """Code de sortie du processus : 0 terminé, 1 divergence (NaN), 3 interrompu (reprenable).

    Pourquoi ? On enchaîne `pretrain train ... && finetune --init best.pt` : le && ne doit PAS
    lancer l'étape suivante sur un modèle divergé ou à peine entraîné. Les appels Python (tests)
    reçoivent toujours le dictionnaire de main().
    """
    status = result.get("status") if isinstance(result, dict) else None
    if not isinstance(status, str):  # commande data, benchmark, run terminé
        return 0
    if status == "nan":
        return 1
    return 3 if status.startswith("interrompu") else 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m chatbot.finetune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="checkpoint de départ (ftb ou étape précédente)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", nargs="+", required=True, help="fichiers de scènes .jsonl.zst")
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=30_000_000, help="tokens d'entraînement vus")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=None,
                   help="accumulation de gradient : lots de N séquences (doit diviser --batch-size)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--persona-augment", type=float, default=0.0,
                   help="probabilité d'ajouter une copie MOLIÈRE de chaque scène de Molière")
    p.add_argument("--note", default=None, help="texte libre recopié dans report.json")
    p.add_argument("--format", choices=sorted(FORMATS), default=DEFAULT_FORMAT,
                   help="repliques (théâtre, Molière) ou tirets (dialogues A/B, français moderne)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--precision", choices=PRECISIONS, default="auto",
                   help="auto : bf16 (ou fp16) sur CUDA, fp32 sur Mac et processeur")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile (CUDA) ; s'il échoue : message et repli sans compilation")
    p.add_argument("--resume", action="store_true", help="reprendre depuis --out/state.pt")
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="s'arrêter proprement après ce pas (état sauvé) : découper le run")
    p.add_argument("--limit-scenes", type=int, default=None, help="essai : N scènes par fichier")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    weights = args.weights or [1.0] * len(args.data)
    if len(weights) != len(args.data) or min(weights) <= 0:
        sys.exit("ERREUR : --weights doit donner un poids > 0 par fichier de --data")
    if not 0.0 <= args.persona_augment <= 1.0:
        sys.exit("ERREUR : --persona-augment est une probabilité entre 0 et 1")
    # Les copies MOLIÈRE renomment des personnages : sans nom de rôle (tirets), cela n'a pas de sens.
    if args.persona_augment > 0 and not get_format(args.format)["personas"]:
        sys.exit(f"ERREUR : --persona-augment n'a de sens qu'en format repliques (ici {args.format})")
    format_version = get_format(args.format)["version"]
    micro = args.micro_batch or args.batch_size
    if micro <= 0 or args.batch_size % micro:
        sys.exit(f"ERREUR : --micro-batch ({micro}) doit diviser --batch-size ({args.batch_size})")
    tokens_per_step = args.batch_size * args.seq_len
    max_steps = max(1, math.ceil(args.tokens / tokens_per_step))
    # Sans pas restants après la montée, il n'y a pas de descente cosinus (LR jamais au maximum).
    if args.warmup_steps >= max_steps:
        sys.exit(f"ERREUR : --warmup-steps ({args.warmup_steps}) doit être < au nombre de pas "
                 f"({max_steps}) ; baisse --warmup-steps ou augmente --tokens")
    out = Path(args.out)
    state_path = out / "state.pt"
    # Reprise : refusée avant de toucher à quoi que ce soit dans --out.
    if state_path.exists() and not args.resume:
        sys.exit(f"ERREUR : {out} contient un run interrompu (state.pt) : ajoute --resume, ou "
                 f"choisis un autre --out")
    if args.resume and not state_path.exists() and (out / "report.json").exists():
        sys.exit(f"ERREUR : --resume, mais {out} contient un run terminé (report.json, pas de state.pt)")
    device = pick_device(args.device)
    accel = Accel(device, args.precision, args.compile)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    torch.manual_seed(args.seed)
    tok = open_tokenizer(args.tokenizer)
    model, ckpt = load_checkpoint(args.init)
    init_meta = ckpt.get("meta", {})
    if init_meta.get("tokenizer_sha256") and init_meta["tokenizer_sha256"] != tok.sha256:
        sys.exit("ERREUR : ce tokenizer n'est pas celui du checkpoint d'initialisation")
    model_config = dict(ckpt["model_config"])
    if args.dropout != model_config.get("dropout", 0.0):
        # Même poids, mais un modèle reconstruit avec un autre taux de dropout pour l'affinage.
        state = model.state_dict()
        model = build_model(dict(model_config, dropout=args.dropout))
        model.load_state_dict(state)
    if args.seq_len > model_config["context_length"]:
        sys.exit(f"ERREUR : --seq-len dépasse le contexte du modèle ({model_config['context_length']})")
    model.to(device)
    n_params = model.num_params()

    # Données d'entraînement (cache) et fenêtres fixes de validation
    arrays, keys = [], []
    for i, path in enumerate(args.data):
        arr, key = cached_tokens(path, tok, out, i, args.seed, args.limit_scenes,
                                 args.persona_augment, args.format)
        arrays.append(arr)
        keys.append(key)
        print(f"données {path} : {len(arr):,} tokens (poids {weights[i]:g}, "
              f"copies MOLIÈRE {key['persona_copies']})", file=sys.stderr)
    sampler = WindowSampler(arrays, weights, args.seq_len, args.seed)
    # Toute la validation, rendue dans le même format que l'entraînement.
    val_windows = load_val_windows(args.val, tok, args.seq_len, fmt_name=args.format)
    byte_lengths = token_byte_lengths(tok)
    # Planning et optimiseur de l'entraînement officiel (ftb.train) : warmup linéaire puis
    # cosinus jusqu'à min_lr ; AdamW avec weight decay sur les matrices seulement.
    sched = {"lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
             "max_steps": max_steps, "weight_decay": args.weight_decay, "beta1": 0.9, "beta2": 0.95}
    opt = make_optimizer(model, sched, device)
    fwd = accel.compile(model, (micro, args.seq_len))  # le modèle lui-même sans --compile
    print(f"modèle {n_params / 1e6:.2f}M paramètres | {max_steps} pas x {tokens_per_step:,} tokens "
          f"(lots de {micro}) | validation {len(val_windows)} fenêtres | {accel.summary()}",
          file=sys.stderr)

    # Ce qui définit le calcul : une reprise doit avoir exactement les mêmes valeurs (sinon la
    # courbe et le choix de best.pt mélangeraient deux entraînements différents). --micro-batch
    # n'y est pas : il ne change que la mémoire utilisée, pas le pas d'optimisation.
    run_id = json.dumps({
        "init_sha256": file_sha256(args.init), "tokenizer_sha256": tok.sha256,
        "data": [{k: v for k, v in key.items() if k != "persona_copies"} for key in keys],
        "weights": weights, "val_sha256": file_sha256(args.val), "format": format_version,
        "tokens": args.tokens, "batch_size": args.batch_size, "seq_len": args.seq_len,
        "lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "dropout": args.dropout,
        "seed": args.seed, "persona_augment": args.persona_augment,
        "limit_scenes": args.limit_scenes}, sort_keys=True)

    best = {"val_loss": float("inf")}
    start_step, prior_time, resumed_from, losses, first = 0, 0.0, [], [], None
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st["run_id"] != run_id:
            old, new = json.loads(st["run_id"]), json.loads(run_id)
            diff = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
            sys.exit(f"ERREUR : state.pt vient d'un run avec d'autres réglages ou données "
                     f"(différences : {', '.join(diff)}) : reprise impossible")
        if st["precision"] != accel.precision:
            sys.exit(f"ERREUR : state.pt a été calculé en {st['precision']}, ici {accel.precision} : "
                     f"relancer avec --precision {st['precision']}, ou choisir un autre --out")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        accel.load_scaler_state(st["scaler"])
        sampler.load_state(st["sampler"])
        torch.random.set_rng_state(st["torch_rng"])  # dropout : mêmes tirages qu'un run d'un trait
        if st.get("cuda_rng") is not None and accel.kind == "cuda":
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start_step, prior_time = st["step"], st["train_time_s"]
        resumed_from = st["resumed_from"] + [start_step]
        losses, best, first = list(st["losses"]), json.loads(st["best"]), json.loads(st["first"])
        print(f"reprise au pas {start_step}", file=sys.stderr)
    log_path = out / "log.jsonl"
    kept = []
    if start_step and log_path.exists():  # on ne garde que le journal antérieur à la reprise
        kept = [l for l in open(log_path, encoding="utf-8") if json.loads(l)["step"] <= start_step]
    log = open(log_path, "w", encoding="utf-8")
    log.writelines(kept)
    log.flush()

    environment = accel.describe()
    meta_base = {"init": str(args.init), "tokenizer_sha256": tok.sha256, "finetune": True,
                 "format": format_version, "persona_augment": args.persona_augment,
                 "note": args.note, "init_meta": {k: init_meta.get(k) for k in
                                                         ("val_bpb", "tokens_seen", "n_params")},
                 "precision": accel.precision, "environment": environment}

    def elapsed():
        return prior_time + time.time() - t_start

    def save_state(step):
        tmp = Path(str(state_path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": accel.scaler_state(), "precision": accel.precision, "step": step,
                    "sampler": sampler.state(), "torch_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if accel.kind == "cuda" else None,
                    "run_id": run_id, "train_time_s": elapsed(), "resumed_from": resumed_from,
                    "losses": losses, "best": json.dumps(best), "first": json.dumps(first)}, tmp)
        replace_file(tmp, state_path)  # écriture atomique : une coupure ne corrompt rien

    def measure(step, train_loss, lr):
        res = evaluate_windows(model, val_windows, byte_lengths, device, args.batch_size, accel)
        rec = {"step": step, "tokens_seen": step * tokens_per_step, "lr": lr,
               "train_loss": None if train_loss is None else round(train_loss, 5),
               "val_loss": round(res["val_loss"], 5), "val_bpb": round(res["val_bpb"], 5),
               "val_tokens": res["val_tokens"], "seconds": round(elapsed(), 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        tl = "   -   " if train_loss is None else f"{train_loss:.4f}"
        print(f"pas {step:>5}/{max_steps} | train {tl} | val {res['val_loss']:.4f} "
              f"| BPB val {res['val_bpb']:.4f} ({res['val_tokens']:,} cibles) | lr {lr:.2e} "
              f"| {rec['seconds']:.0f} s", file=sys.stderr)
        if res["val_loss"] < best["val_loss"]:
            best.update(rec)
            # environment relu : statut de --compile à jour (repli possible en cours de run)
            save_checkpoint(out / "best.pt", model, model_config,
                            dict(meta_base, environment=accel.describe(), **rec))
        return rec

    if start_step == 0:
        first = measure(0, None, 0.0)  # pas 0 : modèle NON affiné, point de comparaison
    model.train()
    lr = 0.0
    status, skips, done = "terminé", 0, start_step
    with StopRequest() as stop:
        for step in range(start_step, max_steps):
            loss, gnorm, lr = train_step(model, opt, sampler, sched, step, args.batch_size, micro,
                                         device, args.grad_clip, accel, fwd)
            done = step + 1
            bad, skips = diverged(loss, gnorm, accel, skips)
            if bad:
                log.close()
                saved = (f"dernier état sain : {state_path}" if state_path.exists() else
                         "aucun état de reprise (divergence avant la première évaluation)")
                print(f"ARRÊT : loss {loss} / gradient {gnorm} non finis au pas {done} (aucune mise "
                      f"à jour faite) ; {saved} ; best.pt intact. Relancer dans un nouveau --out "
                      f"avec un --lr plus bas.", file=sys.stderr, flush=True)
                return {"status": "nan", "step": done}
            losses.append(loss)
            if done % args.eval_every == 0 or done == max_steps:
                measure(done, sum(losses) / len(losses), lr)
                losses = []
                model.train()
                if done < max_steps:
                    save_state(done)
            if done < max_steps and (stop.requested or done == args.stop_at_step):
                save_state(done)
                status = f"interrompu ({stop.requested or '--stop-at-step'})"
                break

    if status != "terminé":
        log.close()
        print(f"{status} au pas {done} : état sauvé dans {state_path} ; reprendre avec --resume",
              file=sys.stderr, flush=True)
        return {"status": status, "step": done}

    last = dict(meta_base, environment=accel.describe(), step=max_steps,
                tokens_seen=max_steps * tokens_per_step)
    save_checkpoint(out / "last.pt", model, model_config, last)
    log.close()

    report = {
        "args": vars(args), "note": args.note, "weights": weights, "device": device,
        "precision": accel.precision, "micro_batch": micro,
        "environment": accel.describe(), "peak_memory_gb": accel.peak_memory_gb(),
        "n_params": n_params, "format": format_version, "persona_augment": args.persona_augment,
        "persona_copies": sum(k["persona_copies"] or 0 for k in keys),
        "steps": max_steps, "tokens_per_step": tokens_per_step,
        "tokens_seen": max_steps * tokens_per_step,
        "data": [{"path": p, "weight": w, "tokens_available": int(len(a)),
                  "windows_sampled": n, "tokens_trained": n * args.seq_len,
                  "sha256": k["file_sha256"], "persona_copies": k["persona_copies"]}
                 for p, w, a, n, k in zip(args.data, weights, arrays, sampler.windows_per_file, keys)],
        "val": {"path": args.val, "sha256": file_sha256(args.val), "windows": len(val_windows),
                "tokens_scored": first["val_tokens"]},
        "init": {"path": args.init, "sha256": file_sha256(args.init)},
        "tokenizer_sha256": tok.sha256,
        "val_step0": first, "best": best,
        "seconds": round(elapsed(), 1), "resumed_from_steps": resumed_from,
        "torch": torch.__version__,
    }
    tmp = out / "report.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    replace_file(tmp, out / "report.json")
    if state_path.exists():
        state_path.unlink()  # run fini : l'état de reprise (gros) ne sert plus
    print(f"terminé : meilleure val {best['val_loss']:.4f} (BPB {best['val_bpb']:.4f}) au pas "
          f"{best['step']} ; pas 0 : {first['val_loss']:.4f} (BPB {first['val_bpb']:.4f})",
          file=sys.stderr)
    return report


if __name__ == "__main__":
    setup_console()  # Windows : accents et symboles sans UnicodeEncodeError
    sys.exit(exit_code(main()))
