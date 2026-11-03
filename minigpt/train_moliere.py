"""Bloc 1 de bout en bout : le mini-GPT entraîné sur Molière, comparé aux bigrammes.

Étapes :
  1. lit le corpus du kit : corpus/moliere_train.txt (25 pièces) et moliere_val.txt (3 pièces) ;
  2. entraîne notre BPE (minigpt/bpe.py) sur le train SEULEMENT ;
  3. mesure les deux bigrammes (octets, tokens BPE) sur la validation : les points de comparaison ;
  4. entraîne le transformer (minigpt/gpt.py) pendant --steps pas (3 401 par défaut) sur CPU ;
  5. mesure son BPB sur TOUTE la validation, génère 5 échantillons, écrit les résultats.
Avec --bridge, une seule étape : le pont vers la compétition (BPE de 8 192 tokens au format
minbpe, que le kit convertit). Elle est à part : de 15 s à 1,5 min selon la charge du CPU, elle
ferait dépasser 10 minutes à la commande principale sur une machine chargée.

Lancer : uv run python -m minigpt.train_moliere --steps 3401   puis   ... --bridge
(nombre de pas fixé : mêmes poids au bit près à chaque relance, sur la même machine.)
Sorties dans minigpt/results/ : loss_curve.png, metrics.json, samples.md, RESULTS.md,
tokenizer.json, model.pt (ignoré par git) et, avec --bridge, bpe8192.model.
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from matplotlib.figure import Figure

from minigpt.bigram import BigramModel
from minigpt.bpe import BPETokenizer
from minigpt.gpt import GPT, GPTConfig

SEED = 1337  # la graine de la config officielle du kit
VOCAB_SIZE = 2048  # choix justifié dans docs/comprendre/minigpt_entrainement.md
BLOCK_SIZE = 128  # contexte du modèle, en tokens (≈ 400 octets de texte)
BATCH_SIZE = 32  # séquences par pas : 32 × 128 = 4 096 tokens
MODEL = dict(n_layer=4, n_head=4, n_embd=128, dropout=0.1)  # modeste : ~1 M de paramètres
LR, MIN_LR = 1e-3, 1e-4  # learning rate maximal, puis plancher en fin de cosinus
WARMUP = 0.05  # les 5 premiers % du budget : montée linéaire du learning rate
STEPS = 3401  # pas du run livré (ceux faits en 8 min de CPU lors du 1er run)
EVAL_EVERY = 50  # nombre de pas entre deux mesures de la loss
EVAL_BATCHES = 10  # batchs par mesure : 10 × 4 096 = 40 960 tokens
PROMPTS = ["ACTE PREMIER", "HARPAGON.", "Monsieur,", "Ah!", "Que"]  # typographie du corpus : « Ah! »
SAMPLE_TOKENS = 100  # ≈ 300 caractères, un token faisant ~3 octets
COMPETITION_VOCAB = 8192  # imposé par le kit : 256 octets + 7 935 fusions + <|endoftext|>
KIT_VALIDATION = Path("data/validation.jsonl")  # validation de la compétition (lecture seule)


def fr(x, digits=2):
    """Nombre à la française : « 3,35 » et « 1 071 872 », pas « 3.35 » et « 1,071,872 »."""
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def read_plays(path):
    """Renvoie le texte du fichier et les titres de ses pièces (séparées par une ligne « ===== »)."""
    text = Path(path).read_text(encoding="utf-8")
    titles = [play.strip().split("\n")[0] for play in text.split("\n=====\n")]  # 1re ligne = titre
    return text, titles


def get_batch(ids, generator=None):
    """BATCH_SIZE fenêtres prises au hasard : x = BLOCK_SIZE tokens, y = les mêmes décalés d'un cran."""
    starts = torch.randint(len(ids) - BLOCK_SIZE, (BATCH_SIZE,), generator=generator)
    x = torch.stack([ids[s : s + BLOCK_SIZE] for s in starts])
    y = torch.stack([ids[s + 1 : s + 1 + BLOCK_SIZE] for s in starts])  # y[i] = le token après x[i]
    return x, y


@torch.no_grad()  # mesure seulement : pas de gradient à préparer
def estimate_loss(model, ids):
    """Loss moyenne sur EVAL_BATCHES batchs. Plus stable que la loss d'un seul batch d'entraînement."""
    model.eval()  # coupe le dropout : on mesure le modèle tel qu'il servira
    # Générateur à part, toujours la même graine : les mêmes fenêtres à chaque mesure, donc une
    # courbe qui ne bouge que parce que le modèle change. Le tirage des batchs d'entraînement,
    # lui, n'est pas modifié par ces mesures.
    generator = torch.Generator().manual_seed(SEED)
    losses = [model(*get_batch(ids, generator))[1].item() for _ in range(EVAL_BATCHES)]
    model.train()  # remet le dropout pour la suite de l'entraînement
    return sum(losses) / len(losses)


def lr_at(progress):
    """Learning rate quand une fraction `progress` (de 0 à 1) du budget est écoulée.

    Warmup : montée linéaire de 0 à LR (au départ, les gradients d'un modèle au hasard sont
    peu fiables : on fait de petits pas). Puis cosinus : descente douce de LR à MIN_LR.
    """
    if progress < WARMUP:
        return LR * progress / WARMUP
    t = (progress - WARMUP) / (1 - WARMUP)  # 0 → 1 sur la partie cosinus
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))  # cos(0) = 1, cos(π) = −1


def train_model(model, train_ids, val_ids, minutes, steps):
    """Entraîne le modèle. Renvoie l'historique [(pas, loss train, loss validation)] et le nombre de pas.

    Budget : `minutes` de calcul (la vitesse dépend de la machine), ou exactement `steps` pas si
    steps > 0 (même graine + même nombre de pas = même modèle, au bit près).
    """
    # β et weight decay de la config officielle du kit. Le weight decay tire chaque poids un peu
    # vers 0 à chaque pas : le modèle apprend moins facilement le train par cœur.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    history, step, start = [], 0, time.perf_counter()
    while True:
        progress = step / steps if steps else (time.perf_counter() - start) / (60 * minutes)
        if step % EVAL_EVERY == 0 or progress >= 1:  # une mesure régulière, et une à la toute fin
            history.append((step, estimate_loss(model, train_ids), estimate_loss(model, val_ids)))
            print(f"pas {step:5d} | loss train {history[-1][1]:.3f} | loss val {history[-1][2]:.3f} "
                  f"| {time.perf_counter() - start:5.0f} s")
        if progress >= 1:
            return history, step
        for group in optimizer.param_groups:  # AdamW n'a qu'un groupe ici : tous les paramètres
            group["lr"] = lr_at(progress)
        _, loss = model(*get_batch(train_ids))
        optimizer.zero_grad()  # sinon les gradients s'ajouteraient à ceux du pas précédent
        loss.backward()
        # Garde-fou (kit : clip 1,0) : si la norme de tous les gradients dépasse 1, on les réduit
        # tous d'un même facteur. Un batch bizarre ne peut plus faire faire un pas énorme.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1


@torch.no_grad()
def total_nats(model, ids):
    """Somme des −ln p(token | contexte) sur tous les tokens de `ids` sauf le 1er (aucun contexte).

    Fenêtres consécutives, sans recouvrement : chaque token est prédit une seule fois, avec
    entre 1 et BLOCK_SIZE tokens de contexte (le modèle ne peut pas lire plus loin).
    """
    model.eval()
    nats = 0.0
    for start in range(0, len(ids) - 1, BLOCK_SIZE):
        y = ids[start + 1 : start + 1 + BLOCK_SIZE]  # les tokens à prédire
        x = ids[start : start + len(y)]  # les entrées, un cran plus tôt : y[i] est prédit à partir de x[0..i]
        _, loss = model(x[None], y[None])  # [None] ajoute la dimension du batch : (1, T)
        nats += loss.item() * len(y)  # loss = moyenne sur la fenêtre ; × nombre de tokens = somme
    return nats


def generate_samples(model, tok):
    """Un échantillon par amorce : SAMPLE_TOKENS tokens, température 0,8, top-k 50."""
    torch.manual_seed(SEED)  # mêmes échantillons à chaque exécution
    model.eval()  # sans dropout
    samples = []
    for prompt in PROMPTS:
        idx = torch.tensor([tok.encode(prompt)])  # (1, T) : un batch d'une seule séquence
        out = model.generate(idx, SAMPLE_TOKENS, temperature=0.8, top_k=50)
        samples.append((prompt, tok.decode(out[0].tolist())))
    return samples


def plot_losses(history, bigram_loss, path):
    """Courbe de loss : train et validation au fil des pas, avec le bigramme BPE pour repère."""
    steps, train_losses, val_losses = zip(*history)  # liste de triplets → 3 listes
    fig = Figure(figsize=(8, 4.5))  # Figure directement, sans pyplot : aucune fenêtre, rien à fermer
    ax = fig.subplots()
    ax.plot(steps, train_losses, color="#2a78d6", linewidth=2, label="entraînement (25 pièces)")
    ax.plot(steps, val_losses, color="#eb6834", linewidth=2, label="validation (3 pièces jamais vues)")
    ax.axhline(bigram_loss, color="#52514e", linestyle="--", linewidth=1, label="bigramme BPE (validation)")
    ax.set_xlabel("Pas d'entraînement")
    ax.set_ylabel("Loss (nats par token)")
    ax.set_title("Mini-GPT sur Molière : loss au fil de l'entraînement")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def competition_bridge(train_text, out):
    """Pont vers la compétition : notre BPE en 8 192 tokens, appris sur le train, au format minbpe."""
    tok = BPETokenizer()
    start = time.perf_counter()
    tok.train(train_text, COMPETITION_VOCAB)
    seconds = time.perf_counter() - start
    tok.save_minbpe(out / f"bpe{COMPETITION_VOCAB}.model")
    # Compression sur la validation du kit (230 textes récents, 4 genres), texte par texte :
    # le même calcul que « python -m ftb.tokenizer check ».
    with open(KIT_VALIDATION, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    return {"vocab_size": COMPETITION_VOCAB, "n_merges": len(tok.merges), "train_seconds": round(seconds, 1),
            "bytes_per_token_kit_val": round(n_bytes / n_tokens, 4)}


def write_report(m, out):
    """RESULTS.md : le tableau des résultats et leur interprétation, écrits depuis metrics.json."""
    lines = ["# Bloc 1 : le mini-GPT sur Molière", ""]
    if "bpb_val_transformer" in m:
        bpb, bigram_bytes, bigram_tokens = m["bpb_val_transformer"], m["bpb_val_bigram_bytes"], m["bpb_val_bigram_tokens"]
        tokens_seen = m["steps"] * BATCH_SIZE * BLOCK_SIZE
        best_step, _, best_val = min(m["history"], key=lambda h: h[2])  # mesure où la loss de validation est minimale
        if m.get("fixed_steps"):
            replay = (f"Rejouer : `uv run python -m minigpt.train_moliere --steps {m['steps']}` (graine {m['seed']}, "
                      f"{m['threads']} threads) redonne les mêmes poids au bit près, sur la même machine avec les "
                      f"mêmes versions de Python et de torch. Puis `… --bridge` pour le pont vers la compétition.")
        else:
            replay = "Budget en minutes : le nombre de pas dépend de la charge du CPU, ce run ne se rejoue pas au bit près."
        lines += [
            f"Validation : les 3 pièces tenues à l'écart par le kit ({fr(m['n_val_bytes'], 0)} octets UTF-8, "
            f"{fr(m['n_val_tokens'], 0)} tokens BPE). BPB = bits de **tout** ce texte ÷ ses octets.",
            "",
            "| Modèle | BPB validation (bits par octet, plus bas = mieux) |",
            "|---|---|",
            f"| Bigramme d'octets (table 256 × 256, lissage de Laplace) | {fr(bigram_bytes, 3)} |",
            f"| Bigramme de tokens BPE (table {m['vocab_size']} × {m['vocab_size']}, Laplace) | {fr(bigram_tokens, 3)} |",
            f"| **Mini-GPT** ({fr(m['n_params'], 0)} paramètres) | **{fr(bpb, 3)}** |",
            "",
            "| Réglage | Valeur |",
            "|---|---|",
            f"| Tokenizer | notre BPE, {m['vocab_size']} tokens, {fr(m['bytes_per_token_val'])} octets par token "
            f"en validation |",
            f"| Modèle | {MODEL['n_layer']} blocs, {MODEL['n_head']} têtes, n_embd {MODEL['n_embd']}, "
            f"contexte {BLOCK_SIZE} tokens, dropout {fr(MODEL['dropout'], 1)} |",
            f"| Entraînement | {fr(m['steps'], 0)} pas de {BATCH_SIZE} × {BLOCK_SIZE} tokens, AdamW, warmup puis "
            f"cosinus (lr {fr(LR, 3)} → {fr(MIN_LR, 4)}), {fr(m['train_seconds'] / 60, 1)} min sur "
            f"{m['threads']} threads CPU |",
            f"| Loss finale (nats par token) | {fr(m['final_train_loss'], 3)} en train, "
            f"{fr(m['final_val_loss'], 3)} en validation |",
            f"| Graine | {m['seed']} |",
            "",
            "Interprétation :",
            "",
            f"1. Sur 3 pièces jamais vues, le mini-GPT dépense **{fr(bpb)} bits par octet** : "
            f"{fr(100 * (1 - bpb / bigram_bytes), 0)} % de moins que le bigramme d'octets et "
            f"{fr(100 * (1 - bpb / bigram_tokens), 0)} % de moins que le bigramme de tokens.",
            f"2. Le bigramme de tokens bat déjà celui d'octets : un token couvre {fr(m['bytes_per_token_val'], 1)} "
            f"octets, donc « le précédent » en dit plus. Le reste du gain vient du contexte : le mini-GPT lit "
            f"jusqu'à {BLOCK_SIZE} tokens (≈ {fr(BLOCK_SIZE * m['bytes_per_token_val'], 0)} octets).",
            f"3. La loss de validation passe de {fr(m['history'][0][2])} (≈ ln {m['vocab_size']} = "
            f"{fr(math.log(m['vocab_size']))} : le hasard) à {fr(m['final_val_loss'])}. "
            + ("Elle baisse encore à la dernière mesure." if best_step == m["steps"] else
               f"Son minimum, {fr(best_val, 3)}, est mesuré au pas {fr(best_step, 0)} ; elle finit "
               f"{fr(m['final_val_loss'] - best_val, 3)} au-dessus : elle plafonne."),
            f"4. Écart validation − train en fin d'entraînement : {fr(m['final_val_loss'] - m['final_train_loss'], 3)} "
            f"nat par token. Plus il grandit, plus le modèle apprend le train par cœur au lieu de généraliser "
            f"(surapprentissage).",
            f"5. Limite : {fr(tokens_seen / 1e6, 1)} M de tokens vus, soit ≈ {fr(tokens_seen / m['n_train_tokens'], 0)} "
            f"passages sur les {fr(m['n_train_tokens'], 0)} tokens du train. Le modèle de la compétition a "
            f"10 M de paramètres et voit 100 M de tokens.",
            "6. Prudence : n_embd (128 plutôt que 192) et le learning rate (0,001 plutôt que 0,002) ont été choisis "
            "sur le BPB de ces 3 pièces (essais de 3 min). La validation a donc un peu servi à régler le modèle : "
            "ce BPB est **légèrement optimiste**. Une mesure sans biais demanderait un 3e jeu, jamais regardé.",
            "",
            replay,
            "",
            "Fichiers : `loss_curve.png` (courbe), `samples.md` (5 échantillons), `metrics.json` (chiffres), "
            "`tokenizer.json` (notre BPE), `model.pt` (poids, ignoré par git).",
            "",
        ]
    if "bridge" in m:
        b = m["bridge"]
        lines += [
            "## Pont vers la compétition : notre BPE en 8 192 tokens",
            "",
            "| Mesure | Valeur |",
            "|---|---|",
            f"| Entraînement sur les 25 pièces du train ({fr(b['n_merges'], 0)} fusions) | {fr(b['train_seconds'], 0)} s |",
            f"| Octets par token, validation du kit (`data/validation.jsonl`, 230 textes) | "
            f"{fr(b['bytes_per_token_kit_val'], 4)} |",
            "",
            "Conversion et vérification par le kit (il encode exactement comme `minigpt/bpe.py`) :",
            "",
            "```bash",
            "uv run python -m ftb.tokenizer from-minbpe --model minigpt/results/bpe8192.model "
            "--out minigpt/results/tokenizer_8192.json",
            "uv run python -m ftb.tokenizer check --tokenizer minigpt/results/tokenizer_8192.json",
            "```",
            "",
        ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_minigpt(train_text, val_text, vocab_size, minutes, steps, out):
    """Étapes 2 à 5 : tokenizer, bigrammes, transformer, BPB et échantillons. Renvoie les chiffres."""
    # 2. Le tokenizer, appris sur le train seulement : la validation reste du texte jamais vu.
    tok = BPETokenizer()
    tok.train(train_text, vocab_size)
    tok.save(out / "tokenizer.json")
    train_ids = torch.tensor(tok.encode(train_text))
    val_ids = torch.tensor(tok.encode(val_text))
    n_val_bytes = len(val_text.encode("utf-8"))  # le diviseur du BPB : les octets de TOUT le texte
    print(f"BPE {vocab_size} tokens : {len(train_ids)} tokens de train, soit {len(train_ids) / vocab_size:.0f} "
          f"occurrences par token en moyenne ; {n_val_bytes / len(val_ids):.2f} octets par token en validation")

    # 3. Les points de comparaison : bigrammes sur octets et sur tokens BPE, même texte de validation.
    val_bytes = list(val_text.encode("utf-8"))
    bigram_bytes = BigramModel(256).fit(list(train_text.encode("utf-8"))).bpb(val_bytes, n_val_bytes)
    bigram = BigramModel(vocab_size).fit(train_ids)
    bigram_tokens = bigram.bpb(val_ids, n_val_bytes)
    print(f"BPB validation : bigramme d'octets {bigram_bytes:.3f}, bigramme de tokens {bigram_tokens:.3f}")

    # 4. Le transformer.
    config = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE, **MODEL)
    model = GPT(config)
    print(f"Mini-GPT : {model.num_params()} paramètres")
    start = time.perf_counter()
    history, n_steps = train_model(model, train_ids, val_ids, minutes, steps)
    train_seconds = time.perf_counter() - start
    torch.save({"config": asdict(config), "model": model.state_dict()}, out / "model.pt")

    # 5. BPB = somme des −log2 p sur tous les tokens ÷ octets du texte. Les nats (ln) deviennent des
    #    bits en divisant par ln 2, car log2(x) = ln(x) / ln(2).
    bpb = total_nats(model, val_ids) / (math.log(2) * n_val_bytes)
    print(f"BPB validation du mini-GPT : {bpb:.3f}")
    with open(out / "samples.md", "w", encoding="utf-8") as f:
        f.write(f"# 5 échantillons du mini-GPT\n\n{SAMPLE_TOKENS} tokens générés après chaque amorce, "
                f"température 0,8, top-k 50, graine {SEED}.\n")
        for i, (prompt, text) in enumerate(generate_samples(model, tok), 1):
            f.write(f"\n## {i}. Amorce « {prompt} » ({len(text)} caractères)\n\n```text\n{text}\n```\n")
    # Loss du bigramme de tokens en nats par token (tous les tokens sauf le 1er) : repère sur la courbe.
    plot_losses(history, bigram.nll_bits(val_ids) * math.log(2) / (len(val_ids) - 1), out / "loss_curve.png")
    return {
        "bpb_val_transformer": bpb,
        "bpb_val_bigram_bytes": bigram_bytes,
        "bpb_val_bigram_tokens": bigram_tokens,
        "bytes_per_token_val": n_val_bytes / len(val_ids),
        "n_params": model.num_params(),
        "train_seconds": round(train_seconds, 1),
        "steps": n_steps,
        "fixed_steps": steps > 0,  # True : nombre de pas imposé, donc run rejouable au bit près
        "seed": SEED,
        "threads": torch.get_num_threads(),
        "vocab_size": vocab_size,
        "model": asdict(config),
        "n_train_tokens": len(train_ids),
        "n_val_tokens": len(val_ids),
        "n_val_bytes": n_val_bytes,
        "final_train_loss": history[-1][1],
        "final_val_loss": history[-1][2],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Entraîne le mini-GPT sur Molière (bloc 1).")
    parser.add_argument("--train", default="corpus/moliere_train.txt")  # 25 pièces (kit)
    parser.add_argument("--val", default="corpus/moliere_val.txt")  # 3 pièces tenues à l'écart (kit)
    parser.add_argument("--out", default="minigpt/results")
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS, help="nombre de pas : run rejouable au bit près")
    parser.add_argument("--minutes", type=float, help="budget en minutes à la place des pas (pas rejouable)")
    parser.add_argument("--bridge", action="store_true", help="seulement le BPE 8 192 de la compétition")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)  # le bloc 1 tourne sur CPU ; 4 threads laissent de la place aux autres chantiers
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # 1. Le corpus du kit : on garde son découpage train / validation.
    train_text, train_titles = read_plays(args.train)
    val_text, val_titles = read_plays(args.val)
    print(f"Entraînement : {len(train_titles)} pièces : {', '.join(train_titles)}")
    print(f"Validation : {len(val_titles)} pièces : {', '.join(val_titles)}")

    # metrics.json garde les chiffres des deux commandes ; RESULTS.md est réécrit à partir de lui.
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if args.bridge:
        metrics["bridge"] = competition_bridge(train_text, out)
        print(f"Pont vers la compétition : {metrics['bridge']}")
    else:
        steps = 0 if args.minutes else args.steps  # steps = 0 : train_model suit le chronomètre
        metrics.update(run_minigpt(train_text, val_text, args.vocab, args.minutes, steps, out))
        metrics.update(train_plays=train_titles, val_plays=val_titles,
                       total_seconds=round(time.perf_counter() - start, 1))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, out)
    print(f"Résultats dans {out}/ ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
