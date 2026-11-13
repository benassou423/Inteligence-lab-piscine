"""Pré-entraînement À PARTIR DE ZÉRO d'un GPT de démo plus gros (HORS CONCOURS).

Pourquoi ? Le modèle du concours (10,2 M de paramètres) sature quand on l'affine sur le théâtre :
son erreur d'entraînement rejoint son erreur de validation, il n'a plus de place pour apprendre.
Pour la démo du chatbot, on entraîne donc un modèle plus large et plus profond, avec EXACTEMENT
la même architecture (classe ftb.model.GPT), initialisation aléatoire du kit, sur nos propres
données, puis on l'affine sur le théâtre avec chatbot/finetune.py (inchangé).

On n'utilise pas ftb/train.py : il est verrouillé sur config/official.json (taille du modèle,
budget) et lit la validation du concours. Le kit n'est pas modifié ; ce modèle n'est pas soumis.

Deux commandes :

    # 1. Préparer les tokens, une source à la fois (un fichier uint16 par source + meta.json)
    uv run python -m chatbot.pretrain data --out data/stages/chatbot_20261112/pretrain \\
        --source "melange=data/mixes/full_reference_masked_20261110/train.jsonl.gz" \\
        --source "gutenberg=data/stages/nettoye/08_exact_dedup/gutenberg_fr/part-*.jsonl.zst" \\
        --source "theatre=data/stages/chatbot_20261112/theatre/theatre_train.jsonl.zst" --workers 6

    # 2. Pré-entraîner (poids = part voulue du budget de tokens, pas poids × taille)
    uv run python -m chatbot.pretrain train --data data/stages/chatbot_20261112/pretrain \\
        --weights melange=0.6 gutenberg=0.3 theatre=0.1 \\
        --model-config '{"d_model":512,"n_layer":8,"n_head":8}' --tokens 300000000 \\
        --val data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst --out runs/demo_pretrain

    (ajouter --benchmark-steps 20 pour mesurer la vitesse sans rien écrire ; --resume pour
    reprendre un run interrompu depuis --out/state.pt)

    Code de sortie de `train` : 0 terminé, 1 divergence (loss non finie), 3 interrompu
    (SIGTERM, Ctrl-C, Ctrl-Break, --stop-at-step : reprendre avec --resume). Donc
    `train ... && finetune` n'affine que si le pré-entraînement est allé au bout.

    Carte NVIDIA (chatbot/accel.py) : --precision {auto,bf16,fp16,fp32} (auto : bf16 sur CUDA si la
    carte le permet, sinon fp16 + GradScaler ; fp32 sur Mac et processeur, calcul inchangé) et
    --compile (torch.compile, repli automatique s'il échoue). La précision est écrite dans
    state.pt : une reprise dans une autre précision est refusée.

Règle d'or : les 3 pièces de validation de Molière (Le Misanthrope, Le Sicilien, L'Avare) ne
doivent JAMAIS entrer dans le pré-entraînement : tout document qui partage au moins 3 suites de
13 mots avec elles est écarté (même mesure que chatbot/theatre.py).
"""

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import zlib
from collections import Counter, deque
from glob import glob
from itertools import chain
from pathlib import Path

import numpy as np
import torch

import chatbot.format as fmt
from chatbot.accel import PRECISIONS, Accel, StopRequest, pick_device, replace_file, setup_console
from chatbot.finetune import (WindowSampler, diverged, evaluate_windows, exit_code,
                              load_val_windows, open_tokenizer, save_checkpoint,
                              token_byte_lengths, train_step)
from chatbot.format import clean_text, render_scene
from chatbot.theatre import (EXCLUSIONS, KIT_VAL, MIN_SHARED, WS_MOLIERE, WS_README,
                             load_exclusions, load_kit, load_ws_moliere, ngram_hashes,
                             parse_scenes, same_title, shared, turns_text)
from ftb.model import build_model
from ftb.tokenizer import file_sha256
from ftb.train import lr_at, make_optimizer
from pipeline.io import read_docs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = ROOT / "submissions" / "2026-11-11_153839" / "tokenizer.json"
DEMO_NOTE = "modèle de démo hors concours, initialisation aléatoire, aucun poids externe"

# Valeurs du kit (config/official.json) : on ne change que la largeur et la profondeur.
KIT_MODEL = {"vocab_size": 8192, "context_length": 512, "mlp_ratio": 4, "dropout": 0.0,
             "tie_embeddings": True}

# ---------------------------------------------------------------------------------------------
# Filtres des documents (une seule raison par document : la première rencontrée, dans l'ordre
# liste d'exclusion -> trop court -> vocabulaire cru -> recouvrement avec la validation ;
# le plus coûteux, le calcul des 13-grammes, passe en dernier).

FILTER_VERSION = "pretrain-filtres-v2"  # changer cette étiquette = refaire la préparation
# v2 : ids du mélange officiel repris de provenance.jsonl.gz (liste d'exclusion enfin applicable)
MIN_CHARS = 200  # prose seulement : en dessous, titre, ligne de table, fragment

# Raisons de chatbot/exclusions_theatre.json qui ne concernent QUE le théâtre : un document
# écarté parce que « ce n'est pas du dialogue » reste une bonne prose pour le pré-entraînement.
# Toutes les autres raisons (obscène, illisible, orthographe trop ancienne, autre) l'écartent aussi.
THEATRE_ONLY_REASONS = {"pas_du_dialogue"}

# Vocabulaire sexuel cru : le chatbot est montré à un jury, on ne veut pas qu'il ait appris des
# obscénités. Liste volontairement courte et sans mots ambigus : on a exclu « baiser » (= embrasser
# au XVIIe siècle), « vit » (verbe vivre), « con » (italien, espagnol, ancien français), « cul »
# (cul-de-sac), « branle » (mettre en branle), « bander » (bander une plaie).
CRUDE = re.compile(r"\b(?:" + "|".join([
    r"foutre", r"fouteu(?:r|se)s?", r"fouterie", r"branl(?:er|ette|eur|euse)s?",
    r"encul(?:er|é|ée|és|ées|eur|ade)", r"bites?", r"couilles?", r"salopes?", r"putains?",
    r"bordels?", r"gamahuch\w*", r"godemich[ée]s?", r"partouzes?", r"fellations?",
    r"cunnilingus", r"sodomis\w*", r"sperme", r"clitoris",
]) + r")\b")
# Seuil : au moins 3 occurrences ET au moins 1 mot cru pour 1 000 mots. Un roman de 100 000 mots
# où « putain » ou « bordel » apparaît dix fois reste (0,1 pour mille) ; un texte érotique, où ces
# mots reviennent à chaque page (plusieurs pour mille), est écarté. Le minimum de 3 évite d'écarter
# un texte court pour un seul mot (un article de médecine qui cite une fois « sperme »).
CRUDE_MIN = 3
CRUDE_PER_THOUSAND = 1.0

BLOCK_DOCS = 2000         # documents mélangés ensemble (et tokenisés ensemble)
BLOCK_CHARS = 20_000_000  # ... ou moins : un livre Gutenberg fait ~500 000 caractères

# Un ou plusieurs sauts de ligne avec seulement des espaces entre eux = fin de paragraphe.
PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n\s*")


def clean_prose(text):
    """Normalisation du chat (chatbot.format.clean_text), paragraphe par paragraphe.

    clean_text retire TOUTES les lignes vides (dans une réplique, une ligne vide voudrait dire
    « fin de réplique »). Pour de la prose, la ligne vide sépare les paragraphes : on coupe d'abord
    sur les lignes vides, on nettoie chaque paragraphe, puis on les recolle avec UNE ligne vide.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = (clean_text(p) for p in PARAGRAPH_BREAK.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def doc_texts(doc):
    """(texte à apprendre, texte pour les 13-grammes) d'un document ou d'une scène.

    Une scène ({"turns": [...]}) est rendue au format du chat ; ses 13-grammes se calculent sur
    les répliques seules (sans les noms, qui couperaient les suites de mots), comme theatre.py.
    """
    if "turns" in doc:
        return render_scene(doc["turns"]), turns_text([doc["turns"]])
    text = clean_prose(doc.get("text", ""))
    return text, text


def crude_count(text):
    return len(CRUDE.findall(text.lower()))


def is_crude(text):
    n = crude_count(text)
    return n >= CRUDE_MIN and n * 1000 >= CRUDE_PER_THOUSAND * max(1, len(text.split()))


def validation_reference():
    """Empreintes des 13-grammes des 3 pièces de validation de Molière (toutes éditions).

    Texte brut du kit (corpus/moliere_val.txt) + répliques seules, plus les éditions Wikisource
    des mêmes pièces (autre orthographe), comme chatbot.theatre.build_moliere.
    """
    parts, titles = [], []
    for title, text in load_kit(KIT_VAL):
        titles.append(title)
        parts += [text, turns_text(parse_scenes(text, strict=False))]
    if WS_MOLIERE.exists() and WS_README.exists():
        for title, _url, text in load_ws_moliere(WS_MOLIERE, WS_README):
            if any(same_title(title, t) for t in titles):
                parts += [text, turns_text(parse_scenes(text, strict=False))]
    return np.unique(np.concatenate([ngram_hashes(p) for p in parts]))


def build_reference(exclusions_path=EXCLUSIONS):
    """Tout ce dont un processus de filtrage a besoin (petit : envoyé une fois à chaque processus)."""
    val = validation_reference()
    excl = {k: v for k, v in load_exclusions(exclusions_path).items() if v not in THEATRE_ONLY_REASONS}
    return {"val": val, "exclusions": excl,
            "val_digest": hashlib.sha256(val.tobytes()).hexdigest()[:16],
            "exclusions_sha256": file_sha256(exclusions_path)}


def filter_reason(doc, text, ngram_text, ref):
    """None si le document est gardé, sinon la raison de l'écarter."""
    for key in (doc.get("id"), doc.get("doc_id")):
        if key in ref["exclusions"]:
            return f"exclusion_{ref['exclusions'][key]}"
    # Une scène courte (deux répliques) est un vrai dialogue, déjà validé par chatbot/theatre.py :
    # le minimum de longueur ne vise que les documents en prose (titres, fragments).
    if "turns" not in doc and len(text) < MIN_CHARS:
        return "trop_court"
    if is_crude(text):
        return "vocabulaire_cru"
    if shared(ngram_hashes(ngram_text), ref["val"]) >= MIN_SHARED:
        return "recouvrement_validation_moliere"
    return None


_REF = {}  # référence d'un processus de filtrage (remplie par _init_worker)


def _init_worker(ref):
    _REF.update(ref)


def process_block(docs):
    """Nettoie et filtre un bloc : [(texte ou None, raison ou None, id)] dans le même ordre."""
    out = []
    for doc in docs:
        text, ngram_text = doc_texts(doc)
        reason = filter_reason(doc, text, ngram_text, _REF)
        doc_id = doc.get("id") or doc.get("doc_id") or f"ligne {doc['_line']}"
        out.append((None if reason else text, reason, doc_id))
    return out


# ---------------------------------------------------------------------------------------------
# Lecture des fichiers et préparation d'une source


PROVENANCE = "provenance.jsonl.gz"  # posé à côté du mélange officiel : l'id de chaque ligne


def read_jsonl(path):
    """Documents d'un fichier .jsonl.zst, .jsonl.gz ou .jsonl, un par un."""
    if str(path).endswith(".zst"):
        return read_docs(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    return (json.loads(line) for line in opener(path, "rt", encoding="utf-8") if line.strip())


def provenance_file(path):
    """provenance.jsonl.gz du même dossier que `path`, ou None."""
    p = Path(path).parent / PROVENANCE
    return p if p.exists() and Path(path).name != PROVENANCE else None


def iter_docs(files, limit=None, counts=None):
    """Documents des fichiers, un par un, au plus `limit` en tout.

    _line = rang du document dans la source (avant mélange) : identifiant de secours dans le
    rapport. Pourquoi la provenance ? Le mélange officiel n'a que le champ "text" : sans id, la
    liste chatbot/exclusions_theatre.json ne pourrait jamais s'y appliquer (3 de ses documents y
    sont). Son fichier provenance.jsonl.gz donne l'id de chaque ligne, dans le même ordre ; on
    vérifie ligne par ligne le rang ET l'empreinte sha256 du texte avant de recopier l'id : un
    décalage d'une seule ligne attribuerait les exclusions aux mauvais documents.
    `counts` (Counter) compte les ids repris de la provenance et les documents restés sans id.
    """
    counts = Counter() if counts is None else counts
    n = 0
    for path in files:
        docs = read_jsonl(path)
        head = next(docs, None)
        if head is None:
            continue
        docs = chain([head], docs)
        prov_path = provenance_file(path)
        # Provenance utilisée seulement si le fichier n'a pas ses propres ids.
        prov = (read_jsonl(prov_path) if prov_path and not (head.get("id") or head.get("doc_id"))
                else None)
        if prov is None and not (head.get("id") or head.get("doc_id")):
            print(f"ATTENTION : {path} n'a pas de champ id ni de {PROVENANCE} à côté : liste "
                  f"d'exclusion inapplicable à ce fichier", file=sys.stderr, flush=True)
        for row, doc in enumerate(docs):
            if limit is not None and n >= limit:
                return
            if prov is not None:
                info = next(prov, None)
                digest = hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()
                if info is None or info["row_zero_based"] != row or info["text_sha256"] != digest:
                    sys.exit(f"ERREUR : {path} ligne {row} ne correspond pas à {prov_path} "
                             f"(rang ou sha256 du texte différents) : ids non fiables, arrêt")
                doc["id"] = info["id"]
                counts["id_from_provenance"] += 1
            elif not (doc.get("id") or doc.get("doc_id")):
                counts["without_id"] += 1
            doc["_line"] = n
            n += 1
            yield doc
        if prov is not None and next(prov, None) is not None:
            sys.exit(f"ERREUR : {prov_path} a plus de lignes que {path} : ids non fiables, arrêt")


def shuffled_blocks(docs, rng):
    """Blocs de documents (BLOCK_DOCS ou BLOCK_CHARS au plus), chacun mélangé par le générateur.

    On ne peut pas mélanger tout le corpus d'un coup (il ne tient pas en mémoire) : on mélange à
    l'intérieur de blocs consécutifs. Assez pour casser l'ordre « un auteur après l'autre ».
    """
    block, chars = [], 0
    for doc in docs:
        block.append(doc)
        chars += len(doc.get("text", "")) + sum(len(t.get("text", "")) for t in doc.get("turns", ()))
        if len(block) >= BLOCK_DOCS or chars >= BLOCK_CHARS:
            yield [block[i] for i in rng.permutation(len(block))]
            block, chars = [], 0
    if block:
        yield [block[i] for i in rng.permutation(len(block))]


def ordered_map(fn, items, pool, ahead):
    """map(fn, items) dans l'ordre, avec au plus `ahead` blocs en cours dans les processus.

    Pool.imap lirait tout le corpus d'avance (mémoire non bornée) : on avance pas à pas.
    """
    if pool is None:
        yield from map(fn, items)
        return
    pending = deque()
    for item in items:
        pending.append(pool.apply_async(fn, (item,)))
        if len(pending) >= ahead:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def parse_source(spec):
    """« NOM=CHEMIN_OU_GLOB[:max_tokens] » -> (nom, fichiers triés, max_tokens ou None)."""
    name, sep, rest = spec.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        sys.exit(f"ERREUR : --source {spec!r} : attendu NOM=CHEMIN_OU_GLOB[:max_tokens]")
    m = re.fullmatch(r"(.+):([\d_]+)", rest)
    pattern, max_tokens = (m.group(1), int(m.group(2))) if m else (rest, None)
    files = sorted(glob(pattern))
    if not files:
        sys.exit(f"ERREUR : --source {name} : aucun fichier pour {pattern!r}")
    return name, files, max_tokens


def read_meta(out):
    path = Path(out) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sources": {}}


def write_json(path, obj):
    """Écriture atomique : le fichier final est soit l'ancien, soit le nouveau complet."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replace_file(tmp, path)


def source_key(files, max_tokens, limit_docs, seed, tok, ref):
    """Tout ce qui détermine le contenu du fichier de tokens : même clé = même résultat."""
    return {"files": [{"path": str(p), "bytes": os.path.getsize(p), "sha256": file_sha256(p),
                       "provenance_sha256": file_sha256(provenance_file(p)) if provenance_file(p)
                       else None}
                      for p in files],
            "max_tokens": max_tokens, "limit_docs": limit_docs, "seed": seed,
            "tokenizer_sha256": tok.sha256, "format_sha256": file_sha256(fmt.__file__),
            "filters": {"version": FILTER_VERSION, "min_chars": MIN_CHARS,
                        "crude_regex_sha256": hashlib.sha256(CRUDE.pattern.encode()).hexdigest()[:16],
                        "crude_min": CRUDE_MIN, "crude_per_thousand": CRUDE_PER_THOUSAND,
                        "min_shared_13grams": MIN_SHARED, "val_digest": ref["val_digest"],
                        "exclusions_sha256": ref["exclusions_sha256"]},
            "block": [BLOCK_DOCS, BLOCK_CHARS]}


def prepare_source(name, files, max_tokens, out, tok, ref, seed=1337, workers=1, limit_docs=None):
    """Écrit out/NOM.bin (uint16) : <|endoftext|> doc <|endoftext|> doc <|endoftext|> ...

    Le fichier commence par <|endoftext|> : chaque document est précédé ET suivi du séparateur,
    exactement comme un prompt du chat. Arrêt net à max_tokens (le dernier document est coupé).
    Renvoie les statistiques écrites dans meta.json.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out)
    if meta.get("tokenizer_sha256", tok.sha256) != tok.sha256:
        sys.exit(f"ERREUR : {out} a été préparé avec un autre tokenizer ; choisis un autre --out")
    key = source_key(files, max_tokens, limit_docs, seed, tok, ref)
    path = out / f"{name}.bin"
    old = meta["sources"].get(name)
    if old and old["key"] == key and path.exists() and path.stat().st_size == 2 * old["tokens"]:
        print(f"source {name} : déjà préparée avec la même clé ({old['tokens']:,} tokens), "
              f"rien à refaire", file=sys.stderr, flush=True)
        return old

    t0 = time.time()
    # Générateur propre à la source : l'ordre ne dépend que de la graine et du nom.
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    byte_lengths = token_byte_lengths(tok)
    kept, rejected, examples, id_counts = 0, Counter(), {}, Counter()
    n_tokens, n_bytes, truncated = 0, 0, False
    tmp = Path(str(path) + ".tmp")
    pool = (mp.get_context("spawn").Pool(workers, _init_worker, (ref,)) if workers > 1 else None)
    if pool is None:
        _init_worker(ref)
    try:
        with open(tmp, "wb") as f:
            first = np.array([tok.eot_id], dtype=np.uint16)
            f.write(first.tobytes())
            n_tokens = 1
            blocks = shuffled_blocks(iter_docs(files, limit_docs, id_counts), rng)
            for results in ordered_map(process_block, blocks, pool, 2 * workers):
                texts = [t for t, _, _ in results if t is not None]
                encoded = iter(tok.encode_batch(texts))  # Rust, plusieurs cœurs
                chunks = []
                for text, reason, doc_id in results:
                    if reason:
                        rejected[reason] += 1
                        examples.setdefault(reason, [])
                        if len(examples[reason]) < 20:  # quelques ids, pour vérifier à l'œil
                            examples[reason].append(doc_id)
                        continue
                    ids = np.array(next(encoded) + [tok.eot_id], dtype=np.uint16)
                    if max_tokens is not None and n_tokens + len(ids) >= max_tokens:
                        ids = ids[:max_tokens - n_tokens]
                        truncated = True
                    chunks.append(ids)
                    n_tokens += len(ids)
                    kept += 1
                    if truncated:
                        break
                if chunks:
                    block = np.concatenate(chunks)
                    n_bytes += int(byte_lengths[block].sum())
                    f.write(block.tobytes())  # écriture en flux : rien n'est gardé en mémoire
                elapsed = time.time() - t0
                print(f"source {name} : {kept:,} gardés, {sum(rejected.values()):,} écartés, "
                      f"{n_tokens:,} tokens, {n_tokens / max(elapsed, 1e-9):,.0f} tokens/s",
                      file=sys.stderr, flush=True)
                if truncated:
                    break
    finally:
        if pool is not None:
            pool.terminate()
    # Écriture atomique : NOM.bin est soit absent, soit complet. Le fichier est fermé (fin du
    # « with ») avant le remplacement. Windows : impossible si un entraînement lit encore NOM.bin
    # (np.memmap le garde ouvert) -> replace_file l'explique au lieu d'un PermissionError brut.
    replace_file(tmp, path)

    seconds = time.time() - t0
    stats = {"key": key, "file": path.name, "tokens": n_tokens, "bytes": n_bytes,
             "documents_kept": kept, "documents_rejected": dict(sorted(rejected.items())),
             "documents_read": kept + sum(rejected.values()), "rejected_examples": examples,
             # Lus avant l'arrêt à max_tokens (un bloc entier est lu d'avance) :
             "ids_from_provenance": id_counts["id_from_provenance"],
             "documents_without_id": id_counts["without_id"],  # > 0 : exclusions inapplicables
             "truncated_at_max_tokens": truncated, "seconds": round(seconds, 1),
             "tokens_per_s": round(n_tokens / max(seconds, 1e-9)),
             "tokenizer_sha256": tok.sha256, "format_sha256": key["format_sha256"], "seed": seed}
    meta = read_meta(out)  # relu : une autre source a pu être ajoutée entre-temps
    meta["tokenizer_sha256"] = tok.sha256
    meta["sources"][name] = stats
    write_json(out / "meta.json", meta)
    print(f"source {name} terminée : {n_tokens:,} tokens, {kept:,} documents gardés, écartés "
          f"{dict(rejected)}, {seconds:.0f} s", file=sys.stderr, flush=True)
    if id_counts["without_id"]:
        print(f"ATTENTION : source {name} : {id_counts['without_id']:,} documents sans id, la "
              f"liste d'exclusion ne s'y applique pas", file=sys.stderr, flush=True)
    return stats


def cmd_data(args):
    out = Path(args.out)
    tok = open_tokenizer(args.tokenizer)
    sources = [parse_source(s) for s in args.source]
    names = [n for n, _, _ in sources]
    if len(set(names)) != len(names):
        sys.exit("ERREUR : deux --source portent le même nom")
    ref = build_reference(args.exclusions)
    print(f"référence validation Molière : {len(ref['val']):,} 13-grammes ; "
          f"{len(ref['exclusions'])} documents exclus par la liste", file=sys.stderr, flush=True)
    return {name: prepare_source(name, files, max_tokens, out, tok, ref, args.seed, args.workers,
                                 args.limit_docs)
            for name, files, max_tokens in sources}


# ---------------------------------------------------------------------------------------------
# Entraînement


class ShareSampler(WindowSampler):
    """Fenêtres de seq_len+1 tokens, tirées dans plusieurs sources.

    DIFFÉRENCE avec finetune.WindowSampler (probabilité ∝ poids × taille) : ici la probabilité de
    tirer une source est ∝ son poids SEUL. Le poids est directement la part voulue du budget :
    melange=0.6 gutenberg=0.3 theatre=0.1 -> 60 % des fenêtres viennent du mélange, quelle que
    soit sa taille. Dans une source, la position est uniforme ; une fenêtre ne mélange jamais deux
    sources (chaque source est un tableau séparé). x = fenêtre[:-1], y = fenêtre[1:].
    """

    def __init__(self, arrays, weights, seq_len, seed):
        super().__init__(arrays, weights, seq_len, seed)
        w = np.asarray(weights, dtype=np.float64)
        self.probs = w / w.sum()


def load_sources(data_dir, names, tok):
    """Fichiers de tokens des sources demandées, ouverts sans les charger en mémoire (memmap)."""
    meta = read_meta(data_dir)
    if meta.get("tokenizer_sha256") != tok.sha256:
        sys.exit(f"ERREUR : {data_dir} n'a pas été préparé avec ce tokenizer")
    arrays = []
    for name in names:
        info = meta["sources"].get(name)
        if info is None:
            sys.exit(f"ERREUR : source {name} absente de {data_dir}/meta.json "
                     f"(disponibles : {', '.join(meta['sources'])})")
        # memmap : lecture à la demande, même pour un fichier de plusieurs Go (> 4 Go compris,
        # Python 64 bits). Windows : le fichier reste ouvert jusqu'à la fin du processus.
        arr = np.memmap(Path(data_dir) / info["file"], dtype=np.uint16, mode="r")
        if len(arr) != info["tokens"]:
            sys.exit(f"ERREUR : {info['file']} a {len(arr)} tokens, meta.json en annonce {info['tokens']}")
        arrays.append(arr)
    return arrays, meta


def parse_weights(specs):
    weights = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        try:
            w = float(value)
        except ValueError:
            w = -1
        if not sep or w <= 0 or name in weights:
            sys.exit(f"ERREUR : --weights {spec!r} : attendu NOM=poids (poids > 0, un par source)")
        weights[name] = w
    return weights


def model_config_from(arg):
    cfg = dict(KIT_MODEL, **json.loads(arg))
    missing = {"d_model", "n_layer", "n_head"} - set(cfg)
    if missing:
        sys.exit(f"ERREUR : --model-config doit donner {sorted(missing)}")
    if cfg["d_model"] % cfg["n_head"]:
        sys.exit("ERREUR : d_model doit être un multiple de n_head")
    return cfg


def benchmark(model, opt, sampler, sched, args, micro, accel, fwd, max_steps, tokens_per_step):
    """3 pas d'échauffement (et de compilation), puis N pas chronométrés. N'écrit rien.

    Sert à choisir --micro-batch et --precision sur le PC avant un long run : la mémoire maximale
    du GPU est affichée (CUDA).
    """
    skips = 0

    def one_step(step):
        # Même contrôle que l'entraînement : une loss NaN pendant la mesure doit donner un
        # message clair (et un code de sortie 1), pas un débit calculé sur des pas ratés.
        nonlocal skips
        loss, gnorm, _ = train_step(model, opt, sampler, sched, step, args.batch_size, micro,
                                    accel.device, args.grad_clip, accel, fwd)
        bad, skips = diverged(loss, gnorm, accel, skips)
        if bad:
            sys.exit(f"ERREUR benchmark : loss {loss} / gradient {gnorm} non finis au pas {step + 1} "
                     f"(précision {accel.precision}) : débit non mesuré. Essayer --precision bf16 "
                     f"ou fp32, ou un --lr plus bas.")

    for step in range(3):
        one_step(step)
    accel.sync()
    t0 = time.time()
    for step in range(3, 3 + args.benchmark_steps):
        one_step(step)
    accel.sync()
    seconds = time.time() - t0
    tps = args.benchmark_steps * tokens_per_step / seconds
    hours = max_steps * tokens_per_step / tps / 3600
    peak = accel.peak_memory_gb()
    print(f"benchmark : {args.benchmark_steps} pas en {seconds:.1f} s = {tps:,.0f} tokens/s | "
          f"budget {max_steps * tokens_per_step:,} tokens ({max_steps} pas) ≈ {hours:.1f} h "
          f"(hors évaluations) | précision {accel.precision} | compile {accel.compile_status}"
          + (f" | mémoire GPU max {peak} Go" if peak is not None else ""),
          file=sys.stderr, flush=True)
    return {"tokens_per_s": tps, "seconds": seconds, "estimated_hours": hours,
            "precision": accel.precision, "micro_batch": micro, "peak_memory_gb": peak,
            "environment": accel.describe()}


def cmd_train(args):
    weights = parse_weights(args.weights)
    names = list(weights)
    model_config = model_config_from(args.model_config)
    micro = args.micro_batch or args.batch_size
    if micro <= 0 or args.batch_size % micro:
        sys.exit(f"ERREUR : --micro-batch ({micro}) doit diviser --batch-size ({args.batch_size})")
    if args.seq_len > model_config["context_length"]:
        sys.exit(f"ERREUR : --seq-len dépasse le contexte du modèle ({model_config['context_length']})")
    tokens_per_step = args.batch_size * args.seq_len
    max_steps = max(1, math.ceil(args.tokens / tokens_per_step))
    # Sans pas restants après la montée, il n'y a pas de descente cosinus (LR jamais au maximum).
    if args.warmup_steps >= max_steps:
        sys.exit(f"ERREUR : --warmup-steps ({args.warmup_steps}) doit être < au nombre de pas "
                 f"({max_steps}) ; baisse --warmup-steps ou augmente --tokens")
    if args.benchmark_steps is None and (args.out is None or args.val is None):
        sys.exit("ERREUR : --out et --val sont obligatoires (sauf avec --benchmark-steps)")
    device = pick_device(args.device)
    accel = Accel(device, args.precision, args.compile)

    tok = open_tokenizer(args.tokenizer)
    if model_config["vocab_size"] != tok.vocab_size:
        sys.exit(f"ERREUR : vocab_size {model_config['vocab_size']} ≠ tokenizer {tok.vocab_size}")
    arrays, data_meta = load_sources(args.data, names, tok)
    sampler = ShareSampler(arrays, [weights[n] for n in names], args.seq_len, args.seed)

    # Initialisation aléatoire du kit (constructeur de ftb.model.GPT), graine fixée : rejouable.
    torch.manual_seed(args.seed)
    model = build_model(model_config).to(device)
    n_params = model.num_params()
    # Planning et optimiseur de l'entraînement officiel (ftb.train) : warmup linéaire puis
    # cosinus jusqu'à min_lr ; AdamW (0,9 / 0,95) avec weight decay sur les matrices seulement.
    sched = {"lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
             "max_steps": max_steps, "weight_decay": args.weight_decay, "beta1": 0.9, "beta2": 0.95}
    opt = make_optimizer(model, sched, device)
    fwd = accel.compile(model, (micro, args.seq_len))  # le modèle lui-même sans --compile
    shares = ", ".join(f"{n} {100 * p:.0f} %" for n, p in zip(names, sampler.probs))
    print(f"modèle {n_params:,} paramètres ({n_params / 1e6:.1f} M) {json.dumps(model_config)} | "
          f"{max_steps} pas x {tokens_per_step:,} tokens (lots de {micro}) | parts : {shares} | "
          f"{accel.summary()}", file=sys.stderr, flush=True)
    if args.benchmark_steps is not None:
        return benchmark(model, opt, sampler, sched, args, micro, accel, fwd, max_steps,
                         tokens_per_step)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.pt"
    if (out / "report.json").exists():
        sys.exit(f"ERREUR : {out} contient un run terminé (report.json) ; choisis un autre --out")
    if state_path.exists() and not args.resume:
        sys.exit(f"ERREUR : {out} contient un run interrompu : ajoute --resume, ou choisis un autre --out")
    # Ce qui définit le calcul : une reprise doit avoir exactement les mêmes valeurs.
    # La validation en fait partie : une autre validation (fichier reconstruit, --val-max-windows)
    # rendrait la courbe de log.jsonl et le choix de best.pt incomparables d'un pas à l'autre.
    run_id = json.dumps({"model_config": model_config, "weights": weights, "tokens": args.tokens,
                         "batch_size": args.batch_size, "seq_len": args.seq_len, "lr": args.lr,
                         "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
                         "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
                         "seed": args.seed, "tokenizer_sha256": tok.sha256,
                         "val": {"sha256": file_sha256(args.val), "max_windows": args.val_max_windows},
                         "data": {n: data_meta["sources"][n]["key"] for n in names}}, sort_keys=True)

    val_windows = load_val_windows(args.val, tok, args.seq_len, args.val_max_windows)
    byte_lengths = token_byte_lengths(tok)
    start_step, prior_time, resumed_from, losses = 0, 0.0, [], []
    best = {"val_loss": float("inf")}
    first = None
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st["run_id"] != run_id:
            old, new = json.loads(st["run_id"]), json.loads(run_id)
            diff = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
            sys.exit(f"ERREUR : state.pt vient d'un run avec d'autres réglages ou données "
                     f"(différences : {', '.join(diff)}) : reprise impossible")
        # Précision : rangée à part (un state.pt d'avant ce champ = fp32, calcul du Mac).
        st_precision = st.get("precision", "fp32")
        if st_precision != accel.precision:
            sys.exit(f"ERREUR : state.pt a été calculé en {st_precision}, ici {accel.precision} : "
                     f"relancer avec --precision {st_precision}, ou choisir un autre --out")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        accel.load_scaler_state(st.get("scaler"))
        sampler.load_state(st["sampler"])
        torch.random.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and accel.kind == "cuda":
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start_step, prior_time = st["step"], st["train_time_s"]
        resumed_from = st["resumed_from"] + [start_step]
        losses, best, first = list(st["losses"]), json.loads(st["best"]), json.loads(st["first"])
        print(f"reprise au pas {start_step}", file=sys.stderr, flush=True)
    log_path = out / "log.jsonl"
    kept = []
    if start_step and log_path.exists():  # on ne garde que le journal antérieur à la reprise
        kept = [l for l in open(log_path, encoding="utf-8") if json.loads(l)["step"] <= start_step]
    log = open(log_path, "w", encoding="utf-8")
    log.writelines(kept)
    log.flush()

    meta_base = {"pretrain": True, "note": DEMO_NOTE, "tokenizer_sha256": tok.sha256,
                 "n_params": n_params, "weights": weights, "format": fmt.FORMAT_VERSION,
                 "precision": accel.precision, "environment": accel.describe()}
    t0 = time.time()

    def train_time():
        return prior_time + time.time() - t0

    def save_state(step):
        tmp = Path(str(state_path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "scaler": accel.scaler_state(), "precision": accel.precision,
                    "sampler": sampler.state(), "torch_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if accel.kind == "cuda" else None,
                    "run_id": run_id, "train_time_s": train_time(), "resumed_from": resumed_from,
                    "losses": losses, "best": json.dumps(best), "first": json.dumps(first)}, tmp)
        replace_file(tmp, state_path)  # écriture atomique

    def measure(step, lr):
        res = evaluate_windows(model, val_windows, byte_lengths, device, micro, accel)
        session_steps = max(step - start_step, 1)
        rec = {"step": step, "tokens_seen": step * tokens_per_step, "lr": lr,
               "train_loss": round(sum(losses) / len(losses), 5) if losses else None,
               "val_loss": round(res["val_loss"], 5), "val_bpb": round(res["val_bpb"], 5),
               "tokens_per_s": round(session_steps * tokens_per_step / max(time.time() - t0, 1e-9))
               if step > start_step else None,
               "seconds": round(train_time(), 1)}
        losses.clear()  # train_loss = moyenne des pas depuis la dernière évaluation (lissée)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        tl = "   -   " if rec["train_loss"] is None else f"{rec['train_loss']:.4f}"
        print(f"pas {step:>6}/{max_steps} | train {tl} | val {res['val_loss']:.4f} | BPB val "
              f"{res['val_bpb']:.4f} | lr {lr:.2e} | {rec['tokens_per_s'] or 0:,} tok/s | "
              f"{rec['seconds'] / 60:.1f} min", file=sys.stderr, flush=True)
        # environment relu à chaque mesure : si --compile échoue en cours de run (repli), les
        # checkpoints suivants le disent, comme report.json.
        meta = dict(meta_base, environment=accel.describe(), **rec)
        # Comparaison sur la valeur EXACTE (rec est arrondi à 5 décimales pour le journal) :
        # best.pt est bien le meilleur pas, même pour un écart de moins de 1e-5.
        if res["val_loss"] < best.get("val_loss_exact", best["val_loss"]):
            best.clear()
            best.update(rec, val_loss_exact=res["val_loss"])
            save_checkpoint(out / "best.pt", model, model_config, meta)
        save_checkpoint(out / "last.pt", model, model_config, meta)
        return rec

    if start_step == 0:
        first = measure(0, 0.0)  # pas 0 : modèle aléatoire, point de départ de la courbe
    model.train()
    status = "terminé"
    lr = 0.0
    skips = 0  # pas sautés d'affilée par le GradScaler (fp16 seulement)
    with StopRequest() as stop:
        t_print = time.time()
        for step in range(start_step, max_steps):
            loss, gnorm, lr = train_step(model, opt, sampler, sched, step, args.batch_size, micro,
                                         device, args.grad_clip, accel, fwd)
            done = step + 1
            bad, skips = diverged(loss, gnorm, accel, skips)
            if bad:
                log.close()
                # state.pt n'existe qu'après la première évaluation (ou un run repris).
                saved = (f"dernier état sain : {state_path}" if state_path.exists() else
                         "aucun état de reprise (divergence avant la première évaluation)")
                print(f"ARRÊT : loss {loss} / gradient {gnorm} non finis au pas {done} (aucune mise "
                      f"à jour faite) ; {saved} ; best.pt intact. Relancer dans un nouveau --out "
                      f"avec un --lr plus bas.", file=sys.stderr, flush=True)
                return {"status": "nan", "step": done}
            losses.append(loss)
            if done % args.log_every == 0 and time.time() - t_print > 5:
                tps = (done - start_step) * tokens_per_step / max(time.time() - t0, 1e-9)
                eta = (max_steps - done) * tokens_per_step / max(tps, 1e-9) / 3600
                print(f"pas {done:>6}/{max_steps} | loss {loss:.4f} | lr {lr:.2e} | "
                      f"{tps:,.0f} tok/s | reste ≈ {eta:.1f} h", file=sys.stderr, flush=True)
                t_print = time.time()
            if done % args.eval_every == 0 or done == max_steps:
                measure(done, lr)
                model.train()
                if done < max_steps:
                    save_state(done)
            if done < max_steps and (stop.requested or done == args.stop_at_step):
                save_state(done)
                status = f"interrompu ({stop.requested or '--stop-at-step'})"
                break

    log.close()
    if status != "terminé":
        print(f"{status} au pas {done} : état sauvé dans {state_path} ; reprendre avec --resume",
              file=sys.stderr, flush=True)
        return {"status": status, "step": done}

    seconds = train_time()
    report = {
        "note": DEMO_NOTE, "user_note": args.note,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "model_config": model_config, "n_params": n_params, "device": device,
        "precision": accel.precision, "micro_batch": micro, "environment": accel.describe(),
        "peak_memory_gb": accel.peak_memory_gb(),
        "steps": max_steps, "tokens_per_step": tokens_per_step,
        "tokens_seen": max_steps * tokens_per_step,
        "sources": [{"name": n, "weight": weights[n], "share_wanted": round(float(p), 4),
                     "tokens_available": int(len(a)), "windows_sampled": w,
                     "tokens_trained": w * args.seq_len,
                     "share_real": round(w / max(1, sum(sampler.windows_per_file)), 4),
                     "passes_over_source": round(w * args.seq_len / len(a), 3),
                     "key": data_meta["sources"][n]["key"]}
                    for n, p, a, w in zip(names, sampler.probs, arrays, sampler.windows_per_file)],
        "val": {"path": args.val, "sha256": file_sha256(args.val), "windows": len(val_windows)},
        "val_step0": first, "best": best,
        "seconds": round(seconds, 1), "resumed_from_steps": resumed_from,
        "tokenizer_sha256": tok.sha256, "pretrain_sha256": file_sha256(__file__),
        "format_sha256": file_sha256(fmt.__file__),
        "data_meta_sha256": file_sha256(Path(args.data) / "meta.json"),
        "torch": torch.__version__,
    }
    write_json(out / "report.json", report)
    if state_path.exists():
        state_path.unlink()  # run fini : l'état de reprise (gros) ne sert plus
    print(f"terminé en {seconds / 3600:.2f} h : meilleure val {best['val_loss']:.4f} (BPB "
          f"{best['val_bpb']:.4f}) au pas {best['step']} ; pas 0 : BPB {first['val_bpb']:.4f}",
          file=sys.stderr, flush=True)
    return report


# ---------------------------------------------------------------------------------------------
# Ligne de commande


def build_parser():
    p = argparse.ArgumentParser(prog="python -m chatbot.pretrain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="préparer les fichiers de tokens, une source à la fois")
    d.add_argument("--out", required=True)
    d.add_argument("--source", action="append", required=True,
                   help="NOM=CHEMIN_OU_GLOB[:max_tokens] (répétable)")
    d.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    d.add_argument("--exclusions", default=str(EXCLUSIONS))
    d.add_argument("--seed", type=int, default=1337)
    d.add_argument("--workers", type=int, default=1, help="processus de nettoyage/filtrage")
    d.add_argument("--limit-docs", type=int, default=None, help="essai : N premiers documents par source")
    d.set_defaults(func=cmd_data)

    t = sub.add_parser("train", help="pré-entraîner à partir de zéro")
    t.add_argument("--data", required=True, help="dossier produit par la commande data")
    t.add_argument("--weights", nargs="+", required=True,
                   help="NOM=poids : part du budget de chaque source (normalisée), pas poids × taille")
    t.add_argument("--model-config", required=True, help='JSON, p. ex. {"d_model":512,"n_layer":8,"n_head":8}')
    t.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    t.add_argument("--val", default=None, help="scènes de validation (moliere_val.jsonl.zst)")
    t.add_argument("--val-max-windows", type=int, default=None)
    t.add_argument("--out", default=None)
    t.add_argument("--tokens", type=int, required=True, help="budget de tokens vus")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--micro-batch", type=int, default=None, help="accumulation de gradient")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--min-lr", type=float, default=1e-4)
    t.add_argument("--warmup-steps", type=int, default=300)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    t.add_argument("--precision", choices=PRECISIONS, default="auto",
                   help="auto : bf16 (ou fp16) sur CUDA, fp32 sur Mac et processeur")
    t.add_argument("--compile", action="store_true",
                   help="torch.compile (CUDA) ; s'il échoue : message et repli sans compilation")
    t.add_argument("--resume", action="store_true", help="reprendre depuis --out/state.pt")
    t.add_argument("--stop-at-step", type=int, default=None,
                   help="s'arrêter proprement après ce pas (état sauvé) : découper le run")
    t.add_argument("--benchmark-steps", type=int, default=None,
                   help="mesurer la vitesse sur N pas, sans rien écrire")
    t.add_argument("--note", default=None, help="texte libre recopié dans report.json")
    t.set_defaults(func=cmd_train)
    return p


def main(argv=None):
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)  # affichage immédiat, même redirigé (nohup)
    args = build_parser().parse_args(argv)
    return args.func(args)


# exit_code (0 terminé, 1 divergence, 3 interrompu) vient de chatbot.finetune : même règle.

if __name__ == "__main__":  # garde indispensable : les processus « spawn » réimportent ce module
    setup_console()  # Windows : accents et symboles (≈, ≠) sans UnicodeEncodeError
    sys.exit(exit_code(main()))
