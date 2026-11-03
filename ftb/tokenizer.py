"""Tokenizer de la compétition : format, vérifications, entraînement rapide et export depuis minbpe.

Format officiel : un fichier `tokenizer.json` de la librairie Hugging Face `tokenizers`.
Règles vérifiées par le kit (et par l'orga) :
  - exactement 8 192 tokens, token spécial `<|endoftext|>` compris ;
  - `<|endoftext|>` présent (séparateur de documents) ;
  - sans perte : decode(encode(texte)) == texte, octet pour octet. Un tokenizer qui normalise
    (minuscules, NFKC, suppression d'espaces...) est refusé : le BPB ne serait plus comparable.

Trois façons d'obtenir ce fichier :
  python -m ftb.tokenizer train --inputs mes_donnees/*.jsonl --out tokenizer.json
  python -m ftb.tokenizer from-minbpe --model mon_bpe.model --out tokenizer.json
  (ou ton propre code, tant que le fichier final respecte les règles)

Puis : python -m ftb.tokenizer check --tokenizer tokenizer.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from ftb import VALIDATION_PATH, load_config

CFG = load_config()
VOCAB_SIZE = CFG["tokenizer"]["vocab_size"]
EOT = CFG["tokenizer"]["eot_token"]

# Motif de découpage de GPT-4 (cl100k), le même que minbpe.RegexTokenizer par défaut
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class TokenizerError(ValueError):
    pass


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FTBTokenizer:
    """Enveloppe minimale autour d'un tokenizer.json validé."""

    def __init__(self, path):
        self.path = str(path)
        self.tk = Tokenizer.from_file(self.path)
        self.sha256 = file_sha256(self.path)
        self.vocab_size = self.tk.get_vocab_size(with_added_tokens=True)
        self.eot_id = self.tk.token_to_id(EOT)

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tk.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=False)

    def validate(self, sample_texts=()):
        """Lève TokenizerError si le tokenizer ne respecte pas les règles."""
        if self.vocab_size != VOCAB_SIZE:
            raise TokenizerError(
                f"le tokenizer a {self.vocab_size} tokens, il en faut exactement {VOCAB_SIZE} "
                f"(token spécial {EOT} compris)"
            )
        if self.eot_id is None:
            raise TokenizerError(f"le token spécial {EOT} est absent du tokenizer")
        ids_seen = set(self.tk.get_vocab(with_added_tokens=True).values())
        if max(ids_seen) >= VOCAB_SIZE:
            raise TokenizerError(f"des identifiants de tokens dépassent {VOCAB_SIZE - 1}")
        # Options qui rendent l'encodage non reproductible ou qui modifient les documents
        if self.tk.padding is not None:
            raise TokenizerError("le tokenizer ajoute du padding : désactive-le (tk.no_padding())")
        if self.tk.truncation is not None:
            raise TokenizerError("le tokenizer tronque les textes : désactive-le (tk.no_truncation())")
        if getattr(self.tk.model, "dropout", None):
            raise TokenizerError("BPE-dropout activé : l'encodage doit être déterministe (dropout=None)")
        sample_texts = list(sample_texts)
        if sample_texts and self.encode_batch(sample_texts) != [self.encode(t) for t in sample_texts]:
            raise TokenizerError("encode_batch et encode donnent des résultats différents")
        for text in sample_texts:
            ids = self.encode(text)
            if self.decode(ids) != text:
                raise TokenizerError(
                    "le tokenizer n'est pas sans perte : decode(encode(texte)) != texte. "
                    "Retire toute normalisation (minuscules, NFKC, strip...) et utilise un "
                    "BPE au niveau des octets."
                )
        return self


def load_tokenizer(path, check_texts=None):
    tok = FTBTokenizer(path)
    sample = check_texts if check_texts is not None else _default_check_texts()
    return tok.validate(sample)


def _default_check_texts():
    texts = [
        "Bonjour, ça va ? « Très bien » — merci !\n\nÉté 2026 : 3,5 % de hausse… L'œuvre d'Ève.",
        "  espaces   multiples\tet tabulations\r\nfin de ligne Windows\n",
        "Emoji 🐉, chinois 中文, arabe العربية, maths ∑x² ≥ 0, URL https://exemple.fr/a?b=1",
    ]
    if VALIDATION_PATH.exists():
        with open(VALIDATION_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 40:
                    break
                texts.append(json.loads(line)["text"])
    return texts


# ---------------------------------------------------------------------------------------------
# Construction


def _bytes_to_unicode():
    """Table octet -> caractère imprimable utilisée par le BPE « byte-level » (GPT-2)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _assemble(model, pattern):
    tk = Tokenizer(model)
    steps = []
    if pattern:
        steps.append(pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False))
    steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tk.decoder = decoders.ByteLevel()
    return tk


def from_merges(merges, pattern=GPT4_SPLIT_PATTERN):
    """Construit un tokenizer.json à partir d'une liste de fusions au format minbpe.

    merges : liste ordonnée de paires (id_a, id_b) ; la i-ème fusion crée le token 256 + i.
    Les identifiants 0..255 sont les octets bruts, exactement comme dans minbpe.
    """
    b2u = _bytes_to_unicode()
    strings = {i: b2u[i] for i in range(256)}
    vocab = {b2u[i]: i for i in range(256)}
    merge_pairs = []
    for i, (a, b) in enumerate(merges):
        idx = 256 + i
        strings[idx] = strings[a] + strings[b]
        if strings[idx] in vocab:
            raise TokenizerError(f"fusion {i} ({a}, {b}) en double")
        vocab[strings[idx]] = idx
        merge_pairs.append((strings[a], strings[b]))
    model = models.BPE(vocab=vocab, merges=merge_pairs, fuse_unk=False, byte_fallback=False)
    tk = _assemble(model, pattern)
    tk.add_special_tokens([AddedToken(EOT, special=True, normalized=False)])
    return tk


def read_minbpe_model(path):
    """Lit un fichier .model écrit par minbpe (Karpathy) : retourne (pattern, merges, spéciaux)."""
    with open(path, encoding="utf-8") as f:
        version = f.readline().strip()
        if version != "minbpe v1":
            raise TokenizerError(f"{path} : en-tête « minbpe v1 » attendu, lu « {version} »")
        pattern = f.readline().strip()
        n_special = int(f.readline().strip())
        special = {}
        for _ in range(n_special):
            tok, idx = f.readline().strip().split()
            special[tok] = int(idx)
        merges = [tuple(int(x) for x in line.split()) for line in f if line.strip()]
    return pattern, merges, special


def train_hf(files, vocab_size=VOCAB_SIZE, pattern=GPT4_SPLIT_PATTERN, max_docs=None):
    """Entraîne un BPE byte-level avec `tokenizers` (rapide) sur des .jsonl / .txt / .parquet."""
    from ftb.data import iter_documents

    model = models.BPE(fuse_unk=False, byte_fallback=False)
    tk = _assemble(model, pattern)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def gen():
        for i, doc in enumerate(iter_documents(files)):
            if max_docs is not None and i >= max_docs:
                break
            yield doc

    tk.train_from_iterator(gen(), trainer=trainer)
    return tk


def check(path, texts=None):
    tok = load_tokenizer(path)
    if texts is None:
        texts = []
        with open(VALIDATION_PATH, encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f]
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(ids) for ids in tok.encode_batch(texts))
    return {
        "tokenizer": path,
        "sha256": tok.sha256,
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "lossless": True,
        "bytes_per_token_validation": round(n_bytes / n_tokens, 4),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.tokenizer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="entraîne un BPE byte-level rapide (librairie tokenizers)")
    pt.add_argument("--inputs", nargs="+", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--max-docs", type=int, default=None)

    pm = sub.add_parser("from-minbpe", help="convertit ton BPE maison (fichier .model de minbpe)")
    pm.add_argument("--model", help="fichier .model écrit par minbpe")
    pm.add_argument("--merges-json", help="ou un JSON {\"pattern\": ..., \"merges\": [[a, b], ...]}")
    pm.add_argument("--out", required=True)

    pc = sub.add_parser("check", help="vérifie les règles et mesure la compression")
    pc.add_argument("--tokenizer", required=True)

    args = p.parse_args(argv)
    try:
        if args.cmd == "train":
            tk = train_hf(args.inputs, max_docs=args.max_docs)
            tk.save(args.out)
            print(json.dumps(check(args.out), indent=2, ensure_ascii=False))
        elif args.cmd == "from-minbpe":
            if args.model:
                pattern, merges, special = read_minbpe_model(args.model)
                extra = {k: v for k, v in special.items() if k != EOT}
                if extra:
                    print(f"attention : tokens spéciaux ignorés : {sorted(extra)}", file=sys.stderr)
            elif args.merges_json:
                with open(args.merges_json, encoding="utf-8") as f:
                    d = json.load(f)
                pattern, merges = d.get("pattern", GPT4_SPLIT_PATTERN), d["merges"]
            else:
                p.error("--model ou --merges-json est requis")
            if not pattern:
                print("attention : BPE sans motif de découpage (BasicTokenizer) : encodage lent "
                      "sur de gros volumes, préfère un RegexTokenizer.", file=sys.stderr)
            n_needed = VOCAB_SIZE - 256 - 1
            if len(merges) != n_needed:
                p.error(f"{len(merges)} fusions : il en faut exactement {n_needed} "
                        f"(256 octets + {n_needed} fusions + {EOT} = {VOCAB_SIZE})")
            tk = from_merges(merges, pattern)
            tk.save(args.out)
            print(json.dumps(check(args.out), indent=2, ensure_ascii=False))
        elif args.cmd == "check":
            print(json.dumps(check(args.tokenizer), indent=2, ensure_ascii=False))
    except TokenizerError as e:
        print(f"ERREUR tokenizer : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
