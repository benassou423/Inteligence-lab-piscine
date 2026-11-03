"""Tokenizer BPE (Byte Pair Encoding) au niveau des octets, codé from scratch.

Principe : on part des 256 octets possibles, puis on fusionne pas à pas la paire de
tokens voisins la plus fréquente du corpus. Chaque fusion crée un nouveau token.
La base est faite d'octets : n'importe quel texte s'encode, aucun caractère n'est inconnu.

Références : Gage 1994 (l'algorithme de compression d'origine), Sennrich et al. 2016
(BPE en traduction automatique : les mots rares coupés en sous-mots), GPT-2 (Radford et al.
2019, BPE sur les octets), A. Karpathy, « Let's build the GPT Tokenizer » et le dépôt minbpe.

Démo : uv run python -m minigpt.bpe [--vocab 2048] (corpus Molière du kit : corpus/moliere_*.txt)
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import regex

# Pré-découpage de GPT-4 (motif cl100k_base de tiktoken). Les fusions restent à
# l'intérieur d'un morceau : un token ne colle jamais un mot au mot suivant, ni à la
# ponctuation qui le suit. Seul 1 caractère placé devant le mot peut y rester : « 'est », « -vous ».
# Les alternatives sont essayées dans l'ordre (\p{L} = lettre, \p{N} = chiffre, \s = blanc) :
#   '(?i:[sdmt]|ll|ve|re)        une contraction anglaise ('s, 't, 'll…, majuscules comprises)
#   [^\r\n\p{L}\p{N}]?+\p{L}+    un mot, avec au plus 1 caractère devant (espace, apostrophe…)
#   \p{N}{1,3}                   un nombre, par paquets de 3 chiffres au plus
#    ?[^\s\p{L}\p{N}]++[\r\n]*   ponctuation ou symboles (€, emoji…), avec un espace devant et
#                                des retours à la ligne derrière, tous deux facultatifs
#   \s*[\r\n]                    des blancs, jusqu'au dernier retour à la ligne compris
#   \s+(?!\S)                    des blancs non suivis d'un caractère visible : devant un mot,
#                                le dernier espace est laissé au mot
#   \s+                          les blancs restants
# « ?+ » et « ++ » sont possessifs : la regex ne revient jamais en arrière dessus. Ici, cela
# ne change pas le découpage (les caractères pris ne pourraient servir à rien d'autre).
# C'est, caractère pour caractère, le motif du kit (ftb.tokenizer.GPT4_SPLIT_PATTERN, vérifié
# par un test) : converti pour la compétition, notre BPE découpe exactement comme ici.
# Prix à payer en français : les contractions anglaises coupent « p'tite » en « p », « 't »,
# « ite ». C'est rare (l'élision se fait devant une voyelle) : 6 morceaux sur 341 416 dans le
# Molière du kit (« v's », « d'mère »).
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Tokens spéciaux : ils ne viennent pas du texte, on les ajoute à la fin du vocabulaire.
# <|endoftext|> sépare deux documents (comme dans GPT-2).
SPECIAL_TOKENS = ("<|endoftext|>",)


def merge(ids, pair, new_id):
    """Remplace chaque occurrence de `pair` dans `ids` par `new_id`, de gauche à droite."""
    out = []
    i = 0
    while i < len(ids):
        # La paire commence ici : on écrit le nouveau token et on saute ses 2 éléments.
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def render(token):
    """Octets d'un token -> texte affichable entre guillemets.

    Un octet isolé d'un caractère sur plusieurs octets (é = 2 octets) s'affiche « � ».
    """
    return repr(token.decode("utf-8", errors="replace"))


class BPETokenizer:
    """Tokenizer BPE sur les octets, avec pré-découpage par regex et tokens spéciaux."""

    def __init__(self):
        self.pattern = SPLIT_PATTERN
        # (id gauche, id droit) -> id du token fusionné. Un dict garde l'ordre d'insertion :
        # on le parcourt donc dans l'ordre où les fusions ont été apprises.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> octets du token
        self.special_tokens = {}  # texte du token spécial -> id, fixé par train() ou load()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Apprend vocab_size − 256 − len(SPECIAL_TOKENS) fusions sur `text`.

        `text` est lu comme du texte ordinaire : un « <|endoftext|> » écrit dedans n'est
        pas reconnu comme token spécial.
        """
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        if num_merges < 0:
            raise ValueError(f"vocab_size doit valoir au moins {256 + len(SPECIAL_TOKENS)}")
        # On repart de zéro : les 256 octets, aucune fusion, aucun token spécial.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}

        # 1. Pré-découpage, puis on regroupe les morceaux identiques : « que » apparaît des
        #    milliers de fois mais n'est traité qu'une fois, avec son nombre d'occurrences.
        chunk_counts = Counter(regex.findall(self.pattern, text))
        # words[j] = les octets du morceau n° j, counts[j] = son nombre d'occurrences.
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        # 2. Nombre d'occurrences de chaque paire de tokens voisins dans tout le corpus.
        pair_counts = Counter()
        for word, n in zip(words, counts):
            for pair in zip(word, word[1:]):  # paires voisines : (w0, w1), (w1, w2)…
                pair_counts[pair] += n

        for step in range(num_merges):
            if not pair_counts:
                raise ValueError("corpus trop petit : plus aucune paire à fusionner")
            # 3. La paire la plus fréquente. À égalité, la plus petite paire d'ids : une règle
            #    fixe, vérifiable à la main, au lieu de l'ordre du dictionnaire (une paire
            #    supprimée puis recréée passe à la fin). Python compare (-compte, paire)
            #    élément par élément.
            pair = min(pair_counts, key=lambda p: (-pair_counts[p], p))
            new_id = self._add_merge(pair)
            if verbose:
                print(f"fusion {step + 1}/{num_merges} : {render(self.vocab[pair[0]])} + "
                      f"{render(self.vocab[pair[1]])} -> {render(self.vocab[new_id])} "
                      f"({pair_counts[pair]} fois)")

            # 4. Seuls les mots qui contiennent la paire changent : on ne met à jour qu'eux.
            for j, word in enumerate(words):
                if pair[0] not in word:  # test rapide : sans pair[0], la paire est absente
                    continue
                new_word = merge(word, pair, new_id)
                if len(new_word) == len(word):  # pair[0] est là, mais pas la paire
                    continue
                for p in zip(word, word[1:]):  # on retire les paires de l'ancien mot…
                    pair_counts[p] -= counts[j]
                    if pair_counts[p] == 0:  # paire disparue : min() ne doit plus la voir
                        del pair_counts[p]
                for p in zip(new_word, new_word[1:]):  # … et on ajoute celles du nouveau
                    pair_counts[p] += counts[j]
                words[j] = new_word

        # 5. Les tokens spéciaux prennent les derniers ids (comme <|endoftext|> = 50256 dans GPT-2).
        for name in SPECIAL_TOKENS:
            new_id = len(self.vocab)
            self.special_tokens[name] = new_id
            self.vocab[new_id] = name.encode("utf-8")

    def _add_merge(self, pair):
        """Crée le token de la fusion `pair` (au prochain id libre) et renvoie son id."""
        new_id = len(self.vocab)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return new_id

    def _encode_chunk(self, ids):
        """Applique les fusions aux octets d'un morceau, dans l'ordre où elles ont été apprises."""
        while len(ids) >= 2:
            # Parmi les paires voisines, celle apprise le plus tôt (le plus petit id).
            # Une paire jamais apprise vaut l'infini.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # plus aucune fusion possible dans ce morceau
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Texte -> liste d'ids.

        Avec allow_special=True, « <|endoftext|> » écrit dans le texte devient son id.
        Par défaut, il est encodé comme du texte ordinaire : un texte ne peut pas
        glisser un vrai token spécial sans qu'on le demande.
        """
        if allow_special and self.special_tokens:
            # On coupe le texte autour des tokens spéciaux. Les parenthèses les gardent dans la
            # liste ; escape() car « | » a un sens dans une regex : « <|endoftext|> » en contient.
            special_pattern = "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]
        ids = []
        for part in parts:
            # Re-tester allow_special : sinon le texte « <|endoftext|> » seul deviendrait l'id spécial.
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in regex.findall(self.pattern, part):
                ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Liste d'ids -> texte. Une suite d'octets qui n'est pas de l'UTF-8 valide (le modèle
        peut en générer) devient « � » au lieu de faire planter le décodage."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        """Sauve en JSON : le motif, les tokens spéciaux et les fusions dans l'ordre appris.

        Avec vocab_size = 8192, le kit convertit ce fichier pour la compétition :
        python -m ftb.tokenizer from-minbpe --merges-json <fichier> --out tokenizer.json
        """
        data = {
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": list(self.merges),  # la fusion n° i (à partir de 0) crée le token 256 + i
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def save_minbpe(self, path):
        """Sauve au format .model de minbpe (Karpathy), celui que lit le kit de la compétition.

        Ligne 1 : « minbpe v1 » ; ligne 2 : le motif ; ligne 3 : le nombre de tokens spéciaux,
        puis une ligne « token id » par token spécial ; ensuite une fusion « a b » par ligne,
        dans l'ordre appris. Avec vocab_size = 8192 :
        python -m ftb.tokenizer from-minbpe --model <fichier> --out tokenizer.json
        """
        lines = ["minbpe v1", self.pattern, str(len(self.special_tokens))]
        lines += [f"{name} {i}" for name, i in self.special_tokens.items()]
        lines += [f"{a} {b}" for a, b in self.merges]  # ordre du dict = ordre d'apprentissage
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load(self, path):
        """Recharge un fichier écrit par save(). Renvoie self : tok = BPETokenizer().load(path)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.pattern = data["pattern"]
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b in data["merges"]:  # rejouées dans l'ordre : chaque token retrouve son id
            self._add_merge((a, b))
        self.special_tokens = data["special_tokens"]
        for name, i in self.special_tokens.items():
            self.vocab[i] = name.encode("utf-8")
        return self


def main():
    parser = argparse.ArgumentParser(description="Démo : entraîne le BPE sur Molière.")
    parser.add_argument("--corpus", default="corpus/moliere_train.txt")  # 25 pièces (kit)
    parser.add_argument("--val", default="corpus/moliere_val.txt")  # 3 pièces jamais vues
    parser.add_argument("--vocab", type=int, default=2048)
    args = parser.parse_args()
    text = Path(args.corpus).read_text(encoding="utf-8")
    val = Path(args.val).read_text(encoding="utf-8")

    tok = BPETokenizer()
    start = time.perf_counter()
    tok.train(text, args.vocab)
    print(f"{len(tok.merges)} fusions apprises en {time.perf_counter() - start:.1f} s "
          f"(vocabulaire : {len(tok.vocab)} tokens)")

    print("\nLes 20 premières fusions :")
    for (a, b), new_id in list(tok.merges.items())[:20]:
        print(f"  {new_id} = {render(tok.vocab[a])} + {render(tok.vocab[b])} -> {render(tok.vocab[new_id])}")

    print("\nCompression (1.00 octet par token sans aucune fusion) :")
    for name, t in [("entraînement", text), ("validation", val)]:
        n_bytes, n_tokens = len(t.encode("utf-8")), len(tok.encode(t))
        print(f"  {name} : {n_bytes} octets -> {n_tokens} tokens, "
              f"soit {n_bytes / n_tokens:.2f} octets par token")

    # L'Avare, acte III : une pièce de la validation, jamais vue à l'entraînement.
    line = "Il faut manger pour vivre, et non pas vivre pour manger."
    ids = tok.encode(line)
    print(f"\nRéplique : {line}")
    print("Tokens   : " + " | ".join(render(tok.vocab[i]) for i in ids))
    print(f"Ids      : {ids}  ({len(ids)} tokens pour {len(line.encode('utf-8'))} octets)")


if __name__ == "__main__":
    main()
