"""Modèle bigramme par comptage : le point de comparaison du mini-GPT.

Le bigramme prédit le symbole suivant en ne regardant que le symbole précédent.
Il n'apprend pas par gradient : il compte les paires (précédent → suivant) du
texte d'entraînement, puis transforme chaque ligne de comptes en probabilités.

Les symboles sont des entiers : des octets (0 à 255) ou des tokens BPE.
Le modèle ne sait pas lesquels : il ne voit que des ids.

Démo sur le Molière du kit officiel : uv run python -m minigpt.bigram
"""
from pathlib import Path

import torch

# Le kit officiel fournit le découpage : le BPB de validation du bloc 1 se calcule sur VAL_PATH,
# pour le bigramme comme pour le transformer. Même texte → comparaison juste.
TRAIN_PATH = Path("corpus/moliere_train.txt")  # 25 pièces, pour l'entraînement
VAL_PATH = Path("corpus/moliere_val.txt")  # 3 pièces tenues à l'écart : Le Misanthrope, L'Avare, Le Sicilien


class BigramModel:
    """Table probs[a, b] = P(le suivant est b | le précédent est a), estimée par comptage."""

    def __init__(self, vocab_size, alpha=1.0):
        self.vocab_size = vocab_size  # V : nombre de symboles possibles (256 pour les octets)
        self.alpha = alpha  # lissage : alpha est ajouté à chaque case avant de normaliser (alpha = 1 : Laplace)
        self.counts = None  # counts[a, b] : nombre de fois où b suit a (rempli par fit)
        self.probs = None  # probs[a, b] : probabilité que b suive a (rempli par fit)

    def fit(self, ids):
        """Compte les paires (ids[i], ids[i+1]), ajoute le lissage, normalise chaque ligne."""
        ids = torch.as_tensor(ids, dtype=torch.long)  # entiers 64 bits, le type standard des indices
        prev_ids, next_ids = ids[:-1], ids[1:]  # décalage d'un cran : la paire i est (ids[i], ids[i+1])
        self.counts = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.long)
        # Même résultat que « for a, b in zip(prev_ids, next_ids): counts[a, b] += 1 », mais en une
        # seule opération : des centaines de fois plus rapide que cette boucle Python.
        # accumulate=True additionne quand une paire revient ; sans lui, chaque case vue vaudrait 1.
        self.counts.index_put_((prev_ids, next_ids), torch.ones_like(prev_ids), accumulate=True)
        # Lissage : une paire jamais vue garde une probabilité > 0, donc -log2(p) reste fini ;
        # la ligne d'un symbole absent du train devient uniforme (1/V) au lieu de 0/0.
        # .double() passe en float64 (environ 16 chiffres significatifs, 7 en float32) : précaution gratuite.
        smoothed = self.counts.double() + self.alpha
        # keepdim=True garde la forme (V, 1) : chaque ligne est divisée par SA propre somme.
        self.probs = smoothed / smoothed.sum(dim=1, keepdim=True)
        return self  # permet d'écrire model = BigramModel(256).fit(ids)

    def nll_bits(self, ids):
        """Coût du texte en bits : somme des -log2 P(ids[i+1] | ids[i]).

        Le premier id n'a pas de précédent : il n'est pas prédit, donc pas compté.
        """
        ids = torch.as_tensor(ids, dtype=torch.long)
        # Deux listes d'indices donnent un vecteur, pas une sous-matrice : p[i] = probs[ids[i], ids[i+1]],
        # la probabilité que le modèle donnait au vrai suivant.
        p = self.probs[ids[:-1], ids[1:]]
        return -torch.log2(p).sum().item()

    def bpb(self, ids, n_bytes):
        """Bits par octet : coût total du texte en bits ÷ sa taille en octets UTF-8."""
        return self.nll_bits(ids) / n_bytes

    def sample(self, start_id, n, seed):
        """Génère n ids (start_id non compris) : chaque suivant est tiré selon la ligne du précédent."""
        generator = torch.Generator().manual_seed(seed)  # générateur à part, à seed fixe : même seed → même suite
        out, current = [], start_id
        for _ in range(n):
            # Tirage au sort d'un id, avec les probabilités de la ligne du symbole courant.
            current = torch.multinomial(self.probs[current], 1, generator=generator).item()
            out.append(current)
        return out


def main():
    torch.set_num_threads(4)  # le bloc 1 tourne sur CPU ; 4 threads laissent de la place aux autres chantiers
    # Niveau octets : chaque octet UTF-8 devient un id de 0 à 255 (donc nombre d'ids = nombre d'octets).
    train_ids = list(TRAIN_PATH.read_text(encoding="utf-8").encode("utf-8"))
    val_ids = list(VAL_PATH.read_text(encoding="utf-8").encode("utf-8"))
    model = BigramModel(vocab_size=256).fit(train_ids)

    print(f"Train : {TRAIN_PATH} ({len(train_ids)} octets) | validation : {VAL_PATH} ({len(val_ids)} octets)")
    print(f"BPB train      : {model.bpb(train_ids, len(train_ids)):.4f}")
    print(f"BPB validation : {model.bpb(val_ids, len(val_ids)):.4f}  (repère : 8 = hasard uniforme sur 256 octets)")
    # On part d'un saut de ligne (« \n » = octet 10), comme en début de ligne.
    # Un caractère UTF-8 fait au plus 4 octets : 4 × 300 octets donnent au moins 300 caractères.
    sample = bytes(model.sample(start_id=ord("\n"), n=4 * 300, seed=0))  # liste d'ids 0-255 → octets
    print("Échantillon (300 caractères) :")
    # errors="replace" : un octet qui ne forme pas un caractère valide (un « é » coupé en deux…)
    # s'affiche « � » au lieu de faire planter le décodage. Ces « � » viennent surtout du lissage.
    print(sample.decode("utf-8", errors="replace")[:300])


if __name__ == "__main__":
    main()
