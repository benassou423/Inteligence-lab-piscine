# Feed the Beast : kit de départ

Kit officiel de la piscine I Lab **« Feed the Beast »** (Intelligence Lab, ECE Paris).
Contact et entraide : jeanbaptiste.caron@edu.ece.fr (par mail ou en visio).

> **Le modèle est le même pour tout le monde. La différence, c'est la donnée.**

Ce dépôt contient tout ce dont tu as besoin pour les trois blocs du sujet :

- **Bloc 1, mini-GPT** : le corpus Molière, découpé en entraînement et validation.
- **Bloc 2, pipeline** : les outils pour transformer tes textes en données d'entraînement mesurables.
- **Bloc 3, compétition** : la configuration officielle du modèle, le script d'entraînement officiel,
  le script d'évaluation (BPB), le set de validation public et l'outil de soumission.

## Démarrage rapide

Ton dépôt de travail est celui que tu as lié à la plateforme : copie-y le contenu de ce kit (ou crée-le
avec le bouton *Use this template* de GitHub). Tes soumissions y vivent.

```bash
git clone https://github.com/Meuleur/Piscine.git && cd Piscine
pip install -r requirements.txt

# 1. Un dataset d'exemple (baseline naïve, ~450 Mo de texte)
python examples/make_example_dataset.py --out example_data/

# 2. Un tokenizer (8 192 tokens) et le flux de tokens
python -m ftb.tokenizer train --inputs example_data/example.jsonl --out tokenizer.json
python -m ftb.prepare --inputs example_data/example.jsonl --tokenizer tokenizer.json --out prepared/

# 3. L'entraînement officiel (GPU : moins d'une heure sur un T4) puis l'évaluation
python -m ftb.train --data prepared/ --tokenizer tokenizer.json --out runs/baseline
python -m ftb.evaluate --ckpt runs/baseline/ckpt.pt --tokenizer runs/baseline/tokenizer.json

# 4. La soumission
python -m ftb.submit --run runs/baseline --participant "Prénom Nom"
git add submissions/ && git commit -m "soumission" && git push
```

Sur Colab ou Kaggle : ouvre `notebooks/quickstart_colab.ipynb` (tout ce qui précède, plus la reprise
d'un entraînement interrompu). Pour vérifier la chaîne en quelques minutes sur CPU, ajoute
`--debug-budget 131072` à `ftb.train` (le run n'est alors pas soumissible).

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `corpus/moliere_train.txt` | 25 pièces de Molière (1,5 Mo), pour le bloc 1 |
| `corpus/moliere_val.txt` | 3 pièces tenues à l'écart (Le Misanthrope, L'Avare, Le Sicilien), pour ta BPB du bloc 1 |
| `corpus/moliere_manifest.json`, `corpus/build_moliere.py` | Source, licence, liste des pièces, script de reconstruction |
| `config/official.json` | Configuration officielle : modèle, hyperparamètres, budget. **Non modifiable** |
| `data/validation.jsonl` | Set de validation public (4 genres, voir plus bas) |
| `data/VALIDATION_SOURCES.md` | Sources et licences de chaque document de validation |
| `ftb/model.py` | Le modèle officiel (~10,2M paramètres) |
| `ftb/tokenizer.py` | Format du tokenizer, vérifications, entraînement rapide, export depuis ton BPE maison |
| `ftb/data.py` (`python -m ftb.prepare`) | Tes textes -> `train.bin` + empreinte du dataset |
| `ftb/train.py` | L'entraînement officiel |
| `ftb/evaluate.py` | Le calcul officiel du BPB |
| `ftb/contamination.py` | Recouvrement entre tes données et un jeu d'évaluation |
| `ftb/submit.py` | Prépare une soumission |
| `examples/make_example_dataset.py` | Baseline : un mélange naïf, à battre |
| `notebooks/quickstart_colab.ipynb` | Tout le démarrage sur un GPU T4 gratuit |
| `tests/` | `python -m pytest tests/ -q` |
| `LEADERBOARD.md` | Le classement, mis à jour par le mentor |

## Bloc 1 : le corpus Molière

Source : *Œuvres complètes de J.-B. Poquelin Molière*, édition Philarète Chasles (1888), tomes 1 à 4,
numérisés par le Projet Gutenberg. Domaine public. Les notices et notes de l'éditeur sont retirées ;
il reste les pièces, avec leurs listes de personnages et leurs didascalies. Les pièces sont séparées
par une ligne `=====`.

Ton mini-GPT, ton BPE et ton bigramme sont **les tiens** : `ftb/model.py` est la référence de la
compétition, pas un modèle à recopier pour le bloc 1. Pour la « BPB sur la validation » du bloc 1,
calcule-la toi-même sur `corpus/moliere_val.txt` avec la formule de l'encadré ci-dessous.

## Bloc 3 : les règles de la compétition

### Ce qui est figé

| Paramètre | Valeur |
|---|---|
| Modèle | GPT décodeur, 6 couches, 5 têtes, d_model 320, MLP x4, contexte 512, LayerNorm pré-bloc, embeddings liés : **10 166 400 paramètres** |
| Budget | **100 millions de tokens vus** : 3 052 pas de 64 séquences de 512 tokens (100 007 936 tokens) |
| Optimiseur | AdamW, β = (0,9 ; 0,95), weight decay 0,1, clip 1,0 |
| Learning rate | 2e-3, warmup linéaire de 150 pas, décroissance cosinus jusqu'à 2e-4 |
| Graine | 1337 |
| Précision | float16 + GradScaler sur tout GPU (comme le ré-entraînement de vérification, sur T4) ; float32 sur CPU |

Le budget est compté **en tokens**, pas en octets : un tokenizer qui compresse bien ton français fait
voir plus de texte au modèle pour le même budget. C'est un des leviers.

Options autorisées pour un run officiel : `--micro-batch` (accumulation de gradient, même résultat si
la mémoire manque), `--compile` (plus rapide sur T4, pas sur P100), `--resume` (reprise après une
coupure de Colab, depuis le dernier point de sauvegarde, tous les 500 pas). Un dossier `--out` qui
contient déjà un run terminé est refusé : rien n'est écrasé par erreur.

### Le tokenizer

Libre, avec trois règles vérifiées par le kit :

1. **Format** : un fichier `tokenizer.json` de la librairie Hugging Face `tokenizers`.
2. **Exactement 8 192 tokens**, token spécial `<|endoftext|>` compris (il sépare les documents).
3. **Sans perte** : `decode(encode(texte)) == texte`, octet pour octet. Pas de normalisation
   (minuscules, NFKC, suppression d'espaces) : le BPB ne serait plus comparable.
4. **Déterministe** : ni BPE-dropout, ni padding, ni troncature (sinon l'empreinte du dataset ne se
   reproduit pas).

Deux façons de l'obtenir :

```bash
# Rapide : BPE byte-level de la librairie tokenizers, motif de découpage de GPT-4
python -m ftb.tokenizer train --inputs mes_donnees/*.jsonl --out tokenizer.json

# Ton BPE maison du bloc 1 (format .model de minbpe : 256 octets + 7 935 fusions)
python -m ftb.tokenizer from-minbpe --model mon_bpe.model --out tokenizer.json

python -m ftb.tokenizer check --tokenizer tokenizer.json   # règles + octets par token sur la validation
```

La conversion depuis minbpe est testée : elle encode exactement comme l'implémentation de Karpathy
(`tests/test_kit.py`). Entraîner ton propre BPE sur tes données compte dans la soutenance.

### Tes données et leur ordre

```bash
python -m ftb.prepare --inputs data/a.jsonl data/b.jsonl --tokenizer tokenizer.json --out prepared/
```

- Formats : `.jsonl` (champ `text`), `.jsonl.gz`, `.txt` (un fichier = un document), `.parquet`.
- Chaque document est suivi de `<|endoftext|>`. **L'ordre est conservé** : fichiers dans l'ordre
  donné, lignes dans l'ordre du fichier. C'est toi qui mélanges, répètes et ordonnes.
- L'entraînement lit le flux dans l'ordre, par fenêtres de 50 pas (~1,6M tokens) mélangées en
  interne avec une graine fixe : tes choix d'ordre (garder le meilleur pour la fin, par exemple)
  sont respectés à cette échelle.
- Ce sont exactement les 100 007 936 premiers tokens du flux qui sont vus, ni plus ni moins. Si ton
  flux est plus court, il est relu depuis le début : mieux vaut choisir toi-même ce que tu répètes.
- `prepared/data_meta.json` contient l'**empreinte du dataset** (`dataset_sha256`, SHA-256 du flux de
  tokens) : c'est le hash demandé dans les soumissions.

Données synthétiques : 20 % des tokens au maximum, générées par un modèle open weights, prompts
publiés dans ton dépôt. Interdit : initialiser depuis des poids pré-entraînés, distiller les logits
d'un autre modèle, chercher à reconstituer le test.

## L'évaluation : bits par octet

> `BPB = (somme sur les tokens de -log2 p(token | contexte)) / (nombre d'octets UTF-8 du texte)`
>
> Avec une loss moyenne en nats : `BPB = loss × nb_tokens / (ln(2) × nb_octets)`. Plus c'est bas, mieux c'est.

```bash
python -m ftb.evaluate --ckpt runs/x/ckpt.pt --tokenizer runs/x/tokenizer.json [--data autre.jsonl]
```

- Chaque document est évalué séparément, précédé de `<|endoftext|>`.
- Un document plus long que le contexte est lu par fenêtres glissantes au pas de 256 : chaque token
  est prédit avec au moins 256 tokens de contexte (sauf au début du document).
- BPB global = somme des bits de tous les documents / somme de leurs octets. Le BPB par genre est
  donné pour t'aider à diagnostiquer ton mélange.
- Le script refuse un tokenizer qui n'est pas sans perte ou un modèle hors configuration officielle.

### Le set de validation public

`data/validation.jsonl` : 230 documents, 813 Ko, quatre genres à parts égales en octets.
Le set caché du leaderboard et le test final ont **la même composition**, avec d'autres documents.

| Genre | Source | Licence | Documents | Période |
|---|---|---|---|---|
| Encyclopédique | Wikipédia FR, articles créés dans la période | CC BY-SA 4.0 | 55 (200 Ko) | 2026-10-24 → 2026-10-31 |
| Littéraire | Projet Gutenberg, livres FR mis en ligne dans la période (extraits) | Domaine public | 35 (205 Ko) | 2026-07-17 → 2026-10-15 |
| Presse | Wikinews FR, articles publiés | CC BY 2.5 | 104 (201 Ko) | 2024-01-01 → 2026-03-31 |
| Dialogue | Assemblée nationale, comptes rendus des séances (extraits) | Licence Ouverte | 36 (206 Ko) | 2026-10-01 → 2026-10-30 |

Chaque ligne : `{"id", "genre", "source", "title", "url", "license", "date", "text"}`.
Les extraits longs sont coupés à ~6 000 caractères sur une fin de paragraphe. Wikinews est fermé
depuis mai 2026 : la presse de la validation date de 2024 à 2026, celle du test viendra d'autres
sources de presse récentes.

Le **test final** est construit de la même façon avec des textes **publiés après le début de la
piscine** : impossible de les avoir dans tes données, même par accident. La validation, elle, est
publique et faite de textes récents qui peuvent se retrouver dans des collectes web : vérifie que
tu ne t'entraînes pas dessus, sinon ton seul repère devient faux.

```bash
python -m ftb.contamination --inputs mes_donnees/*.jsonl            # contre data/validation.jsonl
```

L'orga lance le même contrôle (13-grammes de mots) sur le dataset des meilleurs contre le test caché.

## Soumettre

```bash
python -m ftb.submit --run runs/mon_run --participant "Prénom Nom" \
    --dataset-url https://huggingface.co/datasets/<toi>/<ton-dataset> --note "filtre qualité v2"
git add submissions/ && git commit -m "soumission" && git push
```

`submissions/<date>_<heure>/` contient le checkpoint, le tokenizer, les logs d'entraînement,
`data_meta.json` (empreinte du dataset), les 5 échantillons générés et `submission.json`. Les poids
sont stockés en float16 (~25 Mo par soumission). Le script
refuse un run de mise au point, un hyperparamètre modifié ou un budget différent.

- **2 soumissions par jour** (heure de Paris) au maximum ; au-delà, elles ne sont pas évaluées.
- Le mentor relève les soumissions poussées sur ton dépôt, les évalue sur le **set caché du
  leaderboard** et met à jour `LEADERBOARD.md`.
- `--dataset-url` : où l'orga peut récupérer ton dataset final pour la vérification. Un dataset
  publié sur Hugging Face avec une vraie dataset card concourt pour le prix bonus.

## Vérification et classement final

1. Le classement final se fait sur le **test caché** (même composition, textes publiés après le
   début de la piscine), avec ta meilleure soumission.
2. Les meilleurs participants sont **ré-entraînés par l'orga** sur un T4, avec ce kit, à partir de
   leur dataset (`ftb.prepare` puis `ftb.train`). L'empreinte du dataset doit être identique à celle
   de la soumission, et le BPB obtenu à ±0,02 près du tien. Un écart inexpliqué disqualifie. Les
   versions de `torch` et `tokenizers` que tu as utilisées sont enregistrées dans le checkpoint et
   `data_meta.json` : l'orga reprend les mêmes.
3. Les échantillons de chaque modèle sont générés par l'orga et notés à l'aveugle (15 % de la note).

## Crédits et licences

- Code du kit : Intelligence Lab, ECE Paris, 2026.
- Corpus Molière : domaine public, d'après le Projet Gutenberg (ebooks 40086, 43535, 50173, 57270).
- Set de validation : voir `data/VALIDATION_SOURCES.md`. Articles de Wikipédia (CC BY-SA 4.0) et de
  Wikinews (CC BY 2.5) : auteurs dans l'historique de chaque page, URL fournie pour chaque document ;
  textes du Projet Gutenberg (domaine public) ; comptes rendus de l'Assemblée nationale (open data,
  Licence Ouverte). Les extraits sont reproduits sans autre modification que la mise en forme texte.
