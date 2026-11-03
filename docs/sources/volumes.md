# Volumes collectés (`data/raw/`)

Généré par `uv run python -m pipeline.collect stats` le 2026-11-04 00:32. Mo de texte = octets UTF-8 du champ `text` (le HTML brut de Common Crawl n'est pas compté). Période = plus petite et plus grande valeur du champ `date` (gel au 2026-10-16).

| Source | Domaine | État | Documents | Mo de texte | Période couverte | Fichiers | Disque (Mo, zstd) |
|---|---|---|---:|---:|---|---:|---:|
| `assemblee_nationale` | dialogue | complet | 1 550 | 317.8 | 2017-07-06 → 2026-10-16 | 5 | 89 |
| `wikipedia_fr` | encyclopedique | complet | 2 564 645 | 7 814.2 | 2023-11-01 → 2023-11-01 | 87 | 2 753 |
| `wikisource_fr` | litteraire | complet | 21 597 | 5 343.8 | 2007-09-11 → 2026-10-01 | 73 | 1 840 |
| `gutenberg_fr` | litteraire | complet | 4 228 | 1 786.4 | 1997-01-01 → 2026-10-15 | 43 | 624 |
| `wikinews_fr` | presse | complet | 20 419 | 32.1 | 2005-01-28 → 2026-04-28 | 1 | 11 |
| `commoncrawl` | web | complet | 11 865 | 85.6 | 2026-09-04 → 2026-09-04 | 10 | 349 |
| `fineweb2_fra` | web | complet | 522 923 | 1 781.0 | 2013-05-18 → 2024-04-25 | 22 | 705 |
| **Total** | | | **3 147 227** | **17 160.9** | | | **6 372** |

Budget disque de `data/raw/` : 6.4 Go utilisés sur 30 Go.

Corpus Molière provisoire (`data/moliere/moliere.txt`, remplacé par `corpus/` du kit) : 33 pièces, 53 105 lignes, 2 273 990 caractères.

Détail par source (origine, licence, risques) : voir les fiches `docs/sources/<source>.md`.
