# Preuves des jugements à l'aveugle du chatbot Molière

Les fiches du chatbot donnent des chiffres du genre « 100 réponses sans défaut sur 144 ». Ce dossier
contient ce qu'il faut pour les recompter : les réponses jugées, la clé qui dit quel modèle a écrit quoi, les
notes de chaque juge et le dépouillement.

Il contient aussi les contrôles de qualité des conversations de synthèse : le fichier lu par les lecteurs et
leurs notes.

Ce dossier prouve que les chiffres publiés sortent bien de ces notes. Il ne prouve pas que les juges ont
raison : voir les limites au § 7.

- La consigne donnée aux juges et aux lecteurs, et la façon dont les réponses sont mélangées :
  [`CONSIGNE_DES_JUGES.md`](CONSIGNE_DES_JUGES.md).
- Le programme qui mélange les réponses et compte les votes : [`aveugle.py`](aveugle.py).
- Deux outils de recomptage, qui ne chargent pas PyTorch : [`recompter_jugements.py`](recompter_jugements.py)
  (F, R, S, T par modèle, pour chaque sous-dossier) et [`apparie.py`](apparie.py) (comparaison appariée
  avec un modèle témoin et test des signes).
- Le récit daté des essais : [`docs/chatbot/HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md).

## 1. Les neuf jugements

Un jugement compare plusieurs modèles sur un jeu de messages. Chaque message est joué avec 3 graines (11,
22, 33). Les jeux « reglage » et « test » ont 24 tours de parole, soit 72 réponses par modèle ; les jeux
« neuf » et « neuf2 » en ont 48, soit 144 réponses par modèle. Les messages sont dans
[`tools/nuit_moliere.py`](../../../tools/nuit_moliere.py).

| Sous-dossier | Date | Jeu | Modèles | Réponses | Juges |
| --- | --- | --- | --- | ---: | --- |
| `synthese_qwen_reglage` | 14/11/2026 | reglage | A0, B, C | 216 | 3 par lot, 3 lots |
| `synthese_qwen_test` | 14/11/2026 | test | A0, C | 144 | 3 par lot, 2 lots |
| `decodage_reglage` | 14/11/2026 | reglage | A0, V1, V2, V3 | 288 | 3 par lot, 4 lots |
| `decodage_test` | 14/11/2026 | test | A0, V2 | 144 | 3 par lot, 2 lots |
| `donnees_nettoyees_reglage` | 14/11/2026 | reglage | A0, D, F | 216 | 3 par lot, 3 lots |
| `pretrain_filtre_reglage` | 14/11/2026 | reglage | A0, E | 144 | 3 par lot, 2 lots |
| `synthese_qwen_ancre_reglage` | 14/11/2026 | reglage | A0, C | 144 | 3 par lot, 2 lots |
| `campagne_v4_neuf` | 15/11/2026 | neuf | A0, S03, S05, S10, S15 | 720 | 3 par lot, 10 lots |
| `final_neuf2` | 16/11/2026 | neuf2 | A0, S03t, s03, s05, s10, s15 | 864 | 3 par lot, 12 lots |

Les modèles :

- **A0** : le modèle du lancement du 13/11/2026, affiné sans synthèse. Ses rapports sont dans
  [`../20261113_1956_M/`](../20261113_1956_M/).
- **B** et **C** : le même affinage que A0, avec en plus la première synthèse (générateur `qwen3:8b`), à
  2,87 % et 1,16 % des tokens vus (`chatbot/recettes_rtx_v2.json`, `chatbot/recettes_rtx_v3.json`).
- **V1**, **V2**, **V3** : le modèle A0, avec une autre façon de choisir la réponse. V1 : ancrage sur le
  dernier message, 16 candidates. V2 : ancrage, 32 candidates. V3 : 32 candidates sans ancrage.
- **D** : théâtre d'affinage nettoyé (`chatbot/recettes_rtx_v4a.json`). **F** : pré-entraînement filtré et
  théâtre nettoyé (`chatbot/recettes_rtx_v4.json`). **E** : pré-entraînement filtré et théâtre d'origine
  (`chatbot/recettes_rtx_v4e.json`).
- **S03**, **S05**, **S10**, **S15** : le même affinage que A0, avec en plus la synthèse v4. S03 en a vu
  3,02 % (part visée : 3 %) ; pour S05, S10 et S15, les parts visées sont 5, 10 et 15 %
  (`chatbot/recettes_rtx_v5_s03.json`, `_s05`, `_s10`, `_s15`).
- **S03t**, **s03**, **s05**, **s10**, **s15** (jugement `final_neuf2`, noms tels qu'ils sont dans la clé).
  **S03t** est le témoin : le modèle S03 ci-dessus, qui est le modèle publié. **s03**, **s05**, **s10**,
  **s15** sont les quatre candidats « v6 » : le même affinage que A0, avec en plus la synthèse v4 + v5
  (27 624 conversations) à 3, 5, 10 et 15 % (`chatbot/recettes_rtx_v6_s03.json`, `_s05`, `_s10`, `_s15`). Les
  quatre candidats ne sont pas publiés ; leurs recettes et leurs réponses jugées le sont. **A0** est le même
  modèle que ci-dessus.

Le choix de la réponse, selon le jugement :

- `synthese_qwen_reglage`, `synthese_qwen_test` : ancien choix pour tous (16 candidates, sans ancrage).
- `decodage_reglage`, `decodage_test` : ancien choix pour A0 ; V1, V2, V3 comme ci-dessus.
- les cinq autres, dont `final_neuf2` : choix adopté le 14/11/2026 pour tous (ancrage, 32 candidates).

## 2. Chiffre publié, preuve, recomptage

Les appels `total(...)` et `signes(...)` sont les fonctions du § 3. « DR + DT » veut dire les deux dossiers
`decodage_reglage` et `decodage_test` ensemble ; « SR + ST », `synthese_qwen_reglage` et `synthese_qwen_test`.
Pour les lignes du 16/11/2026 (`final_neuf2`), la colonne « Recomptage » donne deux commandes, qui se lancent
depuis la racine du dépôt (sur le PC Windows du projet : `PYTHONUTF8=1 .venv/Scripts/python.exe` à la place de
`python`) :

- `RECOMPTER` = `python experiments/chatbot_rtx/jugements/recompter_jugements.py`. Elle affiche, pour chaque
  sous-dossier, une ligne par modèle (`n`, `F`, `R`, `S`, `T`). Elle sert aussi aux lignes de `campagne_v4_neuf`
  et des autres sous-dossiers.
- `APPARIE` = `python experiments/chatbot_rtx/jugements/apparie.py experiments/chatbot_rtx/jugements/final_neuf2 S03t`.
  Elle compare chaque modèle au témoin S03t, réponse par réponse (test des signes bilatéral).

Les deux ne chargent pas PyTorch et tournent sur le processeur.

Fiches : FM = [`FICHE_MODELE.md`](../../../docs/chatbot/FICHE_MODELE.md), FD =
[`FICHE_DONNEES.md`](../../../docs/chatbot/FICHE_DONNEES.md), CM =
[`chatbot_moliere.md`](../../../docs/comprendre/chatbot_moliere.md), H =
[`HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md), LM = [`README.md`](../../../README.md) de la racine.

| Chiffre publié | Où | Sous-dossier | Recomptage |
| --- | --- | --- | --- |
| A0 : français 119, rapport 110, style 124, sans défaut 82, sur 144 | FM § 6.2, CM § 6, H ; LM (82 seulement) | `campagne_v4_neuf` | `total(["campagne_v4_neuf"], "A0")`, ou la ligne `A0` de `RECOMPTER` |
| S03 : 124, 114, 138, sans défaut 100, sur 144 | FM § 2 et § 6.2, CM § 4 et § 6, H, LM | `campagne_v4_neuf` | `total(["campagne_v4_neuf"], "S03")` |
| S03 meilleur 45 fois, moins bon 27 fois, p = 0,044 | FM § 6.2, CM § 6, H | `campagne_v4_neuf` | `signes(["campagne_v4_neuf"], "A0", "S03")` rend `(45, 27, 0.0444)` |
| Conversations de plusieurs tours : 57 sans défaut sur 72 contre 38 | FM § 6.2, CM § 6, H | `campagne_v4_neuf` | `total(["campagne_v4_neuf"], "S03", lambda job, tour: job >= 24)`, puis de même avec `"A0"` |
| Messages isolés : 44 puis 43 sans défaut sur 72 | H | `campagne_v4_neuf` | même appel avec `job < 24` |
| S05 97, S10 88, S15 100 sans défaut sur 144 | H | `campagne_v4_neuf` | `total(["campagne_v4_neuf"], "S05")`, etc. |
| Par juge : 83 → 102, 83 → 101, 80 → 98 | H | `campagne_v4_neuf` | dernières lignes du § 3 |
| « Certes » en tête de 28 réponses de S03 sur 240, « Hélas » de 24 ; A0 : 0 et 1 | FM § 6.3, CM § 7, H | `campagne_v4_neuf` | § 3 ; le « 2 » d'abord écrit pour A0 est corrigé : voir § 8 |
| Ancrage : rapport au message 96 → 114, français 125 → 120, style 137 → 133, sur 144 | FM § 5, CM § 5, H ; LM (96 → 114 seulement) | DR + DT | `total(DR + DT, "A0")` puis `total(DR + DT, "V2")` |
| Ancrage, tours 2 et suivants : rapport 29 → 43 sur 54 | CM § 5, H | DR + DT | `total(DR + DT, "V2", lambda job, tour: job >= 12 and tour >= 1)`, puis `"A0"` |
| Ancrage seul à 16 candidates : + 2 sur 72 ; 32 candidates sans ancrage : − 3 sur 72 | CM § 5 | `decodage_reglage` | `total(DR, "A0")["R"]`, puis `"V1"` et `"V3"` : 47, 49, 44 (ou les lignes `A0`, `V1`, `V3` de `RECOMPTER`) |
| 33 réponses meilleures, 15 moins bonnes, p = 0,0133 | H | DR + DT | `signes(DR + DT, "A0", "V2", "R")` |
| Référence : 44 sans défaut sur 72 | FD § 6, CM § 7, H | `donnees_nettoyees_reglage` | `total(["donnees_nettoyees_reglage"], "A0")` |
| Théâtre nettoyé : 34 sans défaut | FD § 6, CM § 7, H | `donnees_nettoyees_reglage` | même appel avec `"D"` |
| Pré-entraînement filtré et théâtre nettoyé : 37 | FD § 6, CM § 7, H | `donnees_nettoyees_reglage` | même appel avec `"F"` |
| Pré-entraînement filtré et théâtre d'origine : 38 (A0 : 44) | FD § 6, CM § 7, H | `pretrain_filtre_reglage` | `total(["pretrain_filtre_reglage"], "E")` |
| Synthèse `qwen3:8b` à 1,16 % : 45 sans défaut contre 44 ; français 55 contre 62 | FD § 6, CM § 7, H | `synthese_qwen_ancre_reglage` | `total(["synthese_qwen_ancre_reglage"], "C")` puis `"A0"` |
| Essai B : rapport 48 comme A0, style 61 contre 69 | H | `synthese_qwen_reglage` | `total(SR, "B")` puis `"A0"` |
| Essai C, jeu de réglage : 65, 53, 70 contre 63, 48, 69 ; sans défaut 49 contre 44 | H | `synthese_qwen_reglage` | `total(SR, "C")` puis `"A0"` |
| Essai C, réglage et test : rapport 94 → 102, style 135 → 142, français 126 et 126, sans défaut 79 → 92 | H | SR + ST | `total(SR + ST, "A0")` puis `"C"` |
| Contrôle de la première synthèse : 38 sans défaut sur 60 (seuil 48) | H | `controle_synthese_qwen` | § 4 |
| Contrôle du pilote des consignes v4 : 48 sur 60 (seuil 48) | H | `controle_synthese_v4_pilote` | § 4 |
| Contrôle du fichier v4 : 113 sur 120 (seuil 96) | FD § 4.3, H | `controle_synthese_v4` | § 4 |
| Contrôle des premières conversations v5 : 52 sur 60 (seuil 48) | H | `controle_synthese_v5_debut` | § 4 |
| Contrôle du fichier v4 + v5 : 109 sur 120 (seuil 96) | H, FD | `controle_synthese_v4_v5` | § 4 |
| Jugement final du 16/11 : A0 73 sans défaut sur 144 (français 113, rapport 94, style 127) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `A0` de `final_neuf2` |
| S03 (témoin `S03t`) : 96 (124, 111, 142) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `S03t` |
| Candidat à 3 % (`s03`) : 93 (130, 105, 139) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `s03` |
| Candidat à 5 % (`s05`) : 101 (133, 109, 144) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `s05` |
| Candidat à 10 % (`s10`) : 88 (132, 104, 134) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `s10` |
| Candidat à 15 % (`s15`) : 105 (128, 118, 141) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `RECOMPTER`, ligne `s15` |
| A0 contre S03 : meilleur 25 fois, moins bon 48 fois, p = 0,0095 | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `APPARIE`, ligne `A0` : `meilleur 25, moins bon 48, écart -23, p = 0.0095` |
| Candidats contre S03 : 28 / 31 (p = 0,79), 36 / 31 (0,63), 25 / 33 (0,36), 31 / 22 (0,27) | FM § 6.2, CM § 6, H, LM ; ce fichier, § 10 | `final_neuf2` | `APPARIE`, lignes `s03`, `s05`, `s10`, `s15` |
| Gardes (BPB sur Molière, par-cœur) des six modèles | ce fichier, § 10 | `final_neuf2` | fichier `gardes.json` : lire les champs `bpb_moliere_val`, `par_coeur`, `garde_bpb`, `garde_par_coeur` |
| Accord unanime des 3 juges : F 829/864, R 793/864, S 839/864 | ce fichier, § 10 | `final_neuf2` | fichier `notes_finales.json`, `resume` → `accord_unanime` |

Tous les chiffres des 14 et 15/11 ont été recomptés le 16/11/2026 à partir des fichiers de ce dossier, ainsi
que ceux du jugement final du 16/11 (§ 10). Un seul écart avec les fiches : le « 2 » donné d'abord pour
les « Hélas » de A0 (§ 8).

## 3. Recompter un jugement

Les commandes se lancent depuis la racine du dépôt. Python suffit, sans PyTorch ni autre bibliothèque
(essayé avec Python 3.12 ; les lignes de Python plus bas demandent Python 3.8 ou plus). Sur le PC Windows du
projet : `.venv/Scripts/python.exe` à la place de `python`, avec `PYTHONUTF8=1` devant.

**Refaire le vote à la majorité.** La commande écrit un nouveau `notes_finales.json` dans un dossier à part
et affiche le résumé (sommes F, R, S par modèle, accord entre juges) :

```bash
J=experiments/chatbot_rtx/jugements
mkdir -p recompte && cp $J/campagne_v4_neuf/cle.json recompte/
python $J/aveugle.py agrege --notes $J/campagne_v4_neuf/notes.json --out recompte
```

Pour vérifier que le résultat est celui du dossier (la commande doit afficher `True`) :

```bash
python -c "import json,sys; a,b=[json.load(open(p,encoding='utf-8')) for p in sys.argv[1:]]; print(a==b)" recompte/notes_finales.json $J/campagne_v4_neuf/notes_finales.json
```

Le dossier `recompte` peut ensuite être supprimé. La comparaison porte sur le contenu : selon la machine,
les fins de ligne du fichier écrit peuvent différer.

**Compter les réponses sans défaut, les sous-ensembles et le test des signes.** À coller dans `python` :

```python
import json, math

J = "experiments/chatbot_rtx/jugements/"
DR, DT = ["decodage_reglage"], ["decodage_test"]
SR, ST = ["synthese_qwen_reglage"], ["synthese_qwen_test"]

def notes(dossiers):
    """(dossier, modèle, message, graine, tour) -> notes à la majorité, avec T = sans défaut."""
    n = {}
    for d in dossiers:
        f = json.load(open(J + d + "/notes_finales.json", encoding="utf-8"))
        for ident, v in f["notes"].items():
            c = f["cle"][ident]
            n[(d, c["modele"], c["job"], c["graine"], c["tour"])] = dict(v, T=int(v["F"] and v["R"] and v["S"]))
    return n

def total(dossiers, modele, garde=lambda job, tour: True):
    """Nombre de réponses d'un modèle et sommes F, R, S, T."""
    t = {"n": 0, "F": 0, "R": 0, "S": 0, "T": 0}
    for (d, m, job, graine, tour), v in notes(dossiers).items():
        if m == modele and garde(job, tour):
            t["n"] += 1
            for c in "FRST":
                t[c] += v[c]
    return t

def signes(dossiers, a, b, critere="T"):
    """Réponse par réponse : fois où b est meilleur que a, fois où il est moins bon, p du test des signes."""
    n = notes(dossiers)
    plus = sum(1 for (d, m, *k), v in n.items() if m == b and v[critere] > n[(d, a, *k)][critere])
    moins = sum(1 for (d, m, *k), v in n.items() if m == b and v[critere] < n[(d, a, *k)][critere])
    p = min(1.0, 2 * sum(math.comb(plus + moins, j) for j in range(min(plus, moins) + 1)) / 2 ** (plus + moins))
    return plus, moins, round(p, 4)

print(total(["campagne_v4_neuf"], "S03"))   # {'n': 144, 'F': 124, 'R': 114, 'S': 138, 'T': 100}
print(total(["campagne_v4_neuf"], "A0"))    # {'n': 144, 'F': 119, 'R': 110, 'S': 124, 'T': 82}
print(signes(["campagne_v4_neuf"], "A0", "S03"))   # (45, 27, 0.0444)
print(total(DR + DT, "A0"), total(DR + DT, "V2"))  # R : 96 puis 114

# « Certes » et « Hélas » en tête de réponse, sur les 240 réponses (5 graines) de A0 et de S03
for nom in ("A0", "S03"):
    r = [json.loads(l) for l in open(J + "campagne_v4_neuf/reponses_5_graines_" + nom + ".jsonl", encoding="utf-8")]
    print(nom, len(r), sum(x["reply"].startswith("Certes") for x in r), sum(x["reply"].startswith("Hélas") for x in r))
# A0 240 0 1
# S03 240 28 24

# Sans défaut par juge (le « juge 1 » réunit les juges n° 1 des 10 lots)
cle = json.load(open(J + "campagne_v4_neuf/cle.json", encoding="utf-8"))
for juge in json.load(open(J + "campagne_v4_neuf/notes.json", encoding="utf-8")):
    print([sum(v["F"] and v["R"] and v["S"] for i, v in juge.items() if cle[i]["modele"] == m) for m in ("A0", "S03")])
# [83, 102]
# [83, 101]
# [80, 98]
```

Dans `garde`, `job` est le numéro du message dans le jeu et `tour` le tour de parole (0 pour le premier).
Jeux « reglage » et « test » : messages isolés 0 à 11, conversations de 4 tours 12 à 14. Jeux « neuf » et
« neuf2 » : messages isolés 0 à 23, conversations de 4 tours 24 à 29.

Pour le jugement du 16/11/2026, les deux commandes du début du § 2 (`RECOMPTER` et `APPARIE`) font le même
travail sans rien coller dans `python` ; `apparie.py` accepte en plus un critère en troisième argument (`F`,
`R` ou `S` ; `T` par défaut).

## 4. Les contrôles de qualité de la synthèse

Trois lecteurs lisent un tirage au hasard de conversations de synthèse. Le seuil est fixé avant la lecture.
La consigne et la règle de dépouillement sont dans [`CONSIGNE_DES_JUGES.md`](CONSIGNE_DES_JUGES.md), § 5.

| Sous-dossier | Date | Fichier contrôlé | Lues | Sans défaut | Seuil |
| --- | --- | --- | ---: | ---: | ---: |
| `controle_synthese_qwen` | 14/11/2026 | première synthèse (`qwen3:8b`), 1 305 conversations | 60 | 38 | 48 |
| `controle_synthese_v4_pilote` | 14/11/2026 | pilote n° 2 des consignes v4, 205 conversations | 60 | 48 | 48 |
| `controle_synthese_v4` | 15/11/2026 | fichier v4, 9 055 conversations | 120 | 113 | 96 |
| `controle_synthese_v5_debut` | 15/11/2026 | premières conversations v5, 470 conversations | 60 | 52 | 48 |
| `controle_synthese_v4_v5` | 16/11/2026 | fichier v4 + v5, 27 624 conversations | 120 | 109 | 96 |

Chaque sous-dossier contient le fichier lu par les lecteurs (`*_controle.txt`) et les notes
(`*_controle_notes.json`). La première ligne du fichier lu donne le nombre de conversations du fichier
contrôlé. Les seuils ne sont pas dans les fichiers de notes, sauf pour le premier contrôle. Ils sont écrits
dans [`HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md) (pour le dernier contrôle : 96 sur 120, comme pour le
fichier v4).

Le dernier contrôle (109 sur 120) porte sur le fichier v4 + v5 de 27 624 conversations, celui des essais du
16/11/2026. Le modèle publié (S03) n'a pas été affiné sur ce fichier : il l'a été sur le fichier v4 seul
(9 055 conversations), contrôlé à 113 sur 120. Le récit de ce contrôle est dans
[`HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md) et dans
[`FICHE_DONNEES.md`](../../../docs/chatbot/FICHE_DONNEES.md).

Pour recompter :

```python
import json

J = "experiments/chatbot_rtx/jugements/"
for f in ("controle_synthese_qwen/fusion_controle_notes.json",
          "controle_synthese_v4_pilote/pilote2_controle_notes.json",
          "controle_synthese_v4/final_controle_notes.json",
          "controle_synthese_v5_debut/pilote_v5b_controle_notes.json",
          "controle_synthese_v4_v5/final_v6_controle_notes.json"):
    d = json.load(open(J + f, encoding="utf-8"))
    d = d.get("result", d)
    print(f, len(d["lignes"]), sum(l["sans_defaut"] for l in d["lignes"]))
# 60 38, puis 60 48, 120 113, 60 52, 120 109
```

## 5. Format des fichiers

Dans chaque sous-dossier de jugement :

- `items.json` : ce que les juges ont vu. Une liste de réponses : `id` (`r001`, `r002`…), `avant` (les
  échanges précédents de la conversation, souvent vide), `message` (le message de l'utilisateur), `reponse`
  (la réponse à noter).
- `cle.json` : la clé, jamais montrée aux juges. Pour chaque `id` : `modele`, `job` (numéro du message dans
  le jeu), `graine` (11, 22 ou 33), `tour`.
- `notes.json` : les notes brutes. Une liste de 3 juges ; pour chaque juge, `id` → `F`, `R`, `S` (0 ou 1).
- `notes_finales.json` : le dépouillement. `notes` donne, pour chaque `id`, les trois notes à la majorité et,
  sous `juges`, les trois notes de départ. `resume` donne les sommes par modèle, le rapport au message par
  message (`R_par_message`), l'accord entre juges et les notes manquantes. `cle` recopie la clé.

`campagne_v4_neuf` contient en plus `reponses_5_graines_A0.jsonl` et `reponses_5_graines_S03.jsonl` : les
240 réponses de chaque modèle au jeu « neuf », avec 5 graines (11, 22, 33, 44, 55), telles que le modèle les
a données (`reply`, `message`, `job`, `turn`, `base_seed`, et des mesures de génération). Les juges n'ont noté
que les graines 11, 22 et 33. Ces deux fichiers servent au compte des « Certes » et des « Hélas ».

Dans chaque sous-dossier de contrôle :

- `*_controle.txt` : 5 lignes de statistiques, puis les conversations numérotées (`### 1.`, `### 2.`…).
- `*_controle_notes.json` : `lignes` donne, par conversation, les quatre notes à la majorité (`F`, `R`, `V`,
  `S`), `sans_defaut` et les défauts relevés. Le fichier donne aussi les totaux et les impressions des trois
  lecteurs. Deux fichiers ont une forme un peu différente : celui du premier contrôle range les notes de
  chaque lecteur sous `notes_brutes` ; celui de `controle_synthese_v5_debut` est la sortie brute de l'outil
  qui a lancé les lecteurs, et le dépouillement y est sous `result`.

`final_neuf2` (jugement du 16/11/2026) contient les quatre mêmes fichiers (`items.json`, `cle.json`,
`notes.json`, `notes_finales.json`), plus :

- `gardes.json` : pour chacun des six modèles (`v6_A0`, `v6_S03t`, `v6_s03`, `v6_s05`, `v6_s10`, `v6_s15`), le
  BPB sur les pièces de validation de Molière (`bpb_moliere_val`), le BPB de la synthèse entraînée et des
  trois fichiers tenus à l'écart, l'écart de par-cœur (`par_coeur`, en fraction : 0,0192 = 1,92 %), les deux
  gardes (`garde_bpb`, `garde_par_coeur`), le nombre moyen de mots par réponse et le temps par réponse ;
- six fichiers `reponses_5_graines_<modèle>.jsonl` (`A0`, `S03t`, `s03`, `s05`, `s10`, `s15`) : les 240
  réponses de chaque modèle, avec 5 graines. Les juges n'ont noté que les graines 11, 22 et 33.

Dans la clé de ce dossier, le témoin S03 s'appelle `S03t`, et les candidats `s03`, `s05`, `s10`, `s15`.

## 6. Qui a jugé

Des **agents IA** (Claude, modèle fermé d'Anthropic). Aucune personne n'a noté une réponse ni une
conversation.

- Jugements : 3 juges par lot de 72 réponses. Chaque juge ne reçoit qu'un lot, sans la clé.
- **Les juges du 16/11/2026 (`final_neuf2`) sont des agents Claude Opus 5.5, un autre modèle d'IA que celui des
  jugements des 14 et 15/11.** Même consigne, mot pour mot ; 36 juges pour 12 lots de 72. Les notes du 16/11
  ne se comparent donc pas directement à celles du 15/11 (par exemple 96 sans défaut pour S03 le 16/11, contre
  100 le 15/11).
- Contrôles de qualité : 3 lecteurs, qui lisent chacun tout le fichier.
- Les juges et les lecteurs notent. Ils n'ont écrit aucune donnée d'entraînement.

## 7. Limites

- **Les juges sont des IA.** On ne sait pas ce que des personnes auraient noté.
- **Les jeux sont petits** : 24 tours de parole (« reglage », « test ») ou 48 (« neuf », « neuf2 »), joués
  avec 3 graines.
- **Une notation refaite ne redonne pas le même chiffre.** Les mêmes 72 réponses de A0 ont été notées
  plusieurs fois, par des juges différents :
  - jeu de réglage, ancien choix : 44 sans défaut le matin (`synthese_qwen_reglage`), 43 à midi
    (`decodage_reglage`) ;
  - jeu de test, ancien choix : 35 (`synthese_qwen_test`), puis 38 (`decodage_test`) ;
  - jeu de réglage, choix adopté : 42 sous le nom V2 (`decodage_reglage`), puis 44, 44 et 44 sous le nom A0
    (`donnees_nettoyees_reglage`, `pretrain_filtre_reglage`, `synthese_qwen_ancre_reglage`).
- **La courbe de la campagne n'est pas régulière** : S10 obtient 88 sans défaut, entre S05 à 97 et S15 à 100.
  Le 16/11, elle ne l'est pas non plus : s03 93, s05 101, s10 88, s15 105.
- **Les jeux ont resservi.** Le jeu « reglage » a servi à 5 jugements, le jeu « test » à 2. Les jeux « neuf »
  et « neuf2 » n'ont servi qu'une fois chacun.
- **Le gain de S03 sur A0 est mesuré sur deux jeux neufs** : de 82 à 100 sur « neuf » (15/11), de 73 à 96 sur
  « neuf2 » (16/11). Les deux jugements n'ont pas les mêmes juges (voir § 6). Sur les jeux « reglage » et
  « test », S03 n'a pas été jugé.
- **Les candidats du 16/11 ne se distinguent pas du témoin.** Pour les quatre candidats, le
  test des signes contre S03 donne un p d'au moins 0,27 (§ 10).
- **L'aveugle tient aux consignes.** Un juge ne reçoit pas la clé, mais rien dans ce dossier ne prouve qu'il
  n'a ouvert aucun autre fichier.
- Les thèmes des messages de test recoupent ceux des conversations de synthèse : voir la fiche du modèle,
  § 6.3.

## 8. Écart relevé en recomptant

Une version antérieure de la fiche du modèle (§ 6.3), de `chatbot_moliere.md` (§ 7) et de l'historique
écrivait que A0 commence 0 réponse par « Certes » et 2 par « Hélas ». Le recomptage sur
`reponses_5_graines_A0.jsonl` donne **0 et 1**. Deux réponses de A0 contiennent le mot « hélas », mais une
seule commence par lui. Les chiffres de S03 (28 et 24) sont exacts.

## 9. Ce qui n'est pas publié, et pourquoi

- **Les réponses complètes des autres modèles** (5 graines, mesures de forme, transcriptions) : les réponses
  jugées sont déjà dans `items.json`. Au jeu « neuf », seules celles de A0 et de S03 sont jointes, parce
  qu'un chiffre publié porte sur leurs 240 réponses. Au jeu « neuf2 » (`final_neuf2`), celles des six
  modèles sont jointes.
- **Les lots de 72 donnés aux juges** : ce sont des tranches de `items.json`, dans l'ordre.
- **Le journal de travail** : il contient des chemins et des détails du poste de travail. Son contenu utile
  est repris dans [`HISTORIQUE.md`](../../../docs/chatbot/HISTORIQUE.md). Les règles de décision y sont
  écrites avec leur heure ; ce dossier ne les contient pas et n'en donne pas de preuve datée. L'historique
  (« Méthode et limites ») dit lesquelles ont été écrites avant les notes, et lesquelles après.
- **Les autres scripts de travail** : la préparation des recettes, le découpage des lots de 72 et le calcul
  des gardes sont dans [`../outils/`](../outils/README.md). Les scripts qui ont lancé les juges ne sont pas
  publiés. Dans ce dossier-ci, seuls `aveugle.py`, `recompter_jugements.py` et `apparie.py` sont rangés. Les
  lignes de Python du § 3 et `apparie.py` refont le dépouillement apparié.
- **La graine de mélange de `final_neuf2`** : elle n'a pas été retrouvée. Le mélange de ce jugement ne peut
  donc pas être rejoué avec `aveugle.py prepare` ; `items.json` et `cle.json` sont publiés tels que les
  juges et le dépouillement les ont utilisés.
- **Les mesures qui ne sont pas des jugements** : BPB, écarts de par-cœur, temps par réponse. Pour les six
  modèles du 16/11, elles sont dans `final_neuf2/gardes.json`. Pour les jugements des 14 et 15/11 (par
  exemple les écarts de par-cœur de 4,6 %, 10,2 %, 23,4 % et 30,5 %), ce dossier ne les prouve pas.
- **Les notes du contrôle du pilote v5a** (49 sur 60 dans l'historique) : aucun fichier de notes n'a été
  retrouvé dans le dossier de travail.
- **Le jury des anciens modèles** (240 verdicts, `chatbot_moliere.md` § 6) : ses fichiers ne sont pas dans
  le dossier de travail d'où viennent ces preuves.
- **Les fichiers de travail du nettoyage des données** (152 Mo) : hors sujet ici.

Trois fichiers de notes de contrôle ne sont pas des copies exactes : `fusion_controle_notes.json`,
`pilote2_controle_notes.json` et `pilote_v5b_controle_notes.json`. Le chemin complet du fichier lu sur le
poste de travail y figurait : des lecteurs l'avaient recopié dans leur impression, et l'outil l'avait gardé
dans l'aperçu de la consigne. Ce chemin est remplacé par le chemin relatif au dépôt (6, 1 et 4 lignes
changées) ; rien d'autre n'est modifié. Tous les autres fichiers des jugements et contrôles des 14 et 15/11 et
du contrôle du 16/11 sont des copies octet pour octet. Deux ont changé de nom : les fichiers `rows.jsonl` de
A0 et de S03 au jeu « neuf » s'appellent ici `reponses_5_graines_A0.jsonl` et `reponses_5_graines_S03.jsonl`.

Licences : voir [`LICENCES.md`](../../../LICENCES.md).

## 10. Le jugement final du 16/11/2026 (`final_neuf2`)

**Ce qui a été jugé.** Six modèles sur le jeu « neuf2 », qui n'avait jamais servi avant ce jugement :
le témoin S03 (le modèle publié, `S03t` dans la clé), A0, et les quatre candidats « v6 » (`s03`, `s05`,
`s10`, `s15`, affinés avec la synthèse v4 + v5 à 3, 5, 10 et 15 %). 144 réponses par modèle, soit 864
réponses mélangées en 12 lots de 72. 36 juges, 3 par lot, 0 note manquante, vote à la majorité. Les juges
sont des agents IA (Claude Opus 5.5), pas des personnes : voir § 6.

**La règle**, écrite le 15/11/2026 à 16 h, avant la génération de la synthèse v5 (ce dossier n'en donne pas
de preuve datée, voir § 9) : le témoin est S03. Un candidat remplace S03 seulement si son nombre de réponses
sans défaut (T) dépasse celui de S03 d'au moins 12, et si F, R et S ne reculent pas de plus de 6. Deux gardes,
sur le candidat : BPB sur Molière au plus égal à celui de A0 + 0,010 (soit 1,06602), et par-cœur sous 10 %.

**Résultats** (sur 144 réponses par modèle ; T = sans défaut ; BPB et par-cœur viennent de `gardes.json`) :

| Modèle (nom dans la clé) | F | R | S | T | T − T(S03) | BPB Molière | Par-cœur |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 (sans synthèse) | 113 | 94 | 127 | 73 | −23 | 1,05602 | −0,65 % |
| S03 (`S03t`, témoin, modèle publié) | 124 | 111 | 142 | **96** | — | 1,05570 | 1,92 % |
| v6 à 3 % (`s03`) | 130 | 105 | 139 | 93 | −3 | 1,05568 | 0,80 % |
| v6 à 5 % (`s05`) | 133 | 109 | 144 | 101 | +5 | 1,05568 | 2,69 % |
| v6 à 10 % (`s10`) | 132 | 104 | 134 | 88 | −8 | 1,05746 | 6,23 % |
| v6 à 15 % (`s15`) | 128 | 118 | 141 | 105 | +9 | 1,05864 | 9,81 % |

Le « par-cœur » est 1 − (BPB de la synthèse entraînée ÷ moyenne des BPB des trois fichiers de synthèse tenus
à l'écart). Pour S03, il est mesuré sur le fichier v4 + v5, dont S03 n'a vu que la partie v4. Les parts de
synthèse des candidats sont des parts attendues, calculées par la formule des recettes (2,99 %, 5,04 %,
10,01 % et 14,98 %) ; seule celle de S03 (3,02 %) est mesurée.

**Les gardes.** Les quatre candidats passent les deux gardes (`garde_bpb` et `garde_par_coeur` valent `true`
dans `gardes.json`) ; celui à 15 % de justesse : 9,81 % pour un plafond de 10 %.

**Comparaison appariée contre S03** (même message, même graine, même tour ; test des signes bilatéral ; sur T) :

| Modèle | Meilleur que S03 | Moins bon | p |
| --- | ---: | ---: | ---: |
| A0 | 25 | 48 | 0,0095 |
| v6 à 3 % | 28 | 31 | 0,79 |
| v6 à 5 % | 36 | 31 | 0,63 |
| v6 à 10 % | 25 | 33 | 0,36 |
| v6 à 15 % | 31 | 22 | 0,27 |

**Accord entre juges.** Les 3 juges donnent la même note à 829 réponses sur 864 pour F, 793 sur 864 pour R et
839 sur 864 pour S.

**Verdict : S03 reste le modèle retenu. Aucun candidat n'atteint l'écart exigé de +12.** Le meilleur écart est
+9 (v6 à 15 %) ; aucun test des signes contre S03 n'est significatif pour un candidat. S03 reste meilleur
que A0 sur ce second jeu neuf (96 contre 73, p = 0,0095), comme sur le premier (100 contre 82, p = 0,044).

**Limites propres à ce jugement.** Les juges du 16/11 ne sont pas ceux du 15/11 (§ 6) : les notes des deux
jours ne se comparent pas directement. La décision n'en dépend pas, puisque S03 est rejugé dans le même lot,
par les mêmes juges, que les candidats. La qualité jugée par des personnes n'a jamais été mesurée.

**Recompter.** Les deux commandes du § 2 (`RECOMPTER` et `APPARIE`) redonnent les nombres ci-dessus. Pour
`final_neuf2`, `RECOMPTER` affiche une ligne par modèle ; `APPARIE` affiche une ligne par modèle autre que S03t
(A0 et les quatre candidats).
