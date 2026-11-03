# Langue : ne garder que le français

## À quoi ça sert
Le test est **100 % en français**. Une page en anglais ou une liste de noms en latin prend du budget sans aider.
On garde un document si un modèle de langue dit **« français » avec assez de confiance**.

## Comment ça marche
- **Deux étapes** : `langid_score` **note** chaque document (cher, ~1 Mo/s par cœur, gardé en **cache**), puis `langid` **trie** avec un seuil (instantané). Changer de seuil ne relance pas le modèle.
- **GlotLID** (modèle fastText, 2 102 langues) lit le document **en entier**, sur une seule ligne, et donne une **probabilité par langue**. On garde dans `meta` : `langid_fr` (français) et `langid_family` (français **+ parlers proches** : normand, franco-provençal, picard, wallon).
- **Deux règles** : `fr` (défaut) garde si **p(français) ≥ 0,65** (= l'ancienne règle « 1re étiquette française et score ≥ 0,65 ») ; `family` garde si **p(famille) ≥ seuil**. Le choix se fait sur l'annotation.
- **Pourquoi la famille ?** GlotLID **partage** un texte français entre des parlers voisins : un compte rendu de foot Wikinews donne **fra 0,58 + normand 0,35** → rejeté à tort.
- **Mesure à la main** : Ben annote ~300 documents **à l'aveugle** ; le rapport compare les deux règles et 12 seuils **sur les mêmes annotations**.

## Les chiffres clés
| Mesure (règle actuelle `fr` ≥ 0,65) | Valeur | D'où |
|---|---|---|
| Retirés : Wikinews | **21 %** (4 289 sur 20 428) : surtout des **comptes rendus de foot** | run complet du 03/11 |
| Retirés : Wikipédia | **3,7 %** (listes, homonymies, mais aussi du bon français étiqueté `frp_Latn`) | idem |
| Repêchés par `family` ≥ 0,65 | Wikinews **323 sur 4 289** (7,5 %) ; Wikipédia **38 %** des rejetés | tout Wikinews ; 40 000 articles Wikipédia |
| Pourquoi si peu sur Wikinews | la probabilité part aussi vers des étiquettes **absurdes** (`run_Latn`, `dag_Latn`, `atj_Latn`) | idem |
| Seuil de départ | **0,65** | FineWeb (Penedo et al., 2024) |

Références : GlotLID (Kargaran et al., EMNLP 2023), utilisé par FineWeb-2 ; repli sur fastText `lid.176`.

## Séance d'annotation (20 à 30 min)
1. `uv run python -m tools.annotate_langid --config nettoye` : **une touche** par document : `f` français, `n` pas français, `m` mélangé, `s` passer, `u` annuler, `q` quitter. La note du modèle n'est **pas affichée**.
2. `uv run python -m tools.langid_report` → `data/annotations/langid_report.md` : précision, rappel, intervalles, **les deux règles × 12 seuils**, et les documents **repêchés** par la famille.

## 3 questions probables du jury
1. **Pourquoi 0,65 et pas autre chose ?**
   Point de départ de FineWeb, puis **choisi sur nos annotations** : le rapport donne précision et rappel de chaque couple (règle, seuil) avec un intervalle de confiance.
2. **Additionner normand et français, ce n'est pas tricher ?**
   Non : ces textes **sont** en français ; c'est le modèle qui hésite entre des voisins. L'annotation dit si la règle famille garde du français (précision) ou laisse passer du vrai normand.
3. **Pourquoi annoter à l'aveugle ?**
   Voir « fra 0,58 » pousserait à répondre comme le modèle. Sans la note, l'annotation mesure le filtre au lieu de le recopier.
