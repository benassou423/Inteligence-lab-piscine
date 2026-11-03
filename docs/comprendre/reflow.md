# Mise en forme des livres : la forme de la val

## À quoi ça sert
Les fichiers Gutenberg sont **coupés tous les ~70 caractères**, au milieu des phrases, et marquent l'italique par `_soulignés_`.
La val, elle, met **un paragraphe par ligne**, sans soulignés. Le modèle doit apprendre la forme qu'il sera noté sur.

## Comment ça marche
- On mesure la **largeur de coupe** du livre (95 % des lignes sont plus courtes).
- Un paragraphe dont les lignes sont **pleines** (≥ 70 % de cette largeur) est de la prose coupée : on **recolle** ses lignes.
- On **ne touche pas** aux vers (lignes courtes, chaque ligne commence par une majuscule), aux tables et aux blocs indentés.
- Une ligne qui commence par **« -- »** (réplique de dialogue) reste une nouvelle ligne, comme dans la val.
- On retire les **soulignés d'italique** (`_mot_` → `mot`).

## Les chiffres clés (90 livres réels comparés à la val)
| Mesure | Livres avant | Livres après | Val du kit |
|---|--:|--:|--:|
| Longueur médiane d'une ligne | 66 car. | 95 car. | 83 car. |
| Lignes de 80 caractères ou moins | 100 % | 47 % | 49 % |
| Retours à la ligne suivis d'une ligne vide | 17 % | 45 % | 47 % |
| Soulignés `_` | 44 039 | 2 755 | **0** |
| Paragraphes de la val sur une seule ligne | | | 874 sur 890 |

## 3 questions probables du jury
1. **Pourquoi ça compte pour le BPB ?**
   Le BPB compte chaque octet, retours à la ligne compris. Un modèle habitué à un « \n » tous les 70 caractères le prédit au mauvais endroit dans la val, et perd des bits sur chaque ligne.
2. **Et la poésie ?**
   Elle reste en vers : ses lignes sont courtes et commencent par une majuscule. Les exemples non recollés vérifiés à l'œil sont du théâtre en vers (Cyrano) et des poèmes.
3. **Pourquoi cette étape après le filtre de langue ?**
   Elle ne change ni les mots ni la langue : la placer après évite de refaire le filtre de langue (l'étape la plus lente) quand on la règle.
