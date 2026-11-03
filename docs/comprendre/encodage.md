# Encodage : réparer le texte abîmé

## À quoi ça sert
Un texte mal décodé affiche **« Ã©tÃ© »** au lieu de **« été »** : c'est le **mojibake**. Le modèle apprendrait ces fautes.
On les répare, et on met le texte dans **une seule forme Unicode**, la même que la val.

## Comment ça marche
- **ftfy** (« fixes text for you ») détecte le mojibake et refait le bon décodage. Il corrige aussi les caractères de contrôle, les ligatures (ﬁ → fi) et les fins de ligne Windows.
- **NFC** : « é » s'écrit toujours avec **un seul** caractère, jamais « e + accent ». Même texte, mêmes tokens.
- **Typographie gardée** : ’ « » — – … restent (`uncurl_quotes=False`). La val en est pleine.
- **Espaces ramenées à l'espace simple** : insécable U+00A0, fine U+202F, fine U+2009 (toute la catégorie Unicode Zs). Les **invisibles** (trait d'union conditionnel, espace sans chasse) sont supprimés.
- **Pré-filtre exact** : ftfy n'est appelé que si le texte contient un caractère qu'il pourrait corriger. Même résultat, ~5x plus vite.
- **Un faux positif de ftfy refusé** : il lit « **»Éminence** » (guillemet de continuation des vieux livres) comme du MacRoman et écrit « **ȃminence** ». Une réparation MacRoman qui **mange un guillemet** est annulée pour la ligne.

## Les chiffres clés
| Mesure | Valeur | D'où |
|---|---|---|
| Espaces Unicode dans la **val du kit** | **0** (ni U+00A0, ni U+202F, sur 813 Ko) | mesuré sur `data/validation.jsonl` |
| Typographie dans la val | ’ 4 383 · ' 2 587 · « 357 · – 195 · — 26 · … 49 | mesuré |
| Livres Gutenberg réparés | 79 sur 1 991 (dont **70 mojibake** : « c\x9cur » → « cœur ») | échantillon `--limit 2000` |
| Articles Wikipédia modifiés | 11 sur 2 000 (ponctuation pleine chasse, ligatures) | idem |
| Vitesse | ftfy seul **0,6 à 1 Mo/s** par cœur ; avec pré-filtre **4 à 6 Mo/s** | mesuré |
| Équivalence du pré-filtre | **0 différence** sur 60 000 textes piégés (hors faux positifs refusés) | test `test_identique_a_ftfy` |
| Faux positif « »É » → « ȃ » | **15 livres** Gutenberg abîmés avant correction, 0 après ; aucun autre document changé | corpus complet (Wikisource compris) |

Référence : ftfy (Robyn Speer). FineWeb et la plupart des pipelines du domaine l'utilisent.

## 3 questions probables du jury
1. **Pourquoi ne pas garder les espaces insécables, c'est la vraie typographie française ?**
   Parce que la val n'en a **aucune**. Le BPB est mesuré sur ce texte-là : apprendre « : » précédé d'une insécable, c'est apprendre une habitude que le test n'a pas. On garde ce qui y est (’ « »), on aligne le reste.
2. **ftfy peut-il abîmer un texte correct ?**
   Oui, rarement : on l'a **mesuré** (« »Éminence » → « ȃminence », 15 livres) et on refuse ce cas précis. L'**échantillon avant/après** (`_changed_sample/`) sert à le vérifier à l'œil.
3. **Pourquoi pas NFKC, qui normalise plus ?**
   NFKC change le sens de certains caractères (« ² » devient « 2 », « ½ » devient « 1/2 »). NFC ne change que la façon d'écrire un même caractère.
