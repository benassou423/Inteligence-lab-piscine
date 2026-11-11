Prochaine action : nettoyer les documents avant de comparer un filtre plus large — lecture 2 min.

# Langue : garder du français utile

1. **À quoi ça sert.** Le modèle apprend sur un budget limité. Un document en langue étrangère utilise une partie de ce budget. Mais un texte français rempli de tableaux ou de répétitions n’est pas automatiquement utile non plus.

2. **Comment ça marche.** GlotLID est un détecteur de langue : il lit tout le document et produit un score par langue. `langid_score` enregistre ces scores pour éviter de refaire le calcul ; `langid` applique ensuite la règle. Aujourd’hui, on garde un texte si le score français est au moins **0,65**. La règle candidate `family` additionne français, normand, francoprovençal, picard et wallon. Elle peut récupérer du français mal reconnu, mais aussi accepter réellement d’autres langues.

3. **Ce qui a été mesuré le 04/11.** Ben a confié à Codex 300 annotations IA sur extraits, réalisées sans voir les prédictions. Ce premier diagnostic suggère un gain, mais 127 extraits sont tronqués et trois sources absentes. L’audit suivant recense les 3 221 115 documents et analyse 888 textes complets par deux détecteurs. Le changement ajouterait 46 987 documents, soit 130 Mo. Les 75 relectures IA trouvent à la fois de vraies brèves françaises, des livres bilingues et des pages sans prose. **Le réglage français seul à 0,65 reste actif.** Ces jugements IA ne constituent pas une validation humaine ni une preuve de gain d’entraînement.

4. **Comment refaire les mesures.** `uv run python -m tools.annotate_langid --config nettoye` permet une annotation humaine à l’aveugle. `uv run python -m tools.langid_report --out-dir data/annotations/ia-20261104` compare les règles sur les annotations IA existantes, avec leurs limites. Les mesures complètes sont dans `experiments/langid_full_audit_20261104.md`. La fiche `docs/comprendre/audit_langue_complete.md` explique leur méthode.

5. **Trois questions du jury.**

   **Pourquoi 0,65 ?** C’est le seuil de départ du projet. L’audit actuel ne justifie pas de le changer ; il faut comparer la qualité des données puis le résultat d’entraînement.

   **Additionner les langues proches récupère-t-il seulement du français ?** Non. Certains textes sont du français mal reconnu, d’autres sont réellement bilingues ou dans une autre langue. Nous avons observé les deux cas.

   **Pourquoi annoter sans les scores ?** Pour ne pas recopier le verdict du détecteur. Les 300 extraits initiaux ont été lus ainsi ; l’audit suivant est au contraire un diagnostic ciblé à partir des alertes et ne fournit pas un taux d’erreur représentatif.
