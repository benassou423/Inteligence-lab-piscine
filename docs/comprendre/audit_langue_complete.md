Prochaine action : améliorer le nettoyage avant d’élargir le filtre — lecture 2 min.

# Vérifier la langue jusqu’à la fin du texte

1. **À quoi ça sert.** Le modèle apprend avec les textes qu’on lui donne. Un livre bilingue ou une page remplie de tableaux utilise une partie de son temps d’apprentissage. Il faut repérer ces cas sans jeter le bon français.

2. **Comment ça marche.** L’outil `tools/audit_langid_full.py` compte ce qu’un nouveau réglage ajouterait. Il récupère ensuite les documents complets et les fait lire par deux détecteurs de langue. Pour trouver les changements au milieu d’un livre, il parcourt aussi des passages successifs de 3 000 caractères. Codex relit des cas et note exactement la partie examinée. Le corpus utilisé pour entraîner le modèle reste inchangé.

3. **Les chiffres mesurés le 04/11.** 3 221 115 documents recensés ; 46 987 ajouts possibles, soit 130 Mo. 888 documents analysés automatiquement jusqu’au bout. 75 relectures IA : 64 intégrales, 11 par passages. Les 513 ajouts hors Wikipédia sont tous dans l’analyse automatique. Source des chiffres : `data/annotations/ia-20261104/full_document_audit/summary.json`.

4. **Ce qu’on a appris.** Une brève sur un poète écrivant en patois est bien rédigée en français : le détecteur la rejette à tort. À l’inverse, un livre alterne provençal et français alors que les deux détecteurs le jugent français. D’autres documents commencent par une vraie biographie, puis deviennent des tableaux bruts. **On garde donc le filtre actuel**, puis on améliorera le nettoyage. Les scores des détecteurs ne sont pas des garanties, et les relectures restent des jugements IA.

5. **Trois questions du jury.**

   **Pourquoi ne pas lire seulement le début ?** Une préface française peut précéder un livre bilingue. Le début ne suffit pas à représenter la suite.

   **Deux détecteurs d’accord, c’est suffisant ?** Non. Nous avons trouvé des éditions bilingues malgré leur accord. Il faut confronter leurs scores au texte réel.

   **Avez-vous prouvé une amélioration du modèle ?** Non. Cet audit évite un changement mal justifié. Le gain devra être mesuré par deux entraînements comparables, avec les mêmes réglages et le même budget.

Rapport détaillé : `experiments/langid_full_audit_20261104.md`. Vérification du code : **277 tests verts**, kit intact. Les données d’entraînement n’ont pas changé ; le dernier contrôle reste à **0 document signalé pour contamination**, c’est-à-dire aucun recouvrement dépassant le seuil officiel avec les textes réservés à l’évaluation.
