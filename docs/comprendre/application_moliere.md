# Fiche : l'application macOS « Molière »

**À faire d'abord (10 min) : ouvrir `data/stages/app_moliere_20261116/Moliere-1.0.0-macos-arm64.dmg`, glisser Moliere dans Applications, l'ouvrir, puis lire cette fiche.** C'est la copie du DMG qui fait foi : la copie `Moliere.app` du dossier `data/stages` ne passe pas la vérification de signature (iCloud y ajoute des attributs, des étiquettes cachées sur un fichier). Procédure complète : `docs/chatbot/APPLICATION.md`.

## Comment elle est faite, en 5 idées

1. **La fenêtre est une page web.** `pywebview` ouvre une vraie fenêtre macOS qui affiche la page `app_moliere/page.html`. La page vient de votre propre Mac.
2. **Un petit serveur local la nourrit.** (Serveur local = un programme qui répond seulement aux requêtes venant de votre ordinateur.) `app_moliere/serveur.py` écoute sur `127.0.0.1`, l'adresse « mon ordinateur ». Il reprend les contrôles de `demo/server.py` : taille limitée, JSON (un format de texte structuré, clés et valeurs), refus des requêtes venant d'un autre site.
3. **Le modèle est téléchargé une fois.** `app_moliere/modele.py` récupère le zip (240 Mo) sur la release GitHub. Il vérifie deux empreintes sha256 (le « code-barres » d'un fichier : un octet change, l'empreinte change), n'extrait que 4 fichiers, puis les place d'un seul geste. Le téléchargement est en HTTPS (connexion chiffrée, vérifiée par un certificat : la preuve d'identité du serveur).
4. **PyInstaller emballe tout.** (PyInstaller = un outil qui copie Python, PyTorch et notre code dans un seul binaire, un programme prêt à lancer.) `tools/build_app_macos.py` construit, signe (codesign, l'outil Apple de signature), fabrique le `.dmg` (image disque) et teste le binaire.
5. **macOS avertit.** Nous signons « ad hoc » : sur un Mac à puce Apple (M1 et suivants), un programme doit être signé pour démarrer. Cette signature permet à macOS de repérer un fichier abîmé après la signature, mais elle ne dit pas qui l'a faite, et n'importe qui peut re-signer un fichier modifié. D'où le bouton « Ouvrir quand même ».

## Pièges rencontrés (utiles à raconter)

1. Une application emballée ne trouve pas les certificats du système : le téléchargement HTTPS aurait échoué. Solution : `certifi` (un paquet qui fournit les certificats).
2. Une application sans terminal n'a pas de sortie standard (l'endroit où un programme écrit ses messages à l'écran) : le journal va dans un fichier, `journal.log`.
3. `ftb/__init__.py` lit `config/official.json` dès l'import : il faut l'embarquer.
4. `codesign` refuse un `.app` qui porte un attribut ajouté par iCloud. On signe donc dans un dossier temporaire.

## 6 questions possibles du jury

1. **Pourquoi ne pas mettre le modèle dans l'application ?** Pour garder l'image disque à environ 230 Mo (environ 470 Mo avec le modèle, estimation), et parce que le modèle a sa propre licence et sa propre publication. Attention : l'adresse et l'empreinte du modèle sont écrites dans le code, donc un nouveau modèle demande aussi une nouvelle version de l'application. L'option `--embarquer-modele` donne une version hors ligne.
2. **Comment sait-on que le fichier téléchargé est le bon ?** Deux empreintes sha256 inscrites dans le code : celle du zip, puis celle des poids après extraction. Si l'une diffère, rien n'est installé.
3. **L'application envoie-t-elle mes messages ?** Non. L'application ne se connecte à Internet que pour télécharger le modèle, une fois (GitHub voit alors l'adresse IP, comme pour tout téléchargement), et si vous cliquez sur un lien. Vos messages restent dans la fenêtre : ni envoyés, ni enregistrés sur le disque ; le journal ne note que l'adresse des requêtes, pas leur contenu. Le serveur n'écoute que sur 127.0.0.1.
4. **Pourquoi macOS dit-il qu'elle est dangereuse ?** macOS ne peut pas savoir qui a fait l'application : elle n'est ni signée avec une identité de développeur ni notarisée. La notarisation est un contrôle automatique fait par Apple, qui demande un compte développeur payant (99 $ par an). Le message dit qu'Apple ne l'a pas vérifiée, pas qu'elle est dangereuse.
5. **Pourquoi seulement les Mac à puce Apple ?** Notre version de PyTorch (2.15) n'existe pas pour les Mac Intel : le seul fichier d'installation (une « roue », le format de paquet Python) pour macOS est celui des puces Apple, macOS 14 minimum. Une version Intel demanderait une ancienne version de PyTorch, puis une construction et des tests sur un Mac Intel.
6. **Une partie du modèle a-t-elle été écrite par une autre IA ?** Aucun poids ne vient d'un autre modèle : il est entraîné de zéro. Mais 3 % des textes de l'affinage (la seconde phase de l'entraînement) sont des conversations écrites par gemma 4, un modèle à poids ouverts. Le reste de l'affinage est du théâtre, et le pré-entraînement (la première phase, 2 milliards de tokens) vient de sources publiques (Wikipédia, Wikisource, Gutenberg, pages web…), sans conversation de synthèse ajoutée par nous.

## Chiffres réels (16/11/2026)

- Application installée : environ 610 Mo. Image disque : environ 230 Mo. Modèle : 240 Mo. Valeurs exactes de la construction : `rapport_construction.json`.
- Premier lancement du binaire emballé, vrai téléchargement GitHub : environ 10 s entre « Télécharger le modèle » et le modèle chargé (connexion du Mac de construction, 16/11/2026), puis 1,8 à 5,1 s par réponse sur le processeur (4 essais).
- Tests : `tests/test_app_moliere.py`.
