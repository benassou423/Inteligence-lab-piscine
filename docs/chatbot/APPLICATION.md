# Application macOS « Molière »

Une fenêtre pour discuter avec le chatbot « à la manière de Molière » (modèle S03, 54 749 440 paramètres). Tout tourne sur votre Mac, sans compte ni clé.

## 1. Installer et utiliser (5 minutes)

Prérequis : un Mac à puce Apple (M1 ou plus récent ; pas de Mac Intel), macOS 14 ou plus (essayé seulement sur macOS 26.6), 1,5 Go d'espace libre pendant l'installation, Internet au premier lancement seulement.

1. Téléchargez `Moliere-1.0.0-macos-arm64.dmg` (environ 225 Mo) dans la release `chatbot-moliere-20261116` du dépôt : https://github.com/benassou423/Inteligence-lab-piscine/releases/tag/chatbot-moliere-20261116 (lien direct : https://github.com/benassou423/Inteligence-lab-piscine/releases/download/chatbot-moliere-20261116/Moliere-1.0.0-macos-arm64.dmg). Le fichier publié le 16/11/2026 fait 224 746 507 octets ; son empreinte sha256 est `228aad290b9ab0a119891952b5ac66d47772801af17c50530a1ce2d67e4a7a35`.

   Double-cliquez ensuite dessus.
2. Dans la fenêtre qui s'ouvre, glissez l'icône **Moliere** sur le dossier **Applications**.
3. Éjectez le volume « Molière » (clic droit, Éjecter).
4. Ouvrez **Moliere** depuis le dossier Applications (voir l'avertissement de sécurité ci-dessous au premier essai).
5. Au premier lancement, cliquez sur **Télécharger le modèle** (240 Mo, une seule fois). Une barre de progression s'affiche. Comptez environ 10 s entre « Télécharger le modèle » et le modèle chargé (connexion du Mac de construction, 16/11/2026) ; plus ou moins selon votre connexion. Ensuite, une réponse prend 1,8 à 5,1 s sur le processeur (4 essais).
6. Quand le chat apparaît, écrivez un message (Entrée envoie, Maj+Entrée va à la ligne). Vous pouvez changer de personnage (11 au choix), recommencer, régénérer une réponse ou la copier.

### Vos messages restent chez vous

L'application ne se connecte à Internet que pour télécharger le modèle, une fois (GitHub voit alors votre adresse IP, comme pour tout téléchargement), et si vous cliquez sur un lien. Vos messages restent dans la fenêtre : ni envoyés, ni enregistrés sur le disque. Le serveur local n'écoute que sur `127.0.0.1`.

### Message de sécurité de macOS

L'application est signée **sans identité de développeur Apple** et n'est pas « notarisée ». macOS affiche donc, au premier essai, que « Moliere » n'a pas été ouvert parce qu'Apple n'a pas pu vérifier l'absence de logiciel malveillant. Ce n'est pas une panne. Pour l'ouvrir quand même, il faut un compte administrateur du Mac :

1. Double-cliquez sur Moliere dans Applications. Une fenêtre dit que « Moliere » n'a pas été ouvert et qu'Apple n'a pas pu vérifier l'absence de logiciel malveillant. Cliquez sur **Terminé**, surtout pas sur **Placer dans la corbeille**.
2. Dans l'heure qui suit : menu Pomme > **Réglages Système** > **Confidentialité et sécurité** (colonne de gauche).
3. Descendez jusqu'à la section **Sécurité** : une ligne dit que « Moliere » a été bloqué. Cliquez sur **Ouvrir quand même**.
4. Une fenêtre s'ouvre : cliquez encore sur **Ouvrir quand même**, puis entrez le mot de passe de la session (ou Touch ID).
5. Les fois suivantes, un double-clic suffit. Si le bouton n'apparaît pas, recommencez à l'étape 1. Sur macOS 14, clic droit sur l'application puis **Ouvrir** suffit encore.

Le texte exact des fenêtres peut varier selon la version de macOS.

Variante dans le Terminal (retire la marque de quarantaine ; le Terminal est dans Applications > Utilitaires > Terminal) :

```bash
xattr -dr com.apple.quarantine /Applications/Moliere.app
```

### Où sont rangées les choses

- Modèle téléchargé : `~/Library/Application Support/Moliere/modele/` (4 fichiers, environ 240 Mo).
- Journal de l'application : `~/Library/Application Support/Moliere/journal.log` (à joindre en cas de problème). Il note l'adresse des requêtes, pas leur contenu.

### Si le téléchargement échoue

Le message est en français dans la fenêtre (pas d'Internet, serveur indisponible, fichier abîmé). Cliquez sur **Réessayer** : le téléchargement repart de zéro et un fichier incomplet n'est jamais utilisé. Plan B : `uv run python -m app_moliere.modele`, depuis `Inteligence-lab-piscine/` (voir plus bas).

### Désinstaller

1. Fermez l'application, glissez `/Applications/Moliere.app` à la Corbeille.
2. Dans le Finder, menu **Aller > Aller au dossier…** (Maj+Cmd+G), tapez `~/Library/Application Support/` et supprimez `Moliere`. Facultatif : supprimez aussi `~/Library/WebKit/io.github.benassou423.moliere` et `~/Library/Caches/io.github.benassou423.moliere`.

## 2. Reconstruire l'application depuis le dépôt

Ces commandes se lancent depuis un clone du dépôt, dans le dossier `Inteligence-lab-piscine/` (Mac Apple Silicon, `uv` installé) :

```bash
# essai depuis les sources (une vraie réponse de Molière, puis sortie)
MOLIERE_MODELE=$PWD/data/models/chatbot_moliere uv run --group app python -m app_moliere --autotest --rapport /tmp/rapport.json

# lancer la fenêtre depuis les sources
uv run --group app python -m app_moliere

# construire Moliere.app + le .dmg + tester le binaire gelé (environ 2 minutes)
uv run --group app python tools/build_app_macos.py
# variante hors ligne (modèle embarqué, plus lourde) :
uv run --group app python tools/build_app_macos.py --embarquer-modele data/models/chatbot_moliere

# installer seulement le modèle (sans l'application)
uv run python -m app_moliere.modele --dest data/models/chatbot_moliere
```

Les fichiers de construction vont dans `data/stages/app_moliere_20261116/` : le `.dmg`, `rapport_construction.json` (taille et empreinte exactes du `.dmg` de la construction) et les rapports `autotest_*.json`. Les tests : `uv run --group app pytest -q tests/test_app_moliere.py`.

Options du script de construction : `--embarquer-modele DOSSIER`, `--sans-dmg`, `--modele-autotest DOSSIER` (modèle local de l'autotest) et `--autotest-telecharge` (autorise l'autotest à télécharger le modèle si le dossier local est absent ; sans elle, le script s'arrête dans ce cas).

Variables utiles : `MOLIERE_MODELE` (dossier d'un modèle déjà présent), `MOLIERE_DONNEES` (déplace le dossier des données, utile aux tests).

### Refaire les preuves (`autotest_*.json`)

Seul `autotest_gele.json` est écrit par le script de construction (essai du binaire gelé, avec un modèle local, `--modele-autotest`). Les trois autres se refont à la main, depuis `Inteligence-lab-piscine/`. Chaque commande écrit un rapport JSON (`"ok": true` si l'application a répondu) :

```bash
D=data/stages/app_moliere_20261116

# 1. depuis les sources (modèle local)  ->  autotest_sources.json
MOLIERE_MODELE=$PWD/data/models/chatbot_moliere uv run --group app python -m app_moliere --autotest --rapport $D/autotest_sources.json

# 2. premier lancement, dossier de données vide : le modèle est téléchargé depuis la release GitHub
#    (240 Mo, demande Internet et que la release soit publique)  ->  autotest_premier_lancement.json
MOLIERE_DONNEES=$(mktemp -d) $D/Moliere.app/Contents/MacOS/Moliere --autotest --rapport $D/autotest_premier_lancement.json

# 3. depuis le .dmg monté  ->  autotest_dmg.json
hdiutil attach $D/Moliere-1.0.0-macos-arm64.dmg -nobrowse -readonly
MOLIERE_MODELE=$PWD/data/models/chatbot_moliere MOLIERE_DONNEES=$(mktemp -d) "/Volumes/Molière/Moliere.app/Contents/MacOS/Moliere" --autotest --rapport $D/autotest_dmg.json
hdiutil detach "/Volumes/Molière"
```

Pour la preuve 2, utilisez de préférence l'application du `.dmg` monté (`/Volumes/Molière/Moliere.app/...`) : la copie `Moliere.app` du dossier `data/stages` ne passe pas la vérification de signature, à cause d'attributs ajoutés par iCloud.

## 3. Ce que contient l'application, et ses limites

Mesures de la construction du 16/11/2026 (les valeurs exactes de chaque construction, dont l'empreinte du `.dmg`, sont dans `rapport_construction.json` : elles changent à chaque reconstruction) :

| Élément | Valeur |
|---|---|
| `Moliere.app` installée | environ 610 Mo |
| `.dmg` (compressé) | environ 230 Mo |
| Modèle téléchargé | environ 240 Mo |
| Premier lancement mesuré (binaire gelé, GitHub réel) | environ 10 s du clic au modèle chargé, puis 1,9 s la première réponse |
| Réponse sur le processeur, binaire gelé et sources (4 essais) | 1,8 à 5,1 s |

Contenu : Python 3.12, PyTorch (calcul sur processeur seulement), le moteur de chat du projet, le kit officiel du concours (le dossier `ftb/` et le fichier `config/official.json`), `certifi` (certificats pour le téléchargement sécurisé), `pywebview` (la fenêtre). Le modèle n'est pas dans l'application par défaut.

Limites :

- **Apple Silicon (arm64) seulement**, macOS 14 minimum (imposé par PyTorch). Pas de version Intel, pas de version Windows construite (le code Windows/Linux de recherche du dossier de données existe, il n'a jamais été essayé).
- **Non notarisée** : signature « ad hoc » (`codesign -s -`), donc l'avertissement de macOS décrit plus haut. Une notarisation demande un compte développeur Apple payant.
- **Modèle d'étude** : il imite un style et ne sait rien de fiable. Une petite part de son affinage (3 %) est faite de conversations écrites par un autre modèle de langue (gemma 4, poids ouverts). Ses réponses sont parfois incohérentes (96 réponses sans défaut sur 144 selon le jugement à l'aveugle d'agents IA).
- Pas de mise à jour automatique ; le modèle est celui de la release `chatbot-moliere-20261116`.
- Les liens de la fenêtre (À propos) ne sont pas garantis de s'ouvrir dans le navigateur ; l'adresse est écrite en toutes lettres.

### Licences

- Poids du modèle : CC BY-SA 4.0. Code du projet : Apache 2.0 (voir `LICENCES.md`).
- **Réserve** : le dossier `ftb/` et le fichier `config/official.json` (kit officiel du concours) sont embarqués tels quels dans l'application parce que le modèle ne fonctionne pas sans eux. Ce kit est publié sans licence explicite : l'application ne doit pas être redistribuée largement sans l'accord de ses auteurs.
- Bibliothèques embarquées (PyTorch, tokenizers, pywebview, certifi, etc.) : leurs licences respectives s'appliquent.
- L'icône a été générée par un outil d'IA : aucune licence n'est revendiquée dessus (voir `LICENCES.md`).
