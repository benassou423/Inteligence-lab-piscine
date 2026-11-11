Prochaine action : transférer l’archive sur le PC et ouvrir son dossier dans Codex — 5 min.

# Passage sur le PC Windows RTX 5090

1. **Transférer.** Copier `FTB_RTX5090_FINAL_20261110.zip` du Mac vers le PC, par clé USB, disque externe ou son stockage personnel. Extraire l’archive sous `C:\FTB-final\`. Le dossier de projet obtenu est `C:\FTB-final\Inteligence-lab-piscine`. Il contient les données sélectionnées, le code actuel et les instructions. Les 17 Go de sources brutes restent sur le Mac.

2. **Ouvrir Codex.** Se connecter au même compte sur le PC. Ouvrir le dossier extrait comme projet. Donner le message ci-dessous à Codex. Le transfert de fichiers est nécessaire en plus de la connexion au compte.

3. **Faire vérifier le PC.** Codex contrôle le pilote NVIDIA, installe Python 3.12 et les dépendances figées, puis réalise une vraie opération sur la carte. Prévoir 10 à 20 min pour une première installation, selon la connexion. Une mise à jour du pilote peut demander un redémarrage effectué par Ben.

4. **Entraîner.** Le kit prépare le flux et vérifie son empreinte. Le calcul utilise ensuite la RTX 5090 en float16, avec le budget officiel de 100 007 936 tokens. La durée sera mesurée sur les premiers pas ; aucun débit de ce PC n’a encore été mesuré. Les points de reprise permettent de relancer après une interruption.

5. **Récupérer.** À la fin, rapporter `RESULTATS_RTX5090_MASKED.zip` sur le Mac ou dans ce chat. Cette archive contient les poids, l’évaluation, les versions et les journaux. La mise à disposition du dataset et le commit/push de soumission restent à finaliser dans le dossier principal.

## Message à donner à Codex sur le PC

```text
Lis AGENTS.md en entier, puis AGENTS_PC.md et docs/remise/PASSAGE_PC_RTX5090.md.
Ce PC Windows possède une RTX 5090. Prends en charge l’installation et l’entraînement
officiel avec les fichiers de ce dossier. Vérifie d’abord l’intégrité avec
tools/windows_gpu.py. Ne modifie jamais ftb/, config/, corpus/, data/validation.jsonl
ou tests/test_kit.py. Préserve les dossiers lourds sous forme de liens/jonctions
vers leurs dossiers .nosync. Utilise uv depuis la racine de ce projet.
Installe uniquement les dépendances depuis les sources officielles indiquées.
Lance le calcul CUDA en arrière-plan avec un journal, sans --debug-budget.
Il faut 100 007 936 tokens réellement vus, le même dataset et le même tokenizer.
Vérifie la précision float16, les règles du kit et le score final. Garde la reprise.
Produis RESULTATS_RTX5090_MASKED.zip. Ne fais aucun commit ni push. Ne présente pas le
projet entier comme fini : les livrables et la publication sont suivis sur le Mac.
Explique chaque étape simplement, avec une prochaine action et une durée.
```

## Instructions techniques pour Codex sur Windows

Le dossier est un instantané de travail ; `.git` et les environnements du Mac sont exclus. Les modèles du bloc 1 et les corpus bruts complets restent disponibles sur le Mac. Le rôle de ce PC est de produire le run CUDA officiel et ses preuves.

Installer `uv` depuis sa distribution officielle si nécessaire. Ne pas changer les protections PowerShell. Depuis la racine du projet :

```powershell
uv run --no-project --python 3.12 tools/windows_gpu.py verify
uv run --no-project --python 3.12 tools/windows_gpu.py setup
uv run --no-project --python .venv.nosync/Scripts/python.exe tools/windows_gpu.py run
```

La troisième commande est longue : Codex doit la lancer avec `Start-Process`, redirection stdout/stderr vers `data/stages.nosync/gpu_handoff/`, puis suivre son PID et les journaux. Ne jamais lancer deux entraînements portant le même nom. L’option `run` reprend automatiquement `state.pt` lorsqu’il existe, et refuse un changement du kit ou des données.

Versions : Python 3.12, PyTorch **2.15.0 + CUDA 13.2**, NumPy **2.5.4**, tokenizers **0.23.3**, regex **2026.9.29**. Le binaire Windows CPython 3.12 a été trouvé dans l’index officiel PyTorch. L’installation doit réussir puis le test GPU doit passer avant le calcul long. Aucun binaire GPU n’a encore été exécuté sur ce PC depuis ce chat.

Le script `setup` crée les jonctions Windows vers les sept dossiers `.nosync`, sans remplacer un dossier ordinaire existant. Les commandes `--no-project` évitent qu’une synchronisation automatique remplace la version CUDA par une autre distribution.

Dataset attendu : **114 701 016 tokens disponibles**, **72 612 documents**, **419 295 632 octets de texte**. Le budget officiel utilise ses 100 007 936 premiers tokens. SHA-256 du flux : `b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee`. Tokenizer : `35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584`.

Le Mac calcule en parallèle le même mélange final avec coordonnées masquées en float32. L’ancienne archive sans FINAL reste une référence de calibration distincte : ne pas mélanger les deux dossiers ni leurs résultats. Le run à proposer à la soumission sera celui du PC en CUDA float16. Le mentor pourra le réentraîner sur T4 ; la tolérance de ±0,02 BPB et l’identité du dataset restent des exigences de vérification, pas une garantie déjà établie.

Sources officielles consultées le 10/11/2026 : [carte NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/), [matrice PyTorch 2.15 CUDA Windows/Blackwell/Turing](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [index Windows CUDA 13.2](https://download.pytorch.org/whl/cu132/torch/), [démarrage de Codex](https://developers.openai.com/learn/developers-codex-plugin).
