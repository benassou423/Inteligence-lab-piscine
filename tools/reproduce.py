"""Une commande pour la collecte et le pipeline, depuis la racine du projet.

    uv run --no-project python tools/reproduce.py --recipe final

L'entrée ne dépend que de Python standard : les liens nosync sont créés AVANT
que `uv run` ne prépare les dépendances du projet. Un dossier ordinaire à la
place d'un lien est refusé. Ne publie rien, n'entraîne pas de modèle par défaut.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKS = ['data/raw', 'data/stages', 'data/mixes', 'data/models', 'data/moliere', 'runs', '.venv']


def unavailable_cache(path):
    # macOS sys/stat.h : SF_DATALESS = 0x40000000. Le nom existe, mais la
    # lecture attendrait un fournisseur cloud. Ne pas confondre UF_COMPRESSED.
    return sys.platform == 'darwin' and bool(getattr(path.stat(), 'st_flags', 0) & 0x40000000)


def repair_cache(root, pipelines=('nettoye', 'complet_20261110'), unavailable=unavailable_cache):
    """Conserver à part les seules métadonnées de cache indisponibles.

    Les bruts et les parts de corpus ne sont jamais déplacés. Une réception
    absente force naturellement le moteur à régénérer le résultat concerné.
    """
    stage_root = root / 'data/stages'
    quarantine = stage_root / 'reproduction/cache_placeholders' / uuid.uuid4().hex[:12]
    selected = []
    for pipeline in pipelines:
        directory = stage_root / pipeline
        for path in directory.glob('[0-9][0-9]_*/_units/**/*'):
            if path.is_symlink() or not path.is_file():
                continue
            if not path.resolve().is_relative_to(directory.resolve()):
                raise RuntimeError('Cache hors du pipeline attendu')
            if unavailable(path):
                selected.append((path, quarantine / pipeline / path.relative_to(directory)))
    moved = []
    for source, destination in selected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append({'original': str(source), 'preserved': str(destination)})
    if moved:
        (quarantine / 'manifest.json').write_text(json.dumps({'reason': 'SF_DATALESS cache placeholders',
                'original_corpus_modified': False, 'moved': moved}, indent=2))
        print(f'{len(moved)} métadonnées de cache préservées à part ; régénération automatique.', flush=True)
    return moved


def ensure_links(root):
    # Valider l'ensemble avant de créer le premier lien ; ne déplacer aucun dossier.
    for name in LINKS:
        path = root / name
        if path.exists() and not path.is_symlink():
            if os.name == 'nt' and path.is_junction():
                if path.resolve() != path.with_name(path.name + '.nosync').resolve():
                    raise RuntimeError(f'{name} pointe vers une cible inattendue')
                continue
            raise RuntimeError(f'{name} est un dossier ordinaire : arrêt sans remplacement')
        if path.is_symlink() and path.resolve() != path.with_name(path.name + '.nosync').resolve():
            raise RuntimeError(f'{name} pointe vers une cible inattendue')
    for name in LINKS:
        path = root / name
        target = path.with_name(path.name + '.nosync')
        target.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            continue
        if os.name == 'nt':
            # Même principe que le transfert Windows : jonction, sans droits admin.
            proc = subprocess.run(['cmd', '/c', 'mklink', '/J', str(path), str(target)], capture_output=True)
            if proc.returncode:
                raise RuntimeError(f'Création de la jonction impossible : {name}')
        else:
            path.symlink_to(target.name, target_is_directory=True)


def check_kit(root):
    expected = json.loads((root / 'docs/remise/KIT_SHA256.json').read_text())
    for name, digest in expected['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Kit différent : {name}')
    for folder in ['ftb', 'config', 'corpus', 'examples', 'notebooks']:
        extras = [p for p in (root / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.name != '.DS_Store' and str(p.relative_to(root)) not in expected['files']]
        if extras:
            raise RuntimeError(f'Fichier ajouté au kit : {extras[0].relative_to(root)}')
    return len(expected['files'])


def execute(module, args):
    command = ['uv', 'run', 'python', '-u', '-m', module, *args]
    print('$ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['reference', 'complete', 'final'], default='final')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--skip-collect', action='store_true', help='Rejouer seulement le traitement de sources déjà collectées')
    parser.add_argument('--foreground', action='store_true', help='Mode interne du processus détaché')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Lancer depuis Inteligence-lab-piscine/')
    if args.check_only:
        print(json.dumps({'kit_files_identical': check_kit(ROOT),
                          'links': {name: str((ROOT / name).readlink()) if (ROOT / name).is_symlink()
                                    else 'absent ou jonction Windows à vérifier' for name in LINKS}}, indent=2))
        return
    ensure_links(ROOT)
    kit = check_kit(ROOT)
    if os.name == 'nt':
        raise RuntimeError('Reproduction intégrale : utiliser macOS/Linux ; entraînement Windows : voir PASSAGE_PC_RTX5090.md')
    if not args.foreground:
        logs = ROOT / 'data/stages/logs'
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = logs / f'reproduction-{args.recipe}-{stamp}.log'
        command = ['nohup', sys.executable, str(Path(__file__).resolve()), '--recipe', args.recipe,
                   '--workers', str(args.workers), '--foreground']
        if args.skip_collect:
            command.append('--skip-collect')
        with log_path.open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(log_path), 'complete': False}, indent=2))
        return
    if not (ROOT / '.venv.nosync/pyvenv.cfg').is_file():
        subprocess.run(['uv', 'venv', '--python', '3.12', str(ROOT / '.venv.nosync')], cwd=ROOT, check=True)
    audit = ROOT / 'data/stages/reproduction' / args.recipe
    audit.mkdir(parents=True, exist_ok=True)
    status = {'started': datetime.now().astimezone().isoformat(), 'complete': False,
              'recipe': args.recipe, 'kit_files_identical': kit, 'skip_collect': args.skip_collect}
    path = audit / 'status.json'
    path.write_text(json.dumps(status, indent=2))
    try:
        status['cache_placeholders_preserved'] = len(repair_cache(ROOT))
        if not args.skip_collect:
            execute('pipeline.collect', ['all'])
        execute('pipeline.run', ['--config', 'configs/pipeline/nettoye.yaml', '--workers', str(args.workers)])
        recipe = 'nettoye'
        if args.recipe == 'complete':
            execute('pipeline.run', ['--config', 'configs/pipeline/complet_20261110.yaml', '--workers', str(args.workers)])
            recipe = 'complet_20261110'
        if args.recipe == 'final':
            # Refaire le tokenizer depuis sa vraie recette, pas depuis un fichier
            # lourd absent d'un clone Git. Refuser une dérive des entrées figées.
            execute('bench.run', ['--config', 'configs/bench/proxy_wiki20261001_fr065.yaml', '--until', 'prepare'])
            tokenizer = ROOT / 'data/mixes/proxy_wiki20261001_fr065/tokenizer.json'
            if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != '35c0f184dade57df1dec18a4bcb05e42dda7f0ad528cb71e398a33b9e243d584':
                raise RuntimeError('Tokenizer différent de la référence figée')
            execute('bench.run', ['--config', 'configs/bench/full_reference_20261110.yaml', '--until', 'prepare'])
            parent = json.loads((ROOT / 'data/mixes/full_reference_20261110/prepared/data_meta.json').read_text())
            if parent['dataset_sha256'] != 'e117b978a787c8410ed9e56d916fd02b65080825805cd38559cb41f46aa06498':
                raise RuntimeError('Mélange de référence différent : vérifier la collecte et le nettoyage')
            execute('tools.export_mix_provenance', [])
            execute('tools.prepare_masked_reference', [])
            final = json.loads((ROOT / 'data/mixes/full_reference_masked_20261110/prepared/data_meta.json').read_text())
            if final['dataset_sha256'] != 'b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee':
                raise RuntimeError('Mélange final différent : vérifier les transformations')
            status.update(output_mix='full_reference_masked_20261110', dataset_sha256=final['dataset_sha256'])
        else:
            # Le contrôleur renvoie un code non nul en cas de contamination.
            execute('tools.check_contamination', ['--config', recipe, '--fast', '--workers', str(args.workers)])
        check_kit(ROOT)
        status.update(complete=True, output_pipeline=recipe)
    except Exception as exc:
        status['error'] = str(exc)
        raise
    finally:
        status['updated'] = datetime.now().astimezone().isoformat()
        path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
