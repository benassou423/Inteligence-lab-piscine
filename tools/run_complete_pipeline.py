"""Lancer la variante complète après son essai relu, puis contrôler la contamination.

    nohup uv run python -u -m tools.run_complete_pipeline > data/stages/logs/complet-20261110-runner.log 2>&1 &

Ne change ni la référence ni la recette d'entraînement. Aucun entraînement ni
publication automatique : les paires retirées devront être relues après le run.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline.stats import write_json


AUDIT = Path('data/stages/complet_20261110_audit')
CONFIG = 'configs/pipeline/complet_20261110.yaml'
CODE_FILES = ['pipeline/run.py', 'pipeline/clean/privacy.py', 'pipeline/clean/quality.py',
              'pipeline/dedup/minhash.py', CONFIG]
LINKS = ['data/raw', 'data/stages', 'data/mixes', 'data/models', 'data/moliere', 'runs', '.venv']
REFERENCE = Path('data/stages/nettoye/08_exact_dedup')


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def signatures():
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(REFERENCE.glob('*/part-*.jsonl.zst'))}


def verify_kit():
    reference = Path('../Piscine-kit')
    roots = ['ftb', 'config', 'corpus', 'examples', 'notebooks', 'data/validation.jsonl', 'tests/test_kit.py']
    files = []
    for root in roots:
        path = reference / root
        if not path.exists():
            continue
        files.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.name != '.DS_Store'])
    if not files:
        raise RuntimeError('Copie de référence du kit introuvable')
    for src in files:
        target = src.relative_to(reference)
        if not target.is_file() or sha(src) != sha(target):
            raise RuntimeError(f'Kit différent : {target}')
    for name in LINKS:
        p = Path(name)
        if not p.is_symlink() or not str(p.readlink()).endswith('.nosync'):
            raise RuntimeError(f'Lien nosync absent ou différent : {name}')
    protected = ['pipeline/io.py', 'pyproject.toml', 'uv.lock', 'Makefile', 'docs/ARCHITECTURE.md']
    result = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *protected], capture_output=True)
    if result.returncode:
        raise RuntimeError('Un fichier protégé diffère de HEAD')
    return {'reference_files_identical': len(files), 'links_preserved': len(LINKS), 'protected_diff_empty': True}


def preflight():
    review = json.loads((AUDIT / 'pilot_review.json').read_text())
    if not review.get('approved_for_full_run') or review.get('reviewer') != 'Codex (IA)':
        raise RuntimeError('Essai non relu ou provenance de relecture ambiguë')
    current = {p: sha(p) for p in CODE_FILES}
    if review['code_sha256'] != current or review['pilot_report_sha256'] != sha(AUDIT / 'pilot_report.json'):
        raise RuntimeError('Le code ou le rapport ont changé depuis la relecture')
    report = json.loads((AUDIT / 'pilot_report.json').read_text())
    if report['documents'] != 2000 or not all(report['gates'].values()):
        raise RuntimeError('Critères de l’essai non satisfaits')
    tests = json.loads((AUDIT / 'tests.json').read_text())
    if tests['returncode'] != 0:
        raise RuntimeError('Les tests ne passent pas')
    return {'kit': verify_kit(), 'code_sha256': current, 'reference_signatures': signatures()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    before = preflight()
    if args.check_only:
        print(json.dumps({k: v for k, v in before.items() if k != 'reference_signatures'}, indent=2))
        return 0
    status_path = AUDIT / 'full_status.json'
    state = {'pid': os.getpid(), 'started': datetime.now().astimezone().isoformat(),
             'state': 'pipeline', 'complete': False, **before}
    write_json(status_path, state)
    t0 = time.monotonic()
    try:
        with Path('data/stages/logs/complet-20261110-full.log').open('a') as log:
            proc = subprocess.run([sys.executable, '-u', '-m', 'pipeline.run', '--config', CONFIG,
                                   '--workers', str(args.workers)], stdout=log, stderr=subprocess.STDOUT)
        state['pipeline_returncode'] = proc.returncode
        if proc.returncode:
            raise RuntimeError('Échec du pipeline ; voir complet-20261110-full.log')
        state['state'] = 'contamination'
        write_json(status_path, state)
        from tools.check_contamination import fast_check
        final = Path('data/stages/complet_20261110/06_exact_dedup')
        contamination = fast_check(final, workers=args.workers)
        write_json(AUDIT / 'contamination.json', contamination)
        if contamination['n_signales'] != 0:
            raise RuntimeError('Contamination non nulle : ne pas entraîner')
        state['kit_after'] = verify_kit()
        if signatures() != before['reference_signatures']:
            raise RuntimeError('Le corpus de référence a changé pendant le run')
        state.update(state='done', complete=True, contamination_flagged=0,
                     seconds=time.monotonic() - t0, pair_review_required=True)
    except Exception as exc:
        state.update(state='failed', error=str(exc), seconds=time.monotonic() - t0)
        raise
    finally:
        state['updated'] = datetime.now().astimezone().isoformat()
        write_json(status_path, state)
    print(json.dumps({k: v for k, v in state.items() if k != 'reference_signatures'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
