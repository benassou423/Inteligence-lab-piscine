"""Archive de transfert locale : état de travail et sélection complète, aucun secret."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['reference', 'final'], default='final')
    parser.add_argument('--replace', action='store_true', help='Remplacer uniquement une archive de travail déjà générée')
    args = parser.parse_args()
    mix_name = 'full_reference_masked_20261110' if args.target == 'final' else 'full_reference_20261110'
    run_name = 'full_reference_masked_rtx5090_20261110' if args.target == 'final' else 'full_reference_rtx5090_20261110'
    result_name = 'RESULTATS_RTX5090_MASKED.zip' if args.target == 'final' else 'RESULTATS_RTX5090.zip'
    base = ROOT / 'data/stages/finalisation_20261110/transfer'
    base.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    entries = {}
    for name in files:
        if not name or name.startswith(('submissions/', 'data/annotations/')):
            continue
        path = ROOT / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 10_000_000:
            entries[name] = path
    entries['AGENTS.md'] = ROOT / 'AGENTS.md'
    mix = ROOT / 'data/mixes' / mix_name
    for name in ['train.jsonl.gz', 'tokenizer.json', 'manifest.json', 'provenance.jsonl.gz', 'provenance_report.json']:
        entries['data/mixes.nosync/' + mix_name + '/' + name] = mix / name
    entries['data/mixes.nosync/' + mix_name + '/expected_data_meta.json'] = mix / 'prepared/data_meta.json'
    contamination_path = ('data/stages/finalisation_20261110/masked_reference/contamination.json'
                          if args.target == 'final' else 'data/stages/finalisation_20261110/mix_contamination.json')
    assert json.loads((ROOT / contamination_path).read_text())['n_signales'] == 0
    entries[contamination_path.replace('data/stages/', 'data/stages.nosync/')] = ROOT / contamination_path
    expected = json.loads((mix / 'prepared/data_meta.json').read_text())
    critical = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in entries.items()
                if name.startswith(('ftb/', 'config/', 'corpus/', 'data/mixes.nosync/')) or
                name in ['data/validation.jsonl', 'tests/test_kit.py', 'tools/windows_gpu.py']}
    manifest = {'dataset_sha256': expected['dataset_sha256'], 'tokenizer_sha256': expected['tokenizer_sha256'],
                'mix_name': mix_name, 'run_name': run_name,
                'budget_tokens_seen': 100007936, 'files_sha256': critical,
                'note': 'Les fichiers de documentation peuvent évoluer ; kit, données et lanceur sont vérifiés.'}
    pc = '''# Mission sur ce PC Windows

Lis AGENTS.md puis docs/remise/PASSAGE_PC_RTX5090.md. Les chemins Mac de la mémoire
sont historiques : ce dossier est la copie de travail Windows du projet.

Le rôle de ce PC est l'entraînement CUDA sur RTX 5090 et l'export des résultats.
Le Mac conserve les sources brutes, les archives, les comparaisons et le suivi des
livrables. N'efface aucun fichier du Mac et ne recrée pas les gros dossiers en
dossiers ordinaires. tools/windows_gpu.py setup crée leurs jonctions .nosync.

Vérifie TRANSFER_MANIFEST.json avant les opérations. Ne modifie pas le kit.
Utilise Python 3.12, PyTorch 2.15.0 CUDA 13.2 et tokenizers 0.23.3. Exécute
un vrai test de calcul GPU avant l'entraînement. Pas de --debug-budget.
Le run doit voir 100 007 936 tokens, avec le hash attendu dans le manifeste.
Lance et suis le travail en arrière-plan, garde les journaux et la reprise.

Remets RESULTATS_RTX5090.zip. Aucun commit ni push n'est autorisé. Aucune nouvelle
publication de dataset ni soumission distante n'est autorisée par ce transfert.
Le but global reste actif sur le Mac ; ne le déclare pas atteint depuis ce seul run.
'''
    pc = pc.replace('RESULTATS_RTX5090.zip', result_name)
    pc += f'\nMélange attendu : {mix_name}. Run : {run_name}.\nDataset SHA-256 : {expected["dataset_sha256"]}.\n'
    dest = base / ('FTB_RTX5090_FINAL_20261110.zip' if args.target == 'final' else 'FTB_RTX5090_20261110.zip')
    if dest.exists() and not args.replace:
        raise FileExistsError(f'Archive existante conservée : {dest}')
    with zipfile.ZipFile(dest.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(entries.items()):
            z.write(path, 'Inteligence-lab-piscine/' + name,
                    compress_type=zipfile.ZIP_STORED if name.endswith('.gz') else zipfile.ZIP_DEFLATED)
        z.writestr('Inteligence-lab-piscine/TRANSFER_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('Inteligence-lab-piscine/AGENTS_PC.md', pc)
    dest.with_suffix('.tmp').replace(dest)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert not any(name.startswith('/') or '..' in Path(name).parts for name in z.namelist())
        for name, expected_sha in critical.items():
            assert hashlib.sha256(z.read('Inteligence-lab-piscine/' + name)).hexdigest() == expected_sha
    proof = {'archive': str(dest), 'bytes': dest.stat().st_size, 'files': len(entries)+2,
             'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'critical_files_checked': len(critical),
             'dataset_sha256': manifest['dataset_sha256'], 'zip_integrity_passed': True, 'windows_gpu_executed': False}
    (base / ('verification_final.json' if args.target == 'final' else 'verification.json')).write_text(
        json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
