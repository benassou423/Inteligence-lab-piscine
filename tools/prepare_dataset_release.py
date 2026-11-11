"""Prépare les fichiers locaux de diffusion ; aucune publication distante."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='full_reference_masked_20261110')
    args = parser.parse_args()
    mix = Path('data/mixes') / args.run
    base = Path('data/stages/finalisation_20261110')
    masked = args.run == 'full_reference_masked_20261110'
    release = base / ('dataset_release_masked' if masked else 'dataset_release')
    release.mkdir(exist_ok=True)
    files = {'train.jsonl.gz': mix / 'train.jsonl.gz', 'tokenizer.json': mix / 'tokenizer.json',
             'manifest.json': mix / 'manifest.json', 'expected_data_meta.json': mix / 'prepared/data_meta.json',
             'provenance.jsonl.gz': mix / 'provenance.jsonl.gz', 'provenance_report.json': mix / 'provenance_report.json',
             'README.md': Path('docs/remise/DATASET_CARD.md'), 'DATASHEET.md': Path('docs/remise/DATASHEET.md'),
             'datasheet.pdf': Path('output/pdf/datasheet.pdf')}
    manifest = json.loads((mix / 'manifest.json').read_text())
    provenance = json.loads((mix / 'provenance_report.json').read_text())
    assert sha(files['train.jsonl.gz']) == manifest['train_sha256'] == provenance['train_gz_sha256']
    assert provenance['documents'] == manifest['documents'] == 72612
    assert provenance['missing_urls'] == provenance['missing_declared_licenses'] == 0
    hashes = {}
    for name, source in files.items():
        target = release / name
        shutil.copyfile(source, target)
        hashes[name] = sha(target)
        assert hashes[name] == sha(source)
    card = (release / 'README.md').read_text()
    # Une configuration explicite empêche l'index de provenance d'être chargé
    # comme s'il s'agissait d'un second fichier d'entraînement.
    if 'configs:' not in card:
        card = card.replace('license: other\n', 'license: other\nconfigs:\n  - config_name: default\n    data_files:\n      - split: train\n        path: train.jsonl.gz\n')
    (release / 'README.md').write_text(card)
    hashes['README.md'] = sha(release / 'README.md')
    (release / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    report = {'folder': str(release), 'files_sha256': hashes, 'documents': manifest['documents'],
              'train_bytes': manifest['bytes'], 'dataset_sha256': json.loads(files['expected_data_meta.json'].read_text())['dataset_sha256'],
              'published': False, 'public_url': None, 'status': 'local_draft',
              'remaining': ['contrôle des fichiers à diffuser', 'résultat officiel final après masquage',
                            'publication distante et vérification du téléchargement']}
    state = base / 'masked_training_status.json'
    if masked and state.exists() and json.loads(state.read_text()).get('complete'):
        report['remaining'].remove('résultat officiel final après masquage')
    (base / ('dataset_release_masked_status.json' if masked else 'dataset_release_status.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
