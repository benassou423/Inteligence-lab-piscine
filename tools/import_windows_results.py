"""Importe le retour RTX après contrôle ; prépare un dossier local de soumission."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

RUN = 'full_reference_rtx5090_20261110'
RUN_MIX = {RUN: 'full_reference_20261110',
           'full_reference_masked_rtx5090_20261110': 'full_reference_masked_20261110'}
FILES = ['ckpt.pt', 'tokenizer.json', 'train_log.jsonl', 'data_meta.json', 'samples.txt', 'eval.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_result(archive, dest, run_name=RUN):
    if run_name not in RUN_MIX:
        raise ValueError('Nom de run GPU inconnu')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'Archive endommagée'
        assert sum(i.file_size for i in z.infolist()) < 1_000_000_000, 'Archive anormalement grande'
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename:
                raise ValueError('Chemin dangereux dans les résultats')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Lien inattendu dans les résultats')
            allowed = info.filename.startswith('runs/' + run_name + '/') or info.filename.startswith('data/stages/gpu_handoff/')
            if not allowed:
                raise ValueError(f'Fichier étranger au résultat GPU : {info.filename}')
        z.extractall(dest)
    run = Path(dest) / 'runs' / run_name
    if not all((run / f).is_file() for f in FILES):
        raise ValueError('Le résultat GPU est incomplet')
    return run


def validate_meta(meta, expected, config):
    if not meta.get('official') or meta.get('tokens_seen') != 100007936:
        raise ValueError('Le run ne respecte pas le budget officiel')
    if meta.get('precision') != 'torch.float16' or '5090' not in meta.get('device', ''):
        raise ValueError('Le résultat ne vient pas du calcul RTX 5090 float16 attendu')
    for k in ['dataset_sha256', 'tokenizer_sha256']:
        if meta.get(k) != expected[k]:
            raise ValueError(f'Empreinte différente : {k}')
    for k, value in config['train'].items():
        if meta.get('train_config', {}).get(k) != value:
            raise ValueError(f'Hyperparamètre différent : {k}')
    if not math.isfinite(meta.get('val_bpb', float('nan'))):
        raise ValueError('Score manquant ou non fini')


def main():
    import torch
    from ftb import load_config
    from ftb.evaluate import check_model_config
    from ftb.model import build_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--dataset-url', default='')
    args = parser.parse_args()
    base = Path('data/stages/finalisation_20261110')
    base.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names = {PurePosixPath(i.filename).parts[1] for i in archive.infolist()
                 if len(PurePosixPath(i.filename).parts) > 1 and PurePosixPath(i.filename).parts[0] == 'runs'}
    if len(names) != 1 or not names <= RUN_MIX.keys():
        raise ValueError('Run GPU inconnu ou mélange de plusieurs runs')
    run_name = next(iter(names))
    mix_name = RUN_MIX[run_name]
    expected = read(Path('data/mixes') / mix_name / 'prepared/data_meta.json')
    config = load_config()
    with tempfile.TemporaryDirectory(prefix='gpu-import-', dir=base) as tmp:
        run = extract_result(args.archive, tmp, run_name)
        checkpoint = torch.load(run / 'ckpt.pt', map_location='cpu', weights_only=True)
        meta = checkpoint['meta']
        validate_meta(meta, expected, config)
        check_model_config(checkpoint['model_config'])
        model = build_model(config['model'])
        model.load_state_dict(checkpoint['model'], strict=True)
        assert model.num_params() == 10166400
        assert sha(run / 'tokenizer.json') == expected['tokenizer_sha256']
        data = read(run / 'data_meta.json')
        assert data['dataset_sha256'] == expected['dataset_sha256']
        ev = read(run / 'eval.json')
        assert ev['tokenizer_sha256'] == expected['tokenizer_sha256'] and ev['n_documents'] == 230
        assert abs(ev['bpb'] - meta['val_bpb']) < 1e-5
        logs = [json.loads(s) for s in (run / 'train_log.jsonl').read_text(encoding='utf-8').splitlines()]
        assert any(r.get('tokens') == 100007936 for r in logs)
        assert logs[-1]['step'] == config['train']['max_steps']
        target = Path('runs') / run_name
        if target.exists():
            assert all((target / f).exists() and sha(target / f) == sha(run / f) for f in FILES), 'Un résultat différent existe déjà'
        else:
            shutil.copytree(run, target)
        proof = {'imported': True, 'run': str(target), 'meta': meta, 'evaluation': ev,
                 'source_archive_sha256': sha(args.archive), 'dataset_url': args.dataset_url,
                 'remote_submission_performed': False, 'publication_ready': False,
                 'files_sha256': {f: sha(target / f) for f in FILES}}
        filename = 'gpu_import_masked_status.json' if mix_name == 'full_reference_masked_20261110' else 'gpu_import_status.json'
        (base / filename).write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    dest = base / ('submission_draft_masked' if mix_name == 'full_reference_masked_20261110' else 'submission_draft')
    subprocess.run([sys.executable, '-m', 'ftb.submit', '--run', str(target), '--participant', 'Benjamin Assouline',
                    '--dataset-url', args.dataset_url, '--note', 'Référence nettoye, dump frwiki 20261001, 100M CUDA RTX5090',
                    '--dest', str(dest)], check=True)
    print('Dossier local préparé. Vérifier l’accès au dataset et obtenir l’accord de commit/push avant le dépôt distant.')


if __name__ == '__main__':
    main()
