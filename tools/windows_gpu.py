"""Passage contrôlé du kit officiel sur le PC Windows de Ben (RTX 5090)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

LINKS = {'data/raw': 'raw.nosync', 'data/stages': 'stages.nosync',
         'data/mixes': 'mixes.nosync', 'data/models': 'models.nosync',
         'data/moliere': 'moliere.nosync', 'runs': 'runs.nosync', '.venv': '.venv.nosync'}
RUN = 'full_reference_rtx5090_20261110'
MIX = 'full_reference_20261110'
TARGETS = {MIX: RUN, 'full_reference_masked_20261110': 'full_reference_masked_rtx5090_20261110'}


def target_names(manifest):
    mix = manifest.get('mix_name', MIX)
    run_name = manifest.get('run_name', RUN)
    if mix not in TARGETS or TARGETS[mix] != run_name:
        raise ValueError('Couple mélange/run inattendu dans le manifeste')
    return mix, run_name


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'TRANSFER_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Fichier manquant ou modifié : {name}')
    return manifest


def init_links(root):
    root = Path(root)
    for name, target in LINKS.items():
        link = root / name
        dest = link.parent / target
        dest.mkdir(parents=True, exist_ok=True)
        is_link = link.is_symlink() or (hasattr(link, 'is_junction') and link.is_junction())
        if is_link:
            if link.resolve() != dest.resolve():
                raise ValueError(f'Lien vers une mauvaise cible : {name}')
            continue
        if link.exists():
            raise ValueError(f'{name} est un dossier normal ; ne pas le remplacer automatiquement')
        if os.name == 'nt':
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(dest)], check=True, capture_output=True)
        else:
            link.symlink_to(target, target_is_directory=True)


def command(args, root, log=None):
    print('Commande :', subprocess.list2cmdline(list(map(str, args))), flush=True)
    if log:
        with log.open('a', encoding='utf-8') as out:
            subprocess.run(list(map(str, args)), cwd=root, check=True, stdout=out, stderr=subprocess.STDOUT)
    else:
        subprocess.run(list(map(str, args)), cwd=root, check=True)


def setup(root):
    if os.name != 'nt':
        raise RuntimeError('Installation CUDA prévue sur Windows seulement')
    verify(root)
    init_links(root)
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('Installer uv depuis sa distribution officielle, puis relancer')
    python = root / '.venv.nosync/Scripts/python.exe'
    if not python.exists():
        command([uv, 'venv', '--python', '3.12', str(root / '.venv.nosync')], root)
    command([uv, 'pip', 'install', '--python', python, 'torch==2.15.0',
             '--index-url', 'https://download.pytorch.org/whl/cu132'], root)
    command([uv, 'pip', 'install', '--python', python, 'numpy==2.5.4',
             'tokenizers==0.23.3', 'regex==2026.9.29'], root)
    print('Installation prête. Lancer run avec le Python de .venv.nosync.', flush=True)


def run(root):
    import torch
    import tokenizers
    manifest = verify(root)
    mix_name, run_name = target_names(manifest)
    init_links(root)
    assert torch.__version__.split('+')[0] == '2.15.0', 'Version PyTorch différente'
    assert tokenizers.__version__ == '0.23.3', 'Version du tokenizer différente'
    assert torch.cuda.is_available(), 'CUDA indisponible : vérifier le pilote NVIDIA et la version de PyTorch'
    name = torch.cuda.get_device_name(0)
    assert '5090' in name, f'Carte inattendue : {name}'
    # Une opération réelle évite de conclure sur le seul indicateur is_available.
    x = torch.ones((32, 32), device='cuda', dtype=torch.float16, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()
    output = root / 'data/stages/gpu_handoff'
    output.mkdir(parents=True, exist_ok=True)
    probe = {'device': name, 'memory_bytes': torch.cuda.get_device_properties(0).total_memory,
             'capability': torch.cuda.get_device_capability(0), 'torch': str(torch.__version__),
             'cuda': torch.version.cuda, 'tokenizers': tokenizers.__version__, 'kernel_probe_passed': True}
    (output / 'gpu_probe.json').write_text(json.dumps(probe, indent=2) + '\n', encoding='utf-8')
    mix = root / 'data/mixes' / mix_name
    prepared = mix / 'prepared'
    command([sys.executable, '-m', 'ftb.prepare', '--inputs', mix / 'train.jsonl.gz',
             '--tokenizer', mix / 'tokenizer.json', '--out', prepared], root, output / 'prepare.log')
    meta = json.loads((prepared / 'data_meta.json').read_text(encoding='utf-8'))
    assert meta['dataset_sha256'] == manifest['dataset_sha256'], 'Empreinte du dataset différente du Mac'
    assert meta['tokenizer_sha256'] == manifest['tokenizer_sha256']
    assert meta['n_tokens'] >= 100007936
    run_dir = root / 'runs' / run_name
    if not (run_dir / 'ckpt.pt').exists():
        args = [sys.executable, '-m', 'ftb.train', '--data', prepared, '--tokenizer', mix / 'tokenizer.json',
                '--out', run_dir, '--device', 'cuda', '--micro-batch', '16']
        if (run_dir / 'state.pt').exists():
            args.append('--resume')
        command(args, root, output / 'train.log')
    checkpoint = torch.load(run_dir / 'ckpt.pt', map_location='cpu', weights_only=True)
    actual = checkpoint['meta']
    assert actual['official'] and actual['tokens_seen'] == 100007936
    assert actual['dataset_sha256'] == manifest['dataset_sha256']
    assert actual['tokenizer_sha256'] == manifest['tokenizer_sha256']
    assert actual['precision'] == 'torch.float16'
    command([sys.executable, '-m', 'ftb.evaluate', '--ckpt', run_dir / 'ckpt.pt',
             '--tokenizer', run_dir / 'tokenizer.json', '--device', 'cuda', '--out', run_dir / 'eval.json'],
            root, output / 'eval.log')
    verify(root)
    (output / 'completed.json').write_text(json.dumps({'complete': True, 'run': run_name, 'meta': actual}, indent=2) + '\n', encoding='utf-8')
    export(root)


def export(root):
    manifest = verify(root)
    mix_name, run_name = target_names(manifest)
    output = root / 'data/stages/gpu_handoff'
    complete = json.loads((output / 'completed.json').read_text(encoding='utf-8'))
    assert complete['complete'] and complete['meta']['official']
    assert complete['meta']['tokens_seen'] == 100007936
    assert complete['run'] == run_name and complete['meta']['dataset_sha256'] == manifest['dataset_sha256']
    for name in ['ckpt.pt', 'tokenizer.json', 'train_log.jsonl', 'data_meta.json', 'samples.txt', 'eval.json']:
        assert (root / 'runs' / run_name / name).is_file(), name
    files = list((root / 'runs' / run_name).glob('*')) + list(output.glob('*'))
    selected = [p for p in files if p.is_file() and p.suffix in {'.pt', '.json', '.jsonl', '.txt', '.log'}]
    dest = root / ('RESULTATS_RTX5090_MASKED.zip' if mix_name == 'full_reference_masked_20261110' else 'RESULTATS_RTX5090.zip')
    with zipfile.ZipFile(dest.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            z.write(p, p.relative_to(root))
    dest.with_suffix('.tmp').replace(dest)
    print(f'Résultats à rapporter sur le Mac : {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'init', 'setup', 'run', 'export'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    {'verify': verify, 'init': init_links, 'setup': setup, 'run': run, 'export': export}[args.action](root)


if __name__ == '__main__':
    main()
