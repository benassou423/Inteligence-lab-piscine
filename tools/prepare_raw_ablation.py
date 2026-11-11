"""Préparer les témoins bruts de l'ablation, sans dupliquer 18 Go sur le disque.

Deux tirages de 100 Mo par hash d'ID (même méthode que le banc). Les IDs réservés
sont exclus AVANT sélection ; les textes sélectionnés passent ensuite uniquement
par la protection de validation. Pas d'encodage réparé, de filtre langue, qualité,
reflow ou déduplication dans ces témoins. Ils ne sont pas destinés à publication.
"""
import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from bench.config import load_run_config
from bench.mix import build_mix
from pipeline.clean.heldout import heldout_files, load_ids
from pipeline.io import write_docs
from pipeline.run import ROOT, Runner
from pipeline.stats import write_json
from tools.check_contamination import fast_check


SOURCES = ['wikipedia_fr', 'wikinews_fr', 'wikisource_fr', 'gutenberg_fr',
           'assemblee_nationale', 'fineweb2_fra', 'commoncrawl']
TOKENIZER = 'data/mixes/proxy_wiki20261001_fr065/tokenizer.json'


def prepare(seed, workers=4):
    name = f'ablation_brut_s{seed}'
    audit = Path('data/stages/complet_20261110_audit') / name
    audit.mkdir(parents=True, exist_ok=True)
    mix = load_run_config(overrides={'seed': seed, 'mix': {
        'sources': [{'name': s, 'path': f'data/raw/{s}/part-*.jsonl.zst'} for s in SOURCES],
        'max_bytes': 100_000_000, 'order': 'shuffle', 'workers': workers,
        'tokenizer_sample_bytes': 1_000_000}})['mix']
    excluded = load_ids(heldout_files('data/val/*/heldout_ids.txt', ROOT))
    manifest = build_mix(mix, audit / 'preselection', seed, excluded_ids=excluded, keep_metadata=True)
    groups = defaultdict(list)
    with gzip.open(audit / 'preselection' / 'train.jsonl.gz', 'rt') as f:
        for line in f:
            doc = json.loads(line)
            if doc['id'] in excluded or doc['source'] not in SOURCES:
                raise RuntimeError('Sélection brute invalide')
            groups[doc['source']].append(doc)
    for source, docs in groups.items():
        write_docs(audit / 'input' / source / 'part-00000.jsonl.zst', docs)
    cfg = {'name': name, 'input': str(audit / 'input'), 'sources': SOURCES,
           'source_order': SOURCES, 'prune_intermediate_files': True,
           'stages': [{'name': 'exclude_heldout', 'params': {'glob': 'data/val/*/heldout_ids.txt'}},
                      {'name': 'decontaminate', 'params': {'against': 'data/validation.jsonl',
                          'doc_min_coverage': .20, 'formula_min_docs': 20, 'mask_lines': True}}]}
    pipeline_config = audit / 'pipeline.yaml'
    pipeline_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    if Runner(pipeline_config, workers=workers).run() != 0:
        raise RuntimeError('Préparation brute interrompue')
    final = Path('data/stages') / name / '02_decontaminate'
    contamination = fast_check(final, workers=workers)
    write_json(audit / 'contamination.json', contamination)
    if contamination['n_signales']:
        raise RuntimeError('Contamination non nulle : entraînement interdit')
    run_cfg = {'run': name, 'seed': seed,
               'notes': 'Ablation brute avec protections validation, tokenizer figé, budget 25M ; aucun filtre qualité/langue/doublon.',
               'mix': {'sources': [{'name': name, 'pipeline': name, 'stage': '02_decontaminate'}],
                       'max_bytes': 100_000_000, 'order': 'shuffle', 'tokenizer_sample_bytes': 1_000_000},
               'tokenizer': {'path': TOKENIZER},
               'train': {'mode': 'proxy', 'debug_budget': 25_000_000, 'micro_batch': 16, 'device': 'mps', 'compile': False}}
    bench_config = Path('configs/bench') / f'{name}.yaml'
    bench_config.write_text(yaml.safe_dump(run_cfg, sort_keys=False, allow_unicode=True))
    command = [sys.executable, '-u', '-m', 'bench.run', '--config', str(bench_config), '--until', 'prepare']
    proc = subprocess.run(command)
    if proc.returncode:
        raise RuntimeError('Préparation du flux brut échouée')
    from bench.run import stream_coverage
    loaded = load_run_config(bench_config)
    coverage = stream_coverage(loaded)
    if coverage['passes'] > 1:
        raise RuntimeError('Flux brut trop court ; ne pas entraîner avec répétition involontaire')
    result = {'seed': seed, 'run': name, 'pipeline_config': str(pipeline_config),
              'bench_config': str(bench_config), 'preselection': manifest,
              'contamination_flagged': 0, 'coverage': coverage, 'prepared': True,
              'human_annotation': False, 'trained': False}
    write_json(audit / 'preparation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'preselection'}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1337,2027')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for link in ['data/raw', 'data/stages', 'data/mixes']:
        if not Path(link).is_symlink():
            raise RuntimeError(f'Lien nosync absent : {link}')
    for seed in map(int, args.seeds.split(',')):
        prepare(seed, args.workers)


if __name__ == '__main__':
    main()
