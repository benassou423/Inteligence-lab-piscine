"""Masquer les coordonnées du mélange retenu, sans retirer ses faits encyclopédiques.

Le mélange déjà évalué et sa provenance sont lus dans leur ordre d'origine.
Les coordonnées structurées sont masquées ; les nouveaux doublons exacts sont
retirés. Les fichiers sources restent intacts. Ce script prépare puis contrôle
le flux final ; l'entraînement reste une étape distincte.
"""
import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path

import yaml

from pipeline.clean.privacy import mask_text
from pipeline.dedup.exact import text_key
from pipeline.stats import write_json
from tools.reproduce import check_kit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('data/mixes/full_reference_20261110')
OUTPUT = Path('data/stages/finalisation_20261110/masked_reference')
RUN = 'full_reference_masked_20261110'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@contextmanager
def gz_write(path):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as text:
                yield text


def transform(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seen, counts = set(), Counter()
    by_source = defaultdict(Counter)
    dropped = []
    data_tmp, prov_tmp = output / 'input.tmp.gz', output / 'provenance.tmp.gz'
    with gzip.open(source / 'train.jsonl.gz', 'rt', encoding='utf-8') as texts, \
         gzip.open(source / 'provenance.jsonl.gz', 'rt', encoding='utf-8') as provenance, \
         gz_write(data_tmp) as data_out, gz_write(prov_tmp) as prov_out:
        for original_row, (line, origin) in enumerate(zip_longest(texts, provenance)):
            if line is None or origin is None:
                raise ValueError('Texte et provenance ont des longueurs différentes')
            text, info = json.loads(line)['text'], json.loads(origin)
            original_sha = hashlib.sha256(text.encode()).hexdigest()
            if info['row_zero_based'] != original_row or info['text_sha256'] != original_sha:
                raise ValueError(f'Provenance non alignée : ligne {original_row}')
            after, masks = mask_text(text, info['source'])
            key = text_key(after)
            counts['documents_in'] += 1
            counts['original_text_bytes'] += len(text.encode())
            if key in seen:
                dropped.append({'original_row': original_row, 'id': info['id']})
                counts['new_exact_duplicates'] += 1
                continue
            seen.add(key)
            index = counts['documents_out']
            row = {k: info[k] for k in ['id', 'source', 'domain', 'url', 'date', 'license']}
            row['text'] = after
            data_out.write(json.dumps(row, ensure_ascii=False) + '\n')
            info.update(original_row_zero_based=original_row, row_zero_based=index,
                        original_text_sha256=original_sha, text_sha256=hashlib.sha256(after.encode()).hexdigest(),
                        transformation={'structured_masks': dict(masks)})
            prov_out.write(json.dumps(info, ensure_ascii=False) + '\n')
            counts['documents_out'] += 1
            counts['documents_masked'] += bool(masks)
            counts['text_bytes'] += len(after.encode())
            counts.update({f'masks_{k}': v for k, v in masks.items()})
            by_source[info['source']].update(documents=1, bytes=len(after.encode()), masked_documents=bool(masks))
    data_tmp.replace(output / 'input.jsonl.gz')
    prov_tmp.replace(output / 'provenance.jsonl.gz')
    result = {'counts': dict(counts), 'by_source': {s: dict(v) for s, v in by_source.items()},
              'dropped_new_exact_duplicates': dropped, 'order': 'original order, minus new exact duplicates',
              'input_file_sha256': sha(output / 'input.jsonl.gz'),
              'provenance_sha256': sha(output / 'provenance.jsonl.gz'),
              'parent_train_sha256': sha(source / 'train.jsonl.gz'),
              'privacy_code_sha256': sha('pipeline/clean/privacy.py'), 'original_data_modified': False}
    write_json(output / 'transformation.json', result)
    return result


def command(module, args, log_name):
    path = Path('data/stages/logs') / log_name
    with path.open('w') as log:
        result = subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} a échoué, voir {path}')


def verify_provenance(mix):
    mix = Path(mix)
    manifest = json.loads((mix / 'manifest.json').read_text())
    count, missing_urls, missing_licenses = 0, 0, 0
    digests, sources = set(), Counter()
    with gzip.open(mix / 'train.jsonl.gz', 'rt', encoding='utf-8') as texts, \
         gzip.open(mix / 'provenance.jsonl.gz', 'rt', encoding='utf-8') as refs:
        for line, origin in zip_longest(texts, refs):
            if line is None or origin is None:
                raise ValueError('Provenance finale incomplète')
            info = json.loads(origin)
            digest = hashlib.sha256(json.loads(line)['text'].encode()).hexdigest()
            if info['row_zero_based'] != count or digest != info['text_sha256']:
                raise ValueError(f'Provenance finale divergente : {count}')
            count += 1
            digests.add(digest)
            sources[info['source']] += 1
            missing_urls += not bool(info.get('url'))
            missing_licenses += not bool(info.get('license'))
    if count != manifest['documents'] or dict(sources) != {s: v['documents'] for s, v in manifest['by_source'].items()}:
        raise ValueError('Volumes de provenance différents du manifeste')
    report = {'documents': count, 'unique_texts': len(digests), 'by_source': dict(sources),
              'all_rows_matched_by_exact_text_sha256': True,
              'original_texts_linked_via_original_sha256': True,
              'missing_urls': missing_urls, 'missing_declared_licenses': missing_licenses,
              'provenance_sha256': sha(mix / 'provenance.jsonl.gz'),
              'train_gz_sha256': sha(mix / 'train.jsonl.gz')}
    if report['train_gz_sha256'] != manifest['train_sha256']:
        raise ValueError('Empreinte du texte différente du manifeste')
    write_json(mix / 'provenance_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name in ['data/stages', 'data/mixes', 'runs']:
        if not Path(name).is_symlink():
            raise RuntimeError(f'Lien nosync absent : {name}')
    check_kit(ROOT)
    result = transform(SOURCE, OUTPUT)
    config = {'run': RUN, 'seed': 1337,
              'notes': 'Référence retenue, coordonnées structurées masquées et nouveaux doublons exacts retirés ; ordre conservé.',
              'mix': {'sources': [{'name': 'reference_masked', 'path': str(OUTPUT / 'input.jsonl.gz')}],
                      'order': 'as_is', 'tokenizer_sample_bytes': 1000000},
              'tokenizer': {'path': str(SOURCE / 'tokenizer.json')},
              'train': {'mode': 'full', 'device': 'mps', 'micro_batch': 16, 'compile': False}}
    config_path = Path('configs/bench') / f'{RUN}.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    command('ftb.contamination', ['--inputs', OUTPUT / 'input.jsonl.gz', '--out', OUTPUT / 'contamination.json'],
            'final-masked-contamination.log')
    contamination = json.loads((OUTPUT / 'contamination.json').read_text())
    if contamination['n_signales'] != 0:
        raise RuntimeError('Contamination non nulle : ne pas entraîner')
    command('bench.run', ['--config', config_path, '--until', 'prepare'], 'final-masked-prepare.log')
    mix = Path('data/mixes') / RUN
    # Vérifier que le banc a réellement conservé le texte et l'ordre de la provenance.
    count = 0
    with gzip.open(mix / 'train.jsonl.gz', 'rt', encoding='utf-8') as texts, \
         gzip.open(OUTPUT / 'provenance.jsonl.gz', 'rt', encoding='utf-8') as refs:
        for line, origin in zip_longest(texts, refs):
            if line is None or origin is None:
                raise RuntimeError('Provenance finale incomplète')
            info = json.loads(origin)
            if info['row_zero_based'] != count or hashlib.sha256(json.loads(line)['text'].encode()).hexdigest() != info['text_sha256']:
                raise RuntimeError('Texte ou ordre modifié par le banc')
            count += 1
    shutil.copy2(OUTPUT / 'provenance.jsonl.gz', mix / 'provenance.jsonl.gz')
    verify_provenance(mix)
    meta = json.loads((mix / 'prepared/data_meta.json').read_text())
    if meta['n_tokens'] < 100007936:
        raise RuntimeError('Flux final insuffisant pour le budget officiel')
    check_kit(ROOT)
    result.update(prepared=True, contamination_flagged=0, run=RUN, dataset_sha256=meta['dataset_sha256'],
                  tokenizer_sha256=meta['tokenizer_sha256'], dataset_tokens=meta['n_tokens'], provenance_rows=count)
    write_json(OUTPUT / 'preparation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
