"""Rattache chaque ligne du mélange aux métadonnées de son document d'origine."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

from pipeline.io import read_docs

_TARGETS = set()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def scan(path):
    rows = []
    for i, doc in enumerate(read_docs(path)):
        digest = text_hash(doc['text'])
        if digest in _TARGETS:
            rows.append({'text_sha256': digest, 'id': doc['id'], 'source': doc['source'],
                         'domain': doc.get('domain'), 'url': doc.get('url'), 'date': doc.get('date'),
                         'license': doc.get('license'), 'title': doc.get('meta', {}).get('title'),
                         'corpus_file': str(path), 'corpus_row_zero_based': i})
    return rows


def main():
    global _TARGETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mix', default='full_reference_20261110')
    args = parser.parse_args()
    mix = Path('data/mixes') / args.mix
    digests = []
    with gzip.open(mix / 'train.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            digests.append(text_hash(json.loads(line)['text']))
    _TARGETS = set(digests)
    found = {}
    files = sorted(Path('data/stages/nettoye/08_exact_dedup').glob('*/part-*.jsonl.zst'))
    with get_context('fork').Pool(8) as pool:
        for i, records in enumerate(pool.imap_unordered(scan, files), 1):
            for row in records:
                assert row['text_sha256'] not in found, 'Texte identique présent deux fois dans le corpus de référence'
                found[row['text_sha256']] = row
            if i % 25 == 0:
                print(f'Provenance : {i}/{len(files)} fichiers, {len(found)}/{len(_TARGETS)} trouvés', flush=True)
    assert set(found) == _TARGETS, f'{len(_TARGETS - set(found))} textes sans provenance'
    by_source = Counter()
    dest = mix / 'provenance.jsonl.gz'
    with dest.with_suffix('.tmp').open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
        for index, digest in enumerate(digests):
            row = {'row_zero_based': index, **found[digest]}
            by_source[row['source']] += 1
            gz.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
    dest.with_suffix('.tmp').replace(dest)
    expected = json.loads((mix / 'manifest.json').read_text())
    assert len(digests) == expected['documents']
    assert dict(by_source) == {s: r['documents'] for s, r in expected['by_source'].items()}
    report = {'documents': len(digests), 'unique_texts': len(found), 'by_source': dict(by_source),
              'all_rows_matched_by_exact_text_sha256': True,
              'missing_urls': sum(not found[d].get('url') for d in digests),
              'missing_declared_licenses': sum(not found[d].get('license') for d in digests),
              'provenance_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
              'train_gz_sha256': hashlib.sha256((mix / 'train.jsonl.gz').read_bytes()).hexdigest()}
    assert report['train_gz_sha256'] == expected['train_sha256']
    (mix / 'provenance_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
