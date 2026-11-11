"""Contrôle IA de masquages réels, ciblant particulièrement les sources soignées.

Les extraits avant masquage restent locaux ; ne pas les joindre au dataset public.
L'échantillon est déterministe, au plus 50 documents masqués par source.
"""
import json
from multiprocessing import get_context
from pathlib import Path

from bench.mix import iter_lines
from pipeline.clean.privacy import EMAIL, PHONE, PHONE_ID_PREFIX, mask_text
from pipeline.stats import BottomK, sample_hash, write_json


ROOT = Path('data/stages/complet_20261110_audit')


def select_file(path):
    sample = BottomK(50)
    for _, line in iter_lines(path):
        if '"privacy":' not in line:
            continue
        doc = json.loads(line)
        if doc.get('meta', {}).get('privacy'):
            rank = sample_hash('privacy-full-audit-20261110', doc['id'])
            if sample.wants(rank):
                sample.add(rank, doc['id'], {'id': doc['id'], 'file': str(path),
                                            'text_after': doc['text'], 'source': doc['source']})
    return path.parent.name, sample.items()


def inspect_file(task):
    path, selected = task
    selected = {r['id']: r for r in selected}
    out = []
    for _, line in iter_lines(path):
        # Les sorties de la référence ont leurs IDs en première propriété.
        if not any(ident in line[:220] for ident in selected):
            continue
        doc = json.loads(line)
        if doc['id'] not in selected:
            continue
        before = doc['text']
        after, counters = mask_text(before, doc['source'])
        if after != selected[doc['id']]['text_after']:
            raise ValueError(f'Masquage divergent : {doc["id"]}')
        spans = []
        for kind, pattern in [('EMAIL', EMAIL), ('TELEPHONE', PHONE)]:
            for match in pattern.finditer(before):
                if kind == 'TELEPHONE' and PHONE_ID_PREFIX.search(before[max(0, match.start() - 60):match.start()]):
                    continue
                spans.append({'kind': kind, 'context': before[max(0, match.start() - 70):match.end() + 70]})
        out.append({'id': doc['id'], 'source': doc['source'], 'masks': dict(counters), 'spans': spans,
                    'text_matches': True})
    if len(out) != len(selected):
        raise ValueError(f'Originaux manquants : {path}')
    return out


def main():
    stage = Path('data/stages/complet_20261110/03_quality')
    manifest = Path('data/stages/complet_20261110/manifest.json')
    if manifest.exists():
        stage = Path(json.loads(manifest.read_text())['final_dir'])
    samples = {}
    with get_context('spawn').Pool(4) as pool:
        for source, items in pool.imap_unordered(select_file, sorted(stage.glob('*/part-*.jsonl.zst'))):
            bottom = samples.setdefault(source, BottomK(50))
            for rank, record in items:
                if bottom.wants(rank):
                    bottom.add(rank, record['id'], record)
    tasks = {}
    for bottom in samples.values():
        for _, record in bottom.items():
            path = Path(record['file'])
            original = Path('data/stages/nettoye/08_exact_dedup') / path.parent.name / path.name
            tasks.setdefault(original, []).append(record)
    rows = []
    with get_context('spawn').Pool(4) as pool:
        for group in pool.imap_unordered(inspect_file, list(tasks.items())):
            rows.extend(group)
    result = {'provenance': 'Diagnostic pour relecture IA, pas annotation humaine',
              'stage': str(stage), 'documents': len(rows), 'text_divergences': 0,
              'records': sorted(rows, key=lambda r: (r['source'], r['id']))}
    write_json(ROOT / 'privacy_full_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
