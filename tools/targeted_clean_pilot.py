"""Essai déterministe du nettoyage sur 2 000 documents et les cas d'audit séparés."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

from pipeline.clean.targeted import TargetedCleanStage
from pipeline.io import read_docs, write_docs
from pipeline.stats import BottomK, sample_hash, write_json

QUOTAS = {'wikipedia_fr': 800, 'wikisource_fr': 300, 'gutenberg_fr': 200,
          'wikinews_fr': 200, 'commoncrawl': 200, 'fineweb2_fra': 200, 'assemblee_nationale': 100}


def sample_file(path):
    source = path.parent.name
    sample = BottomK(QUOTAS[source])
    n = 0
    for doc in read_docs(path):
        n += 1
        h = sample_hash('targeted-fresh-v1', doc['id'])
        if sample.wants(h):
            sample.add(h, doc['id'], doc)
    return source, n, sample.items()


def run(output, workers=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path('data/stages/nettoye/08_exact_dedup').glob('*/part-*.jsonl.zst'))
    signatures = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    cached = output / 'pilot_input.jsonl.zst'
    receipt = output / 'pilot_input.json'
    if not (receipt.exists() and json.loads(receipt.read_text()).get('input_signatures') == signatures and cached.exists()):
        samples = {s: BottomK(k) for s, k in QUOTAS.items()}
        scanned = Counter()
        with get_context('spawn').Pool(workers) as pool:
            for i, (source, n, items) in enumerate(pool.imap_unordered(sample_file, files), 1):
                scanned[source] += n
                for h, doc in items:
                    if samples[source].wants(h):
                        samples[source].add(h, doc['id'], doc)
                if i % 25 == 0:
                    print(f'Échantillon : {i}/{len(files)} fichiers', flush=True)
        docs = [doc for s in sorted(samples) for _, doc in samples[s].items()]
        assert len(docs) == sum(QUOTAS.values()) == 2000
        stats = json.loads(Path('data/stages/nettoye/08_exact_dedup/stats.json').read_text())
        assert sum(scanned.values()) == stats['total']['docs_out']
        assert signatures == {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
        write_docs(cached, docs)
        write_json(receipt, {'input_signatures': signatures, 'scanned': dict(scanned), 'quotas': QUOTAS})
    fresh = list(read_docs(cached))
    stress = list(read_docs('data/stages/langid_full_audit_20261104/selected_full.jsonl.zst'))
    reviews = [json.loads(x) for x in Path('data/annotations/ia-20261104/full_document_audit/reviews.jsonl').read_text().splitlines()]
    good = {r['id'] for r in reviews if r['decision'] in {'candidate_ok', 'control_ok'}}
    stage = TargetedCleanStage({}, SimpleNamespace())
    totals = defaultdict(Counter)
    changed, good_failures = [], []
    def outputs():
        for scope, docs in [('fresh', fresh), ('stress', stress)]:
            for doc in docs:
                before = doc['text']
                outcome = stage.apply(doc)
                assert doc['text'] == before
                after = outcome.doc['text'] if outcome.keep else ''
                c = totals[scope + '/' + doc['source']]
                c['docs'] += 1
                c['kept'] += outcome.keep
                c['changed'] += after != before
                c['bytes_before'] += len(before.encode())
                c['bytes_after'] += len(after.encode())
                c.update(outcome.counters or {})
                if scope == 'stress' and doc['id'] in good and (not outcome.keep or before != after):
                    good_failures.append(doc['id'])
                row = {'id': doc['id'], 'source': doc['source'], 'scope': scope, 'keep': outcome.keep,
                       'reason': outcome.reason, 'before': before, 'after': after,
                       'rules': outcome.counters, 'title': doc.get('meta', {}).get('title', ''),
                       'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                       'after_sha256': hashlib.sha256(after.encode()).hexdigest()}
                if after != before:
                    changed.append({k: v for k, v in row.items() if k not in {'before', 'after'}} |
                                   {'bytes_removed': len(before.encode()) - len(after.encode())})
                yield row
    write_docs(output / 'pilot_results.jsonl.zst', outputs())
    fresh_total = sum((c for s, c in totals.items() if s.startswith('fresh/')), Counter())
    gates = {'positive_controls_unchanged': not good_failures,
             'fresh_kept_at_least_99pct': fresh_total['kept'] >= .99 * fresh_total['docs'],
             'fresh_bytes_kept_at_least_95pct': fresh_total['bytes_after'] >= .95 * fresh_total['bytes_before']}
    result = {'created': datetime.now().astimezone().isoformat(), 'by_source': {s: dict(c) for s, c in totals.items()},
              'fresh': dict(fresh_total), 'gates': gates, 'positive_controls': len(good),
              'positive_failures': good_failures, 'manual_review_required': True,
              'changed': sorted(changed, key=lambda r: (r['scope'], -r['bytes_removed'], r['id'])),
              'code_sha256': hashlib.sha256(Path('pipeline/clean/targeted.py').read_bytes()).hexdigest()}
    write_json(output / 'pilot.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'changed'}, ensure_ascii=False, indent=2), flush=True)
    return all(gates.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/stages/targeted_clean_20261104_audit')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not Path('data/stages').is_symlink():
        raise SystemExit('data/stages doit rester un lien nosync')
    raise SystemExit(0 if run(args.output, args.workers) else 1)


if __name__ == '__main__':
    main()
