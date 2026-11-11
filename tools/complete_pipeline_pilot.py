"""Essai réel de la variante complète, avant tout traitement intégral.

Réutilise le tirage de 2 000 documents dont la couverture du corpus est prouvée.
Contrôles injectés séparément : rappel LSH et vérification Jaccard, pas une
estimation d'erreur sur l'ensemble du corpus et pas une annotation humaine.
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from pipeline.clean.privacy import PrivacyStage
from pipeline.clean.quality import QualityStage
from pipeline.dedup.minhash import MinHashDedupStage, WORDS, band_hashes, jaccard, shingles
from pipeline.io import read_docs, write_docs
from pipeline.run import Runner
from pipeline.stats import write_json


OUT = Path('data/stages/complet_20261110_audit')
SAMPLE = Path('data/stages/targeted_clean_20261104_audit/pilot_input.jsonl.zst')


def controlled_pairs(docs):
    stage = MinHashDedupStage({}, None)
    results = []
    for doc in docs:
        words = WORDS.findall(doc.get('text', '').lower())[:300]
        if len(words) < 200 or len(set(words)) < 100:
            continue
        original = ' '.join(words)
        a = shingles(original)
        sa = np.frombuffer(stage.key({'text': original}), np.uint32).reshape(1, -1)
        for ratio in [.01, .03, .05, .08, .11, .15, .25]:
            n = int(ratio * len(words))
            altered = ' '.join(words[:-n] + [f'controlezz{i}' for i in range(n)])
            b = shingles(altered)
            sb = np.frombuffer(stage.key({'text': altered}), np.uint32).reshape(1, -1)
            truth = jaccard(a, b)
            candidate = any(band_hashes(sa, band, stage.rows)[0] == band_hashes(sb, band, stage.rows)[0]
                            for band in range(stage.bands))
            results.append({'source': doc['source'], 'similarity': truth, 'lsh_candidate': bool(candidate),
                            'would_remove': bool(candidate and truth >= stage.threshold)})
        if len(results) >= 350:
            break
    qualifying = [r for r in results if r['similarity'] >= stage.threshold]
    recall = sum(r['lsh_candidate'] for r in qualifying) / max(1, len(qualifying))
    return {'pairs': len(results), 'at_or_above_threshold': len(qualifying), 'observed_recall': recall,
            'false_removals_below_threshold': sum(r['would_remove'] and r['similarity'] < .8 for r in results),
            'results': results, 'provenance': 'Perturbations contrôlées, pas des annotations humaines'}


def main():
    if not Path('data/stages').is_symlink():
        raise RuntimeError('Le lien nosync doit être conservé')
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    docs = list(read_docs(SAMPLE))
    assert len(docs) == len({d['id'] for d in docs}) == 2000
    by_source = defaultdict(list)
    stats = defaultdict(Counter)
    reviews = []
    privacy, quality = PrivacyStage({}, None), QualityStage({}, None)
    for doc in docs:
        by_source[doc['source']].append(doc)
        p = privacy.apply(doc)
        q = quality.apply(p.doc)
        st = stats[doc['source']]
        st['documents'] += 1
        st['masked_documents'] += p.doc['text'] != doc['text']
        st['quality_removed'] += not q.keep
        st['bytes_before'] += len(doc['text'].encode())
        st['bytes_after'] += len(q.doc['text'].encode()) if q.keep else 0
        st.update({f'masks_{k}': v for k, v in (p.counters or {}).items() if k != 'documents_masked'})
        if p.doc['text'] != doc['text'] or not q.keep:
            reviews.append({'id': doc['id'], 'source': doc['source'], 'before': doc['text'],
                            'after': p.doc['text'], 'quality_keep': q.keep, 'quality_reason': q.reason,
                            'quality_signals': q.doc.get('meta', {}).get('quality'), 'masks': p.counters})
    write_docs(OUT / 'reviews.jsonl.zst', reviews)
    for source, group in by_source.items():
        write_docs(OUT / 'input' / source / 'part-00000.jsonl.zst', group)
    cfg = yaml.safe_load(Path('configs/pipeline/complet_20261110.yaml').read_text())
    cfg.update(name='pilot', input=str(OUT / 'input'), output=str(OUT), prune_intermediate_files=False)
    config_path = OUT / 'pilot.yaml'
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    controls = controlled_pairs(docs)
    write_json(OUT / 'controlled_pairs.json', controls)
    result = Runner(config_path, workers=4, rejected_k=200).run()
    assert result == 0
    stages = [json.loads((OUT / 'pilot' / f'{i + 1:02d}_{s["name"]}' / 'stats.json').read_text())
              for i, s in enumerate(cfg['stages'])]
    # Gardes mécaniques, à compléter par la relecture des changements réels.
    protected = [s for s in stats if s not in quality.sources]
    gates = {'documents_2000': len(docs) == 2000,
             'quality_protected_sources': all(stats[s]['quality_removed'] == 0 for s in protected),
             'controlled_lsh_recall_at_least_90pct': controls['observed_recall'] >= .9,
             'no_false_removal_in_controls': controls['false_removals_below_threshold'] == 0,
             'real_near_minimum_80pct': (stages[3]['report']['minimum_removed_similarity'] or 1) >= .8}
    report = {'documents': len(docs), 'by_source': {s: dict(v) for s, v in stats.items()},
              'gates': gates, 'requires_review': True, 'seconds': time.monotonic() - t0,
              'near_dedup': stages[3]['report'], 'controlled_pairs': {k: v for k, v in controls.items() if k != 'results'}}
    write_json(OUT / 'pilot_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
