"""Vérifier intégralement les 50 paires échantillonnées de quasi-doublons.

Recalcule Jaccard sur les chaînes de cinq mots elles-mêmes (sans hash), à
partir des originaux et du masquage appliqué. Les extraits et titres servent
à la relecture ; le calcul intégral ne constitue pas une lecture humaine.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from bench.mix import _ID_RE, iter_lines
from pipeline.clean.privacy import PrivacyStage
from pipeline.io import write_docs
from pipeline.stats import write_json


AUDIT = Path('data/stages/complet_20261110_audit')
BASELINE = Path('data/stages/nettoye/08_exact_dedup')


def actual_sets(text, size):
    words = re.findall(r'\w+', text.lower())
    return {tuple(words[i:i + size]) for i in range(max(0, len(words) - size + 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    state = json.loads((AUDIT / 'full_status.json').read_text())
    while args.wait and not state.get('complete'):
        if state['state'] in {'failed', 'stopped_for_revision'}:
            raise RuntimeError(f'Pipeline interrompu : {state["state"]}')
        time.sleep(5)
        state = json.loads((AUDIT / 'full_status.json').read_text())
    if not state.get('complete') or state.get('contamination_flagged') != 0:
        raise RuntimeError('Attendre le pipeline complet et sa contamination à zéro')
    stats = json.loads(Path('data/stages/complet_20261110/04_minhash_dedup/stats.json').read_text())
    pairs = stats['report']['audit_pairs']
    wanted = {p['removed'] for p in pairs} | {p['kept']['id'] for p in pairs}
    source_ids = defaultdict(set)
    for ident in wanted:
        source_ids[ident.split(':', 1)[0]].add(ident)
    documents = {}
    privacy = PrivacyStage({}, None)
    for source, ids in source_ids.items():
        for path in sorted((BASELINE / source).glob('part-*.jsonl.zst')):
            for _, line in iter_lines(path):
                match = _ID_RE.match(line)
                if match and json.loads('"' + match[1] + '"') not in ids:
                    continue
                doc = json.loads(line)
                if doc['id'] in ids:
                    documents[doc['id']] = privacy.apply(doc).doc
        print(f'Originaux retrouvés : {source}, {sum(i in documents for i in ids)}/{len(ids)}', flush=True)
    if set(documents) != wanted:
        raise RuntimeError(f'{len(wanted - set(documents))} originaux manquants')
    present = set()
    final = Path('data/stages/complet_20261110/06_exact_dedup')
    for source, ids in source_ids.items():
        for path in sorted((final / source).glob('part-*.jsonl.zst')):
            for _, line in iter_lines(path):
                match = _ID_RE.match(line)
                if match:
                    ident = json.loads('"' + match[1] + '"')
                    if ident in ids:
                        present.add(ident)
                else:
                    ident = json.loads(line)['id']
                    if ident in ids:
                        present.add(ident)
    records, full_text = [], []
    size = stats['report']['shingle_size']
    threshold = stats['report']['threshold']
    for i, pair in enumerate(pairs, 1):
        a, b = documents[pair['kept']['id']], documents[pair['removed']]
        left, right = actual_sets(a['text'], size), actual_sets(b['text'], size)
        similarity = len(left & right) / len(left | right)
        if similarity + 1e-12 < threshold or abs(similarity - pair['similarity']) > 1e-9:
            raise RuntimeError(f'Jaccard indépendant divergent : {pair["removed"]}')
        # Chaque retrait doit être absent du résultat final. Le témoin peut être
        # modifié/retiré par la décontamination suivante : en conserver la trace.
        if b['id'] in present:
            raise RuntimeError('Un document retiré par MinHash est encore dans la sortie')
        record = {'index': i, 'kept_id': a['id'], 'removed_id': b['id'],
                  'kept_title': a.get('meta', {}).get('title', ''),
                  'removed_title': b.get('meta', {}).get('title', ''),
                  'kept_url': a.get('url'), 'removed_url': b.get('url'),
                  'jaccard_independent': similarity, 'kept_present_in_final': a['id'] in present,
                  'chars_kept': len(a['text']), 'chars_removed': len(b['text']),
                  'kept_start': a['text'][:2200], 'removed_start': b['text'][:2200]}
        records.append(record)
        full_text.append({'index': i, 'kept_id': a['id'], 'removed_id': b['id'],
                          'kept_text': a['text'], 'removed_text': b['text']})
        print(f'Paire {i}/{len(pairs)} : Jaccard sans hash {similarity:.6f}', flush=True)
    write_docs(AUDIT / 'near_pairs_full_text.jsonl.zst', full_text)
    result = {'pairs': len(records), 'independent_jaccard_passed': True,
              'min_similarity': min((r['jaccard_independent'] for r in records), default=None),
              'removed_ids_absent_from_final': True,
              'witnesses_absent_after_later_stages': sum(not r['kept_present_in_final'] for r in records),
              'qualitative_review_pending': True, 'records': records}
    write_json(AUDIT / 'near_pairs_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
