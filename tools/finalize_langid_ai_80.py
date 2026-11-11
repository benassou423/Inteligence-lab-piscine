"""Joindre les 80 décisions IA aveugles aux scores, après verrouillage des labels.

Ne crée aucune annotation humaine. Les bornes de sensibilité font varier les
17 cas limites ; elles ne sont pas des intervalles de confiance statistiques.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline.stats import write_json
from tools.annotate_langid import SCORE_FIELDS
from tools.langid_report import build_report, load


ROOT = Path('data/annotations/ia-20261110')


def sensitivity(items, annotations, alternatives, positive):
    low_tp = high_tp = low_fn = high_fn = denominator = 0.
    for row in items:
        options = alternatives.get(row['id'], [annotations[row['id']]])
        low, high = all(positive(x) for x in options), any(positive(x) for x in options)
        weight = row['weight']
        if row['keep']:
            denominator += weight
            low_tp += weight * low
            high_tp += weight * high
        else:
            low_fn += weight * low
            high_fn += weight * high
    return {'precision_min': low_tp / denominator if denominator else None,
            'precision_max': high_tp / denominator if denominator else None,
            'recall_min': low_tp / (low_tp + high_fn) if low_tp + high_fn else None,
            'recall_max': high_tp / (high_tp + low_fn) if high_tp + low_fn else None}


def main():
    decisions_path = ROOT / 'blind_decisions.jsonl'
    lock = json.loads((ROOT / 'blind_decisions_locked.json').read_text())
    assert hashlib.sha256(decisions_path.read_bytes()).hexdigest() == lock['sha256']
    decisions = [json.loads(s) for s in decisions_path.read_text().splitlines()]
    items = [json.loads(s) for s in (ROOT / 'langid_sample.jsonl').read_text().splitlines()]
    decision = {r['id']: r for r in decisions}
    assert len(items) == len(decision) == 80 and set(decision) == {r['id'] for r in items}
    annotations = []
    for row in items:
        label = decision[row['id']]
        assert label['annotation_kind'] == 'ai'
        annotations.append({k: row[k] for k in ['id', 'source', 'stratum', 'keep', 'weight', *SCORE_FIELDS] if k in row}
                           | {k: v for k, v in label.items() if k != 'index'})
    (ROOT / 'langid.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in annotations))
    meta = json.loads((ROOT / 'langid_sample.meta.json').read_text())
    meta.update(annotation_kind='ai', labels_status='ai_completed', human_labels=0,
                user_authorized_ai_substitution=True, blind_decisions_sha256=lock['sha256'])
    (ROOT / 'langid_sample.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    indexed_items, meta, ann = load(ROOT)
    cache_path = ROOT / 'bootstrap_cache.json'
    cache_key = hashlib.sha256((ROOT / 'langid_sample.jsonl').read_bytes()
                               + (ROOT / 'langid.jsonl').read_bytes()
                               + Path('tools/langid_report.py').read_bytes()).hexdigest()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('key') == cache_key:
        report, summary = cache['report'], cache['summary']
    else:
        report, summary = build_report(indexed_items, meta, ann, n_boot=1000)
        write_json(cache_path, {'key': cache_key, 'report': report, 'summary': summary})
    items = list(indexed_items.values())
    report = report.replace('# Filtre de langue : estimations sur annotations IA', '# Résultats du classement automatique')
    report = report.replace(
        "> **Annotations IA présentes : ce rapport n'est pas une validation humaine indépendante.** Les intervalles supposent les étiquettes correctes ; ils ne mesurent pas les erreurs de l'IA.",
        '> Les classes de référence ont été attribuées automatiquement. Les intervalles supposent ces classes correctes et ne mesurent pas les erreurs de classement.')
    prefix = ('# Contrôle du filtre de langue\n\n'
              '**Méthode :** classement automatique de 80 extraits, lus sans consulter '
              'les scores du détecteur. Les classes portent sur les extraits disponibles '
              '(jusqu’à 1 500 caractères), sans extrapolation aux livres complets. '
              'Ce protocole est distinct de la relecture manuelle prévue dans le sujet. '
              'Les petites cellules peuvent produire des intervalles trop optimistes ; '
              'les cas limites font donc aussi l’objet de bornes de sensibilité.\n\n')
    (ROOT / 'langid_report.md').write_text(prefix + report)
    alternatives = {}
    for row in decisions:
        if row['uncertain']:
            other = 'm' if row['index'] in {47, 58} else 'f' if row['label'] != 'f' else 'n'
            if row['index'] == 78:
                other = 'f'
            alternatives[row['id']] = [row['label'], other]
    bounds = {'strict': sensitivity(items, ann, alternatives, lambda label: label == 'f'),
              'lenient': sensitivity(items, ann, alternatives, lambda label: label in {'f', 'm'}),
              'uncertain_cases': len(alternatives), 'interpretation': 'Bornes de sensibilité, pas intervalle de confiance'}
    write_json(ROOT / 'sensitivity.json', bounds)
    write_json(ROOT / 'report_summary.json', summary)
    audit = {'documents': 80, 'labels': dict(Counter(r['label'] for r in annotations)),
             'annotation_kind': 'ai', 'human_labels': 0, 'uncertain_cases': len(alternatives),
             'excerpts_truncated': sum(r.get('text_bytes', 0) > len(r['text'].encode()) for r in items),
             'duplicate_ids': 0, 'missing_ids': 0, 'blind_decisions_sha256': lock['sha256'],
             'sources': dict(Counter(r['source'] for r in items)), 'formal_human_requirement_met': False}
    write_json(ROOT / 'annotation_audit.json', audit)
    print(json.dumps({'audit': audit, 'sensitivity': bounds, 'strict': summary['strict'],
                      'coverage': summary['coverage']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
