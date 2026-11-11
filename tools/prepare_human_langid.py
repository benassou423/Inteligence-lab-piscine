"""Préparer 80 extraits aveugles couvrant les sept sources, sans créer d'étiquette.

L'annotation humaine se fait ensuite avec tools.annotate_langid --out-dir …
Une taille minimale par source évite qu'un échantillon de 80 soit presque
entièrement composé de Wikipédia. Les poids sont recalculés par cellule.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.annotate_langid import _hash, _largest_remainder, draw_sample, find_langid_dir


QUOTAS = {'wikipedia_fr': 12, 'wikisource_fr': 12, 'gutenberg_fr': 12, 'fineweb2_fra': 12,
          'commoncrawl': 12, 'assemblee_nationale': 10, 'wikinews_fr': 10}


def main():
    out = Path('data/annotations/humain-20261110')
    if (out / 'langid.jsonl').exists():
        raise RuntimeError('Annotations présentes : ne pas changer leur échantillon')
    pool, meta = draw_sample(find_langid_dir('nettoye', 'data/stages'), 100_000)
    cells = defaultdict(list)
    for row in pool:
        cells[(row['source'], row['stratum'])].append(row)
    items = []
    for source, quota in QUOTAS.items():
        available = {h: len(rows) for (s, h), rows in cells.items() if s == source}
        allocation = _largest_remainder(quota, {h: 1 for h in available}, available)
        for stratum, count in allocation.items():
            if not count:
                continue
            rows = sorted(cells[(source, stratum)], key=lambda r: _hash(r['id'], 'human-20261110'))[:count]
            weight = meta['population'][f'{source}|{stratum}'] / count
            items.extend({**r, 'weight': weight} for r in rows)
    assert len(items) == len({r['id'] for r in items}) == sum(QUOTAS.values()) == 80
    items.sort(key=lambda r: _hash(r['id'], 'human-order-20261110'))
    meta.update(n=len(items), config='nettoye', annotation_kind='human', labels_status='pending_human_input',
                prepared_by='Codex (IA), aucune étiquette humaine créée', source_quotas=QUOTAS,
                scope='600 premiers caractères affichés ; pas validation intégrale des livres',
                sampling='Minimum par source, répartition égale des strates disponibles, poids population/cellule')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'langid_sample.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    (out / 'langid_sample.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({'documents': len(items), 'sources': dict(Counter(r['source'] for r in items)),
                      'labels_created': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
