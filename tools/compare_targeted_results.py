"""Compare les quatre essais locaux appariés ; refuse des réglages non comparables."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean

from bench.run import _ckpt_meta

BASE = Path('data/stages/targeted_clean_20261104_audit')
PAIRS = [(1337, 'proxy_wiki20261001_fr065', 'proxy_cible_v1_s1337'),
         (2027, 'proxy_cible_base_s2027', 'proxy_cible_v1_s2027')]


def read(path):
    return json.loads(Path(path).read_text())


def load_run(name):
    run, mix = Path('runs') / name, Path('data/mixes') / name
    meta, evaluation = _ckpt_meta(run / 'ckpt.pt'), read(run / 'eval.json')
    data, config, manifest = read(run / 'data_meta.json'), read(mix / 'config.json'), read(mix / 'manifest.json')
    assert meta['dataset_sha256'] == data['dataset_sha256']
    assert meta['dataset_n_tokens'] == data['n_tokens'] >= meta['tokens_seen']
    assert not meta['official'] and meta['tokens_seen'] == 25001984
    assert meta['tokenizer_sha256'] == data['tokenizer_sha256'] == evaluation['tokenizer_sha256']
    assert hashlib.sha256((run / 'tokenizer.json').read_bytes()).hexdigest() == meta['tokenizer_sha256']
    assert Path(evaluation['data']).resolve() == Path('data/validation.jsonl').resolve()
    assert abs(evaluation['bpb'] - meta['val_bpb']) < 1e-9
    assert config['mix']['max_bytes'] == 100000000 and config['mix']['order'] == 'shuffle'
    return {'name': name, 'meta': meta, 'evaluation': evaluation, 'mix': manifest, 'config': config}


def main():
    state = read(BASE / 'status.json')
    assert state.get('complete') and state.get('data_verified')
    assert read(BASE / 'contamination.json')['n_signales'] == 0
    runs, pairs, expected = {}, [], None
    for seed, base_name, candidate_name in PAIRS:
        for name in [base_name, candidate_name]:
            r = load_run(name)
            assert r['config']['seed'] == seed
            contract = {k: r['meta'][k] for k in ['train_config', 'n_params', 'tokens_seen', 'tokenizer_sha256',
                                                  'device', 'precision', 'torch', 'tokenizers', 'python']}
            if expected is None:
                expected = contract
            assert contract == expected, f'Paramètres différents : {name}'
            runs[name] = r
        a, b = runs[base_name]['evaluation'], runs[candidate_name]['evaluation']
        pairs.append({'seed': seed, 'baseline_run': base_name, 'candidate_run': candidate_name,
                      'baseline_bpb': a['bpb'], 'candidate_bpb': b['bpb'], 'delta_bpb': b['bpb'] - a['bpb'],
                      'genre_deltas': {g: b['bpb_par_genre'][g] - v for g, v in a['bpb_par_genre'].items()}})
    genres = sorted(pairs[0]['genre_deltas'])
    average = {'baseline_bpb': mean(p['baseline_bpb'] for p in pairs),
               'candidate_bpb': mean(p['candidate_bpb'] for p in pairs),
               'delta_bpb': mean(p['delta_bpb'] for p in pairs),
               'genre_deltas': {g: mean(p['genre_deltas'][g] for p in pairs) for g in genres}}
    criteria = read(BASE / 'comparison_spec.json')['adoption_gate']
    passed = (all(p['delta_bpb'] < 0 for p in pairs) and
              average['delta_bpb'] <= -criteria['minimum_mean_bpb_improvement'] and
              max(average['genre_deltas'].values()) <= criteria['maximum_mean_genre_regression'])
    result = {'created': datetime.now().astimezone().isoformat(), 'pairs': pairs, 'average': average,
              'contract': expected, 'criteria': criteria, 'proxy_adoption_gate_passed': passed,
              'recommendation': 'candidate_for_confirmation' if passed else 'keep_current_reference',
              'scope': 'Two input sampling seeds; model initialization and tokenizer fixed. No statistical significance claim.',
              'runs': runs, 'active_corpus_changed': False}
    (BASE / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    Path('experiments/targeted_clean_v1_20261104.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'runs', 'contract'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
