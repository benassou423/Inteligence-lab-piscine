"""Comparer brut/nettoyé/complet à contrat identique, sans changer le corpus actif."""
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from tools.compare_targeted_results import load_run


AUDIT = Path('data/stages/complet_20261110_audit')
RUNS = {
    1337: {'brut': 'ablation_brut_s1337', 'nettoye': 'proxy_wiki20261001_fr065', 'complet': 'ablation_complet_s1337'},
    2027: {'brut': 'ablation_brut_s2027', 'nettoye': 'proxy_cible_base_s2027', 'complet': 'ablation_complet_s2027'},
}


def main():
    state = json.loads((AUDIT / 'full_status.json').read_text())
    assert state.get('complete') and state.get('contamination_flagged') == 0
    assert json.loads((AUDIT / 'pair_review.json').read_text())['accepted_for_benchmark']
    for seed in RUNS:
        assert json.loads((AUDIT / f'ablation_brut_s{seed}' / 'contamination.json').read_text())['n_signales'] == 0
    runs, contract, results = {}, None, []
    for seed, recipes in RUNS.items():
        for recipe, name in recipes.items():
            row = load_run(name)
            assert row['config']['seed'] == seed
            this_contract = {k: row['meta'][k] for k in ['train_config', 'n_params', 'tokens_seen',
                'tokenizer_sha256', 'device', 'precision', 'torch', 'tokenizers', 'python']}
            if contract is None:
                contract = this_contract
            assert this_contract == contract, f'Contrat différent : {name}'
            runs[name] = row
            results.append({'seed': seed, 'recipe': recipe, 'run': name, 'bpb': row['evaluation']['bpb'],
                            'bpb_par_genre': row['evaluation']['bpb_par_genre'],
                            'dataset_sha256': row['meta']['dataset_sha256'],
                            'mix_documents': row['mix']['documents'], 'mix_bytes': row['mix']['bytes']})
    genres = sorted(results[0]['bpb_par_genre'])
    average = {recipe: {'bpb': mean(r['bpb'] for r in results if r['recipe'] == recipe),
                       'bpb_par_genre': {g: mean(r['bpb_par_genre'][g] for r in results if r['recipe'] == recipe)
                                         for g in genres}} for recipe in ['brut', 'nettoye', 'complet']}
    pairs = []
    for seed in RUNS:
        chosen = {r['recipe']: r for r in results if r['seed'] == seed}
        pairs.append({'seed': seed,
                      'nettoye_minus_brut': chosen['nettoye']['bpb'] - chosen['brut']['bpb'],
                      'complet_minus_nettoye': chosen['complet']['bpb'] - chosen['nettoye']['bpb']})
    deltas = {g: average['complet']['bpb_par_genre'][g] - average['nettoye']['bpb_par_genre'][g] for g in genres}
    passes = all(p['complet_minus_nettoye'] < 0 for p in pairs) and all(v <= 0 for v in deltas.values())
    result = {'created': datetime.now().astimezone().isoformat(), 'results': results, 'average': average,
              'pairs': pairs, 'complete_genre_deltas': deltas, 'contract': contract, 'runs': runs,
              'proxy_adoption_gate_passed': passes, 'full_budget_confirmation_required': True,
              'active_corpus_changed': False, 'statistical_significance_claimed': False,
              'scope': 'Deux tirages du mélange ; initialisation du modèle commune et tokenizer figé.'}
    for path in [AUDIT / 'comparison.json', Path('experiments/ablation_complete_20261110.json')]:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Ablation brut, nettoyé et complet', '',
             '| Recette | Tirage 1337 | Tirage 2027 | Moyenne BPB |', '|---|---:|---:|---:|']
    for recipe in ['brut', 'nettoye', 'complet']:
        values = [next(r['bpb'] for r in results if r['recipe'] == recipe and r['seed'] == seed) for seed in RUNS]
        lines.append(f'| {recipe} | {values[0]:.6f} | {values[1]:.6f} | {average[recipe]["bpb"]:.6f} |')
    lines += ['', 'Plus bas est meilleur. Même contrat vérifié, 25 001 984 tokens vus par run.',
              'Le brut est protégé contre la validation avant entraînement. Les corpus bruts ne sont pas publiés.',
              'Les différences incluent les changements naturels de proportions des sources après filtrage.',
              'Deux tirages ne démontrent pas une significativité statistique. Confirmation à 100M distincte.',
              f'Critère conservateur de confirmation du complet satisfait : {passes}. Corpus actif inchangé.', '']
    Path('experiments/ablation_complete_20261110.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in result.items() if k not in {'runs', 'contract'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
