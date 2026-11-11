"""Risques réels : nombres français, répétition littéraire, faux voisins LSH et reprise."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pipeline.clean.privacy import PrivacyStage, mask_text
from pipeline.clean.quality import QualityStage
from pipeline.dedup.minhash import MinHashDedupStage, jaccard, shingles
from pipeline.io import read_docs, write_docs
from pipeline.run import Runner


def doc(ident, text, source='wikipedia_fr'):
    return {'id': ident, 'source': source, 'text': text, 'meta': {'langid_fr': .99},
            'url': 'https://example.org', 'date': '2026-01-01', 'license': 'fixture', 'domain': 'web'}


@pytest.mark.parametrize('value,marker', [
    ('contact+test@example.org', '[EMAIL]'), ('06 12 34 56 78', '[TELEPHONE]'),
    ('+33 6 12 34 56 78', '[TELEPHONE]'), ('01.23.45.67.89', '[TELEPHONE]'),
    ('FR14 2004 1010 0505 0001 3M02 606', '[IBAN]'),
    ('adresse IP : 192.0.2.1', 'adresse IP : [IP]'), ('IPv6 : 2001:db8::1', 'IPv6 : [IP]'),
])
def test_coordonnees_masquees(value, marker):
    result, counters = mask_text(value)
    assert result == marker
    assert sum(counters.values()) == 1
    assert mask_text(result)[0] == result


@pytest.mark.parametrize('text', [
    'Né le 12 octobre 1842, il meurt le 03/04/1912.',
    'La population était de 1 234 567 habitants en 2020.',
    'La version 1.2.3.4 coûte 12,34 euros. IP : 999.0.0.1',
    'Référence ABC0612345678DEF et FR15 2004 1010 0505 0001 3M02 606.',
    'Les caractéristiques de l’antenne n° 0422700190 sont publiques.',
    'Référence : 0123456789. Téléphone : 0123456789.',
])
def test_nombres_et_faux_iban_preserves(text):
    if 'Téléphone :' in text:
        assert mask_text(text) == ('Référence : 0123456789. Téléphone : [TELEPHONE].', {'TELEPHONE': 1})
    else:
        assert mask_text(text) == (text, {})


def test_score_langue_ne_decrit_pas_texte_modifie():
    original = doc('x', 'Écrire à contact@example.org.')
    out = PrivacyStage({}, None).apply(original)
    assert out.doc['text'] == 'Écrire à [EMAIL].'
    assert out.doc['meta']['privacy']['previous_langid'] == {'langid_fr': .99}
    assert 'langid_fr' not in out.doc['meta']
    assert original['meta']['langid_fr'] == .99


@pytest.mark.parametrize('text', [
    'Moyennes annuelles 44.04 45.62 34.76 34.82',
    'Carbone 0.0121312256 et Total 0.0331309728.',
    '04 30 22 18 16 12 10 08 08',
    '01 02 03 04 05 06 07 08 09',
    'LONGUEURS EN CICÉROS\n04 30 22 18 16\n',
    'Fraction périodique : 0126582278 ; autre terme : 0169491525.',
])
def test_tableaux_et_decimales_preserves(text):
    assert mask_text(text, 'wikisource_fr') == (text, {})
    if 'CICÉROS' not in text and 'Fraction' not in text:
        assert mask_text(text, 'commoncrawl') == (text, {})


def test_telephone_contextuel_dans_source_soignee():
    text = 'Le conseiller est joignable au 06 10 03 97 65. Merci.'
    assert mask_text(text, 'assemblee_nationale')[0] == 'Le conseiller est joignable au [TELEPHONE]. Merci.'
    assert mask_text('Téléphone : 01.23.45.67.89.', 'wikipedia_fr')[0] == 'Téléphone : [TELEPHONE].'


def test_telephones_adjacents_et_annuaire_avec_numeros_de_rue():
    text = 'Veuillez m’appellez au 07.98.65.76.38 06.73.13.03.84 pour des informations.'
    assert mask_text(text, 'fineweb2_fra')[1]['TELEPHONE'] == 2
    text = '0254245923 24 la chataigne\n0601947276 3 lotissement lepiney\nAnnuaire des membres.'
    assert mask_text(text, 'fineweb2_fra')[1]['TELEPHONE'] == 2


def test_livres_poesie_et_breves_non_filtres():
    stage = QualityStage({}, None)
    for source in ['wikisource_fr', 'gutenberg_fr', 'wikinews_fr', 'wikipedia_fr', 'assemblee_nationale']:
        assert stage.apply(doc('poeme', 'Ô temps !\nÔ temps !\nÔ temps !', source)).keep
    assert stage.apply(doc('vide', 'Accueil. Contact.', 'commoncrawl')).reason == 'quality_short_web'
    repeated = 'Une annonce qui revient avec la même promesse chaque jour.\n' * 30
    assert stage.apply(doc('spam', repeated, 'commoncrawl')).reason == 'quality_repeated_lines'
    continuous = 'acheter cette offre ' * 80
    assert stage.apply(doc('spam', continuous, 'fineweb2_fra')).reason == 'quality_repeated_trigram'


def test_article_web_varie_garde():
    text = ('Les habitants du quartier ont ouvert un jardin partagé au printemps. '
            'Chaque famille peut cultiver des légumes dans une petite parcelle. '
            'La mairie fournit les outils et organise des ateliers gratuits le samedi. '
            'Les récoltes sont réparties entre les participants et la cantine voisine.')
    assert QualityStage({}, None).apply(doc('article', text, 'commoncrawl')).keep


def test_breve_francaise_preservee_interface_et_annuaire_retires():
    stage = QualityStage({}, None)
    text = 'La dermatite atopique est une maladie cutanée récurrente, inflammatoire et prurigineuse.'
    assert stage.apply(doc('breve', text, 'commoncrawl')).keep
    shell = 'Nous utilisons des cookies pour améliorer votre navigation et vous proposer des contenus adaptés.'
    assert stage.apply(doc('cookies', shell, 'commoncrawl')).reason == 'quality_interface_only'
    directory = doc('annuaire', 'Nom adresse contact à joindre au téléphone dans la commune ' * 3, 'commoncrawl')
    directory['meta']['privacy'] = {'masks': {'TELEPHONE': 6}}
    assert stage.apply(directory).reason == 'quality_contact_directory'
    directory['meta']['privacy'] = {'masks': {'TELEPHONE': 3}}
    # Un avis municipal peut contenir quelques coordonnées sans être un annuaire.
    assert stage.apply(directory).keep


def run_near(tmp, docs, *, workers=1, prune=False, params=None):
    raw = tmp / 'input'
    for source in sorted({d['source'] for d in docs}):
        write_docs(raw / source / 'part-00000.jsonl.zst', [d for d in docs if d['source'] == source])
    cfg = tmp / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'name': 'mini', 'sources': 'all', 'source_order': ['wikipedia_fr', 'commoncrawl'],
        'prune_intermediate_files': prune,
        'stages': [{'name': 'exclude_heldout', 'params': {'glob': str(tmp / 'heldout' / '*.txt')}},
                   {'name': 'minhash_dedup', 'params': {'reserve_bytes': 0, **(params or {})}},
                   {'name': 'exact_dedup'}]}))
    runner = Runner(cfg, input_dir=raw, out_root=tmp / 'output', workers=workers)
    assert runner.run() == 0
    return runner, tmp / 'output' / 'mini'


def text_words(n=300, prefix='terme'):
    return [f'{prefix}{i}' for i in range(n)]


def test_near_entre_sources_witness_conserve_et_reprise(tmp_path):
    a = text_words()
    b = a.copy()
    b[150:153] = ['nouveau', 'passage', 'ici']
    docs = [doc('a', ' '.join(a)), doc('b', ' '.join(b), 'commoncrawl'),
            doc('c', ' '.join(text_words(prefix='ailleurs')), 'commoncrawl'), doc('court', 'bonjour', 'commoncrawl')]
    runner, out = run_near(tmp_path, docs, workers=2)
    final = [d for p in sorted((out / '03_exact_dedup').glob('*/part-*')) for d in read_docs(p)]
    assert {d['id'] for d in final} == {'a', 'c', 'court'}
    st = json.loads((out / '02_minhash_dedup' / 'stats.json').read_text())
    assert st['report']['removed'] == 1
    assert st['report']['minimum_removed_similarity'] >= .8
    assert st['report']['audit_pairs'][0]['kept']['id'] == 'a'
    assert runner.run() == 0
    st = json.loads((out / '02_minhash_dedup' / 'stats.json').read_text())
    assert st['timing']['units_run'] == 0
    assert st['report']['removed'] == 1


def test_pas_de_chaine_transitive(tmp_path):
    a = text_words(300)
    b = a[:-22] + text_words(22, 'change')
    c = a[:-44] + text_words(22, 'autre') + text_words(22, 'change')
    ab, bc, ac = (jaccard(shingles(' '.join(x)), shingles(' '.join(y))) for x, y in [(a, b), (b, c), (a, c)])
    assert ab >= .8 and bc >= .8 and ac < .8
    # Plus de bandes pour que le test vérifie la règle de témoin, pas la chance LSH.
    _, out = run_near(tmp_path, [doc('a', ' '.join(a)), doc('b', ' '.join(b)), doc('c', ' '.join(c))],
                      params={'num_perm': 128, 'bands': 128, 'rows': 1})
    kept = [d['id'] for p in (out / '03_exact_dedup').glob('*/part-*') for d in read_docs(p)]
    assert kept == ['a', 'c']


def test_collision_lsh_confirmee_avant_retrait(tmp_path, monkeypatch):
    from pipeline.dedup import minhash
    monkeypatch.setattr(minhash, 'band_hashes', lambda array, band, rows: np.zeros(len(array), np.uint64))
    _, out = run_near(tmp_path, [doc('a', ' '.join(text_words())), doc('b', ' '.join(text_words(prefix='autre')))])
    st = json.loads((out / '02_minhash_dedup' / 'stats.json').read_text())
    assert st['report']['candidate_documents'] == 2
    assert st['report']['verified_comparisons'] == 1
    assert st['report']['removed'] == 0


def test_prune_limite_au_run_et_reconstruction(tmp_path):
    runner, out = run_near(tmp_path, [doc('a', ' '.join(text_words()))], prune=True)
    assert not list((out / '01_exclude_heldout').glob('*/part-*'))
    assert list((out / '03_exact_dedup').glob('*/part-*'))
    assert list((tmp_path / 'input').glob('*/part-*'))
    assert runner.run() == 0
    with pytest.raises(ValueError):
        runner._prune_intermediate(tmp_path / 'input')
    external = tmp_path / 'external'
    external.mkdir()
    linked = out / '04_link'
    linked.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        runner._prune_intermediate(linked)


def test_config_invalide_refusee():
    with pytest.raises(ValueError):
        MinHashDedupStage({'num_perm': 127}, SimpleNamespace())


def test_corpus_vide_apres_exclusion(tmp_path):
    (tmp_path / 'heldout').mkdir()
    (tmp_path / 'heldout' / 'ids.txt').write_text('a\n')
    _, out = run_near(tmp_path, [doc('a', ' '.join(text_words()))])
    st = json.loads((out / '02_minhash_dedup' / 'stats.json').read_text())
    assert st['total']['docs_in'] == st['total']['docs_out'] == 0


def test_interruption_apres_purge_partielle_rejouable(tmp_path, monkeypatch):
    from pipeline import run
    docs = [doc('a', ' '.join(text_words()))]
    runner, out = run_near(tmp_path, docs, prune=True)
    # Rejouer avec deux fichiers et un changement d'étape pour invalider le cache.
    write_docs(tmp_path / 'input' / 'wikipedia_fr' / 'part-00001.jsonl.zst', [doc('b', 'autre texte')])
    original = run._save_record
    def interrupt(stage_dir, unit, result):
        original(stage_dir, unit, result)
        if stage_dir.name == '02_minhash_dedup' and unit.record_name == 'part-00001.jsonl.zst':
            raise RuntimeError('Interruption contrôlée')
    with monkeypatch.context() as context:
        context.setattr(run, '_save_record', interrupt)
        with pytest.raises(RuntimeError, match='Interruption contrôlée'):
            runner.run()
    assert not list((out / '01_exclude_heldout').glob('*/part-00000*'))
    assert runner.run() == 0
    final = [d['id'] for p in (out / '03_exact_dedup').glob('*/part-*') for d in read_docs(p)]
    assert sorted(final) == ['a', 'b']
