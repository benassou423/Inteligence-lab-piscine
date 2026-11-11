"""Préserver la prose, les chiffres et la poésie en retirant les résidus vérifiables."""
import copy
from types import SimpleNamespace

import pytest

from pipeline.clean.targeted import TargetedCleanStage, clean_structure, contents_only


@pytest.mark.parametrize('text', [
    "Il est né le 18 juin 1904. Sa ville compte 12 345 habitants.",
    "Un nom étranger : David Glantz.\n\nBibliographie\nThe Battle of Kursk (1999).",
    "Demain, dès l'aube, à l'heure où blanchit la campagne,\nJe partirai. Vois-tu, je sais que tu m'attends.",
    "En programmation, a || b représente une disjonction.",
    "La propriété rowspan=2 permet de fusionner deux cellules.",
    "Non non non !\n\nLaissez-nous passer.\n",
    "En Wikidata, Q12345 désigne un élément de la base.",
])
def test_ordinary_text_preserved_byte_for_byte(text):
    assert clean_structure(text, 'wikipedia_fr')[0] == text


def test_orphan_table_keeps_french_paragraph_after_closing_marker():
    body = "L'équipe est composée de dix-neuf coureurs. Elle participe à la course."
    text = "Biographie en français.\n\nEffectif |- | Jean || 12 || France |} " + body
    cleaned, _ = clean_structure(text, 'wikipedia_fr')
    assert 'Biographie en français.' in cleaned and body in cleaned
    assert '|-' not in cleaned and '||' not in cleaned and 'Jean' not in cleaned


def test_balanced_nested_table_keeps_surrounding_prose():
    text = "Avant le tableau.\n{|\n| cellule {|\n| sous-tableau\n|}\n|}\nAprès le tableau."
    cleaned, _ = clean_structure(text, 'wikipedia_fr')
    assert cleaned == 'Avant le tableau.\n\nAprès le tableau.'


def test_unclosed_table_does_not_eat_later_paragraph():
    text = "Introduction.\n{| class=wikitable\n| 1 || 2\n\nLa suite contient une vraie phrase en français."
    cleaned, _ = clean_structure(text, 'wikipedia_fr')
    assert 'La suite contient une vraie phrase en français.' in cleaned
    assert '{|' not in cleaned


def test_distant_table_closing_never_removes_plain_prose():
    prose = "L'emploi de ces divers éléments sonores et leur application constitue l'art de l'instrumentation."
    text = '{| | Description\n\n' + prose + '\n\n|}'
    assert prose in clean_structure(text, 'wikipedia_fr')[0]
    assert clean_structure(text, 'wikisource_fr')[0] == text


def test_literary_parallel_translation_is_not_deleted():
    text = '{| |- | Une phrase française. || Another sentence. |}\nPuis la narration continue.'
    assert clean_structure(text, 'wikisource_fr')[0] == text


def test_orphan_cells_do_not_lose_their_person_names():
    text = 'Introduction.\nNom | rowspan=2 | Jean Martin || Élu en 1898.\nRéélu en 1900.\nRetrait. |- | Pierre || 1902'
    assert clean_structure(text, 'wikipedia_fr')[0] == text


def test_infobox_and_layout_only_not_ordinary_templates():
    text = '{{Infobox Personne\n| nom = Personne\n}}\n\nUne vraie biographie.\ncentré|180px\nQ12345'
    cleaned, _ = clean_structure(text, 'wikipedia_fr')
    assert cleaned == 'Une vraie biographie.'
    citation = '{{citation|Une phrase importante.}}'
    assert clean_structure(citation, 'wikipedia_fr')[0] == citation


def test_repetition_removal_preserves_neighbouring_sentences():
    text = 'Le début.\n' + 'Bretagne ' * 30 + '\nLa suite.'
    cleaned, _ = clean_structure(text, 'wikipedia_fr')
    assert cleaned.count('Bretagne') == 1
    assert cleaned.startswith('Le début.') and cleaned.endswith('La suite.')


def test_contents_without_body_but_poem_and_short_news_survive():
    contents = 'Sommaire\n' + '\n'.join('Lettre ' + x for x in ['I', 'II', 'III', 'IV', 'V', 'VI'])
    assert contents_only(contents)
    assert not contents_only(contents + '\nJe me souviens de toi, au bord de la rivière.')
    assert not contents_only('Le poète est mort hier. Il avait 74 ans.')


def test_stage_immutable_input_and_honest_score_provenance():
    doc = {'id': 'wiki:1', 'source': 'wikipedia_fr', 'text': 'Bonjour.\n| 1 || 2 || 3 |}\nAu revoir.',
           'meta': {'langid_fr': .9, 'langid_model': 'glotlid', 'title': 'Exemple'}}
    before = copy.deepcopy(doc)
    result = TargetedCleanStage({}, SimpleNamespace()).apply(doc)
    assert doc == before and result.keep
    assert 'langid_fr' not in result.doc['meta']
    assert result.doc['meta']['targeted_clean']['previous_langid']['langid_fr'] == .9
    assert result.doc['text'] == 'Bonjour.\n\nAu revoir.'


def test_empty_table_rejected_and_unrelated_sources_preserved():
    stage = TargetedCleanStage({}, SimpleNamespace())
    doc = {'id': 'w:2', 'source': 'wikipedia_fr', 'text': '| 1 || 2 || 3 |}', 'meta': {}}
    assert not stage.apply(doc).keep
    doc.update(source='commoncrawl', text='Le code || est expliqué dans cette phrase.')
    assert stage.apply(doc).doc == doc


def test_single_stray_pipe_keeps_short_french_biography():
    text = "Ben Ryan (né le 16 octobre 1988 à Brighton) est un joueur professionnel américain de hockey sur glace."
    stage = TargetedCleanStage({}, SimpleNamespace())
    result = stage.apply({'id': 'w:3', 'source': 'wikipedia_fr', 'text': '|' + text, 'meta': {}})
    assert result.keep and result.doc['text'] == text


@pytest.mark.parametrize('text', [
    r'jaccard(X,Y)=\frac{|X\cap Y|}{|X\cup Y|}.',
    r'cosinus(X,Y)=\frac{|X\cap Y|}\sqrt{|X||Y|}.',
    r'dice(X,Y)=\frac{2|X\cap Y|}{|X|+|Y|}.',
    'cardinal={|X|}',
    'cardinal={| X |}',
    'cardinal={|X|+|Y|}',
    '|X||Y||Z|',
    '|-x|',
    '|+x|',
    '1 || 2 || 3 || 4',
])
def test_math_delimiters_are_not_wikitables(text):
    assert clean_structure(text, 'wikipedia_fr')[0] == text
