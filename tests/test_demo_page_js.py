"""Vrai JavaScript de demo/index.html, exécuté avec node (sauf node absent) : relecture des réglages,
plafond des candidats, ligne « choix ancré », texte « À propos » et message d'accueil.

On n'extrait que des fonctions du <script> et on les exécute avec un faux DOM minimal : le test vérifie
le comportement (sorties), pas la présence de texte dans le fichier."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from demo import server

HTML = (Path(server.ROOT) / 'demo/index.html').read_text(encoding='utf-8')
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node absent')

FUNCTIONS = ['maxCandidates', 'validCandidates', 'revalidateSettings', 'uid', 'node', 'cleanMessage', 'load',
             'anchorLine', 'plural', 'personas', 'personaLabel', 'memoryLines', 'botDetails', 'aboutText',
             'welcomeMoliere']
STATEMENTS = ["const labels=", "const defaults=", "const KEY="]


def function_source(name):
    """Texte d'une déclaration `function name(...){...}` du script, accolades appariées (les accolades
    des chaînes de texte du script sont équilibrées ou absentes)."""
    start = HTML.index(f'function {name}(')
    depth, index = 0, HTML.index('{', start)
    for position in range(index, len(HTML)):
        depth += {'{': 1, '}': -1}.get(HTML[position], 0)
        if depth == 0:
            return HTML[start:position + 1]
    raise AssertionError(name)


def statement_source(prefix):
    # `const $=..., KEY='...'` : on ne garde que la partie utile.
    start = HTML.index(prefix) if prefix != 'const KEY=' else HTML.index("KEY='")
    end = HTML.index('\n', start)
    text = HTML[start:end]
    return ('const ' + text.rstrip(';').rstrip(',') + ';') if prefix == 'const KEY=' else text


def run_js(scenario):
    """Exécute le scénario JS (qui appelle out(valeur)) et renvoie les valeurs JSON affichées."""
    source = '\n'.join(
        ["'use strict';",
         "const store={};globalThis.sessionStorage={getItem:k=>(k in store?store[k]:null),setItem:(k,v)=>{store[k]=v}};",
         "const document={createElement:t=>({tag:t,className:'',textContent:'',children:[],append(...a){this.children.push(...a)}}),"
         "createElementNS:()=>({setAttribute(){},append(){}})};",
         "let info=null,state=null;const results=[];const out=v=>results.push(v);",
         *[statement_source(p) for p in STATEMENTS],
         *[function_source(name) for name in FUNCTIONS],
         scenario,
         "console.log(JSON.stringify(results));"])
    done = subprocess.run(['node', '-e', source], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def plain(text):
    return text.replace(' ', ' ').replace(' ', ' ')


def test_saved_candidates_are_read_back_with_the_servers_ceiling():
    results = run_js("""
const saved=value=>{const settings={};if(value!==undefined)settings.candidates=value;
 store[KEY]=JSON.stringify({version:1,chats:[],settings});return load().settings.candidates};
info={chat_limits:{candidates:32}};
for(const v of [16,32,33,0,-1,'8','',2.5,null,true,undefined])out(['cap32',v===undefined?'absent':v,saved(v)]);
info={chat_limits:{candidates:16}};
for(const v of [16,32])out(['cap16',v,saved(v)]);
info=null;out(['sans statut',32,saved(32)]);out(['sans statut',33,saved(33)]);
""")
    expected = {('cap32', '16'): 16, ('cap32', '32'): 32, ('cap32', '33'): None, ('cap32', '0'): None,
                ('cap32', '-1'): None, ('cap32', '8'): None, ('cap32', ''): None, ('cap32', '2.5'): None,
                ('cap32', 'None'): None, ('cap32', 'True'): None, ('cap32', 'absent'): None,
                ('cap16', '16'): 16, ('cap16', '32'): None, ('sans statut', '32'): 32, ('sans statut', '33'): None}
    got = {(name, str(value) if value != 'absent' else 'absent'): result for name, value, result in results}
    # Les valeurs JSON deviennent des str() Python : True -> 'True', None -> 'None'.
    assert got == expected


def test_ceiling_is_read_from_chat_limits_and_defaults_to_32():
    results = run_js("""
for(const limits of [{candidates:8},{candidates:32},{candidates:0},{candidates:'x'},{candidates:2.5},{},undefined]){
 info=limits===undefined?{}:{chat_limits:limits};out(maxCandidates())}
info=null;out(maxCandidates());
""")
    assert results == [8, 32, 32, 32, 32, 32, 32, 32]


def test_settings_are_checked_again_when_the_status_arrives():
    """Un nombre de candidats sauvegardé au-dessus du plafond du serveur redevient « défaut du modèle »."""
    results = run_js("""
for(const [saved,cap] of [[32,16],[16,16],[8,16],[null,16],[33,32]]){
 info={chat_limits:{candidates:cap}};state={settings:{candidates:saved}};revalidateSettings();out(state.settings.candidates)}
""")
    assert results == [None, 16, 8, None, None]


@pytest.mark.parametrize('anchor, shown', [(1.0, True), (0.5, True), (0, False), (0.0, False), (-1, False),
                                           (None, False), ('1', False)])
def test_anchor_line_follows_anchor_mu(anchor, shown):
    meta = {'tokens_generated': 5, 'seconds': 0.5}
    if anchor is not None:
        meta['anchor_mu'] = anchor
    results = run_js(f"""
const meta={json.dumps(meta)};
out(anchorLine(meta));
const details=botDetails({{meta,persona:null}},'francais_chat');
out(details.children.map(c=>c.textContent));
""")
    line, paragraphs = results
    assert (line == 'Choix ancré sur le dernier message.') is shown
    assert ('Choix ancré sur le dernier message.' in paragraphs) is shown
    # Absente : aucun plantage, simplement pas de ligne.
    if anchor is None:
        assert line is None


S03_STATUS = {
    'models': {'moliere_chat': True, 'francais_chat': True},
    'chatbots': {
        'moliere_chat': {
            'published_release': 'chatbot-moliere-20261116',
            'training': {'base': {'tokens_vus': 2000027648, 'n_params': 54749440},
                         'part_synthetique': {'part_dans_l_etape': 0.0302},
                         'environnement': {'gpu': {'name': 'NVIDIA GeForce RTX 5090'}}}},
        'francais_chat': {'training': {'base': {'tokens_vus': 2000027648, 'n_params': 54749440}}},
    },
}
S03_TEXT = ("Le chatbot Molière est un modèle de 54 749 440 paramètres, entraîné de zéro sur une RTX 5090 : "
            "pré-entraînement sur 2 000 027 648 tokens, puis affinage sur du théâtre, dont 3 % de conversations "
            "écrites par un autre modèle de langue (gemma 4, poids ouverts). Il est publié dans la release GitHub "
            "chatbot-moliere-20261116. Ses réponses ont été jugées à l’aveugle par des agents IA, jamais par des "
            "personnes : 96 sans défaut sur 144 le 16/11/2026.")


def about(data):
    return plain(run_js(f"out(aboutText({json.dumps(data)}));")[0])


def variant(**changes):
    data = json.loads(json.dumps(S03_STATUS))
    for path, value in changes.items():
        target = data
        *head, last = path.split('.')
        for key in head:
            target = target[key]
        if value is Ellipsis:
            target.pop(last)
        else:
            target[last] = value
    return data


def test_about_text_of_the_published_model():
    text = about(S03_STATUS)
    assert text.startswith(S03_TEXT)
    # Même base de paramètres : pas de nombre répété pour le second chatbot ; français « moderne » partout.
    assert '« français moderne » a la même base, affinée sur des dialogues de romans et des scènes de théâtre en prose' in text
    assert 'courant' not in text and 'modèle français' not in text


def test_about_text_reads_the_numbers_of_the_installed_model_not_constants():
    """Un autre modèle dans data/models/chatbot_moliere : ni release, ni 96/144, ni « gemma », ni RTX en dur."""
    other = variant(**{'chatbots.moliere_chat.published_release': None,
                       'chatbots.moliere_chat.training.base.tokens_vus': 1000000000,
                       'chatbots.moliere_chat.training.base.n_params': 29600000,
                       'chatbots.moliere_chat.training.part_synthetique.part_dans_l_etape': 0.05,
                       'chatbots.moliere_chat.training.environnement': {'gpu': {'name': 'NVIDIA A100'}}})
    text = about(other)
    assert 'de 29 600 000 paramètres' in text and 'sur 1 000 000 000 tokens' in text
    assert 'sur une A100' in text and 'dont 5 % de conversations écrites par un autre modèle de langue' in text
    for forbidden in ('RTX 5090', '2 000 027 648', 'chatbot-moliere-20261116', '96', '144', 'gemma'):
        assert forbidden not in text.split('Le chatbot « français')[0], forbidden
    assert 'aucune évaluation à l’aveugle' in text


def test_about_text_without_synthetic_share_or_report():
    none_share = about(variant(**{'chatbots.moliere_chat.published_release': None,
                                  'chatbots.moliere_chat.training.part_synthetique': None}))
    assert 'autre modèle de langue' not in none_share and 'puis affinage sur du théâtre. ' in none_share
    zero = about(variant(**{'chatbots.moliere_chat.training.part_synthetique.part_dans_l_etape': 0}))
    assert 'autre modèle de langue' not in zero
    no_report = about(variant(**{'chatbots.moliere_chat.training': None, 'chatbots.moliere_chat.published_release': None}))
    assert no_report.startswith('Le chatbot Molière est un modèle, entraîné de zéro : pré-entraînement, puis affinage')


def test_about_text_shows_a_paragraph_only_for_available_chatbots():
    no_french = about(variant(**{'models.francais_chat': False}))
    assert 'français moderne' not in no_french and no_french.startswith('Le chatbot Molière')
    no_moliere = about(variant(**{'models.moliere_chat': False}))
    assert 'Le chatbot Molière' not in no_moliere and no_moliere.startswith('Le chatbot « français moderne »')
    neither = about(variant(**{'models.moliere_chat': False, 'models.francais_chat': False}))
    assert neither == ''
    assert about({}) == ''  # statut sans aucun bloc : pas de plantage


def test_french_chatbot_shows_its_own_parameter_count_when_different():
    text = about(variant(**{'chatbots.francais_chat.training.base.n_params': 29600000}))
    assert '(29 600 000 paramètres)' in text


def test_welcome_message_says_part_of_the_tuning_was_written_by_another_model():
    results = run_js(f"""
info={json.dumps({'chatbots': S03_STATUS['chatbots']})};out(welcomeMoliere());
info={{chatbots:{{moliere_chat:{{training:{{part_synthetique:null}}}}}}}};out(welcomeMoliere());
info=null;out(welcomeMoliere());
""")
    assert 'dont 3 % de conversations écrites par un autre modèle de langue' in plain(results[0])
    assert 'entraîné à partir de zéro, puis affiné sur du théâtre' in plain(results[0])
    for later in results[1:]:
        assert 'autre modèle' not in later and 'affiné sur du théâtre :' in plain(later)


def test_real_installed_status_matches_the_published_model_text():
    """Avec le vrai dossier installé (s'il est le modèle publié) : la page reçoit les vrais nombres."""
    if not server.chatbot_ready('moliere_chat'):
        pytest.skip('modèle Molière non installé')
    status = server.summary()
    bot = status['chatbots']['moliere_chat']
    if bot['published_release'] != 'chatbot-moliere-20261116':
        pytest.skip('le modèle installé n’est pas le modèle publié S03')
    text = about({'models': status['models'], 'chatbots': status['chatbots']})
    assert text.startswith(S03_TEXT)
