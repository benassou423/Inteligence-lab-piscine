"""Second chatbot de la démo : « Français moderne · chatbot » (francais_chat, format « tirets »).

Le vrai modèle n'existe pas encore (entraînement en attente de l'accord de Ben) : on teste le
routage, les validations et le statut avec de faux dossiers et un faux moteur injecté."""
import json
import sys
import types
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

import demo.server as server
from demo.server import chat, summary
import test_demo_moliere_chat as moliere
from test_demo import local_server

FRENCH_CONFIG = {'format': 'tirets', 'candidates': 8, 'temperature': 0.7, 'max_new_tokens': 96}
HELLO = [{'role': 'user', 'content': 'Salut, tu fais quoi ce week-end ?'}]


class FakeFrenchBot(moliere.FakeBot):
    """Même interface que ChatBot en format « tirets » : aucun personnage, persona « » renvoyé."""

    def __init__(self, config):
        super().__init__(config)
        self.personas = []

    def reply(self, messages, persona=None, **options):
        assert persona is None, 'le format tirets ne reçoit jamais de personnage'
        out = super().reply(messages, persona='MOLIÈRE', **options)  # réutilise l'enregistrement
        self.calls[-1]['persona'] = persona
        out.update(reply='Je vais au marché, et toi ?', persona='', interlocutor='',
                   history_turns_used=len(messages), context_truncated=len(messages) > 3)
        return out


class Router:
    """Faux module chatbot.generate : le bon faux moteur selon le dossier chargé."""
    loads = []

    @classmethod
    def load(cls, model_dir, device='cpu'):
        kind = FakeFrenchBot if model_dir.name == 'chatbot_francais' else moliere.FakeBot
        bot = kind.load(model_dir, device)
        cls.loads.append((model_dir.name, bot))
        return bot


def write_model(folder, config):
    folder.mkdir()
    (folder / 'ckpt.pt').write_bytes(b'faux poids')
    (folder / 'tokenizer.json').write_text('{}', encoding='utf-8')
    (folder / 'chat_config.json').write_text(json.dumps(config), encoding='utf-8')
    return folder


@pytest.fixture
def folders(monkeypatch, tmp_path):
    """Deux dossiers factices (Molière et français moderne) et faux modules chatbot."""
    mol = write_model(tmp_path / 'chatbot_moliere', moliere.CONFIG)
    fr = write_model(tmp_path / 'chatbot_francais', FRENCH_CONFIG)
    monkeypatch.setitem(server.CHATBOTS['moliere_chat'], 'dir', mol)
    monkeypatch.setitem(server.CHATBOTS['francais_chat'], 'dir', fr)
    generate = types.ModuleType('chatbot.generate')
    generate.ChatBot = Router
    fmt = types.ModuleType('chatbot.format')
    fmt.PERSONAS = moliere.PERSONAS
    fmt.clean_text = moliere.real_clean_text()
    monkeypatch.setitem(sys.modules, 'chatbot.generate', generate)
    monkeypatch.setitem(sys.modules, 'chatbot.format', fmt)
    Router.loads, moliere.FakeBot.loads = [], []
    server.chatbot_engine.cache_clear()
    yield mol, fr
    server.chatbot_engine.cache_clear()


def french_calls():
    return [bot for name, bot in Router.loads if name == 'chatbot_francais'][-1].calls


def test_francais_chat_is_routed_to_its_own_folder(folders):
    result = chat({'model': 'francais_chat', 'messages': HELLO})
    assert set(result) == moliere.REPLY_KEYS
    assert result['model'] == 'francais_chat' and result['device'] == 'cpu'
    assert result['persona'] == '' and result['interlocutor'] == ''
    assert result['reply'] == 'Je vais au marché, et toi ?' and result['history_messages'] == 1
    assert [name for name, _ in Router.loads] == ['chatbot_francais']
    assert french_calls()[-1]['messages'] == HELLO and french_calls()[-1]['persona'] is None


def test_history_and_settings_reach_the_french_engine(folders):
    messages = [{'role': 'user', 'content': 'Salut', 'id': 'x'},
                {'role': 'assistant', 'content': 'Salut ! Ça va ?', 'meta': {}},
                {'role': 'user', 'content': 'Oui, et toi ?'}]
    result = chat({'model': 'francais_chat', 'messages': messages, 'seed': 12, 'candidates': 4,
                   'temperature': 0.9, 'max_new_tokens': 256})
    call = french_calls()[-1]
    assert call['messages'] == [{k: m[k] for k in ('role', 'content')} for m in messages]
    assert (call['seed'], call['candidates'], call['temperature'], call['max_new_tokens']) == (12, 4, 0.9, 255)
    assert result['history_turns_used'] == 3 and result['seed'] == 12
    chat({'model': 'francais_chat', 'messages': HELLO})
    # Rien d'imposé par la démo : réglages de chat_config.json du français moderne.
    assert (french_calls()[-1]['temperature'], french_calls()[-1]['max_new_tokens']) == (0.7, 96)


@pytest.mark.parametrize('persona', ['MOLIÈRE', '', 'HARPAGON'])
def test_persona_is_refused_for_the_french_chatbot(folders, persona):
    with pytest.raises(ValueError, match='pas de personnage'):
        chat({'model': 'francais_chat', 'persona': persona, 'messages': HELLO})
    assert all(not bot.calls for _, bot in Router.loads)


@pytest.mark.parametrize('persona', [12, ['A']])
def test_persona_of_wrong_type_is_refused_before_loading(folders, persona):
    with pytest.raises(ValueError, match='désigné par son nom'):
        chat({'model': 'francais_chat', 'persona': persona, 'messages': HELLO})
    assert Router.loads == []


def test_null_persona_is_accepted(folders):
    assert chat({'model': 'francais_chat', 'persona': None, 'messages': HELLO})['persona'] == ''


@pytest.mark.parametrize('extra', [
    {'candidates': 33}, {'temperature': 1.6}, {'temperature': float('nan')}, {'max_new_tokens': 0},
    {'seed': -1}, {'seed': True},
])
def test_out_of_bounds_settings_are_rejected(folders, extra):
    with pytest.raises(ValueError):
        chat({'model': 'francais_chat', 'messages': HELLO, **extra})
    assert Router.loads == []


@pytest.mark.parametrize('content', ['<|endoftext|>', ' ', 'a' * 2001])
def test_invalid_messages_never_reach_the_engine(folders, content):
    with pytest.raises(ValueError):
        chat({'model': 'francais_chat', 'messages': [{'role': 'user', 'content': content}]})
    assert Router.loads == []


def test_each_chatbot_is_loaded_once_and_reloaded_alone(folders):
    """Un moteur par dossier : alterner ne recharge rien ; un réexport ne recharge que le sien."""
    mol, fr = folders
    for _ in range(2):
        chat({'model': 'moliere_chat', 'messages': HELLO})
        chat({'model': 'francais_chat', 'messages': HELLO})
    assert [name for name, _ in Router.loads] == ['chatbot_moliere', 'chatbot_francais']
    moliere.rewrite(fr / 'chat_config.json', json.dumps(dict(FRENCH_CONFIG, candidates=3)))
    assert chat({'model': 'francais_chat', 'messages': HELLO})['candidates'] == 3
    chat({'model': 'moliere_chat', 'messages': HELLO})
    assert [name for name, _ in Router.loads] == ['chatbot_moliere', 'chatbot_francais', 'chatbot_francais']


def test_status_without_french_model(monkeypatch, tmp_path):
    monkeypatch.setitem(server.CHATBOTS['francais_chat'], 'dir', tmp_path / 'chatbot_francais')
    status = summary()
    assert status['ui_version'] == 'chat-v4'
    assert status['models']['francais_chat'] is False and status['chatbots']['francais_chat'] is None
    with pytest.raises(ValueError, match='français moderne n’est pas encore entraîné'):
        chat({'model': 'francais_chat', 'messages': HELLO})


@pytest.mark.parametrize('missing', ['ckpt.pt', 'tokenizer.json', 'chat_config.json'])
def test_incomplete_french_folder_is_not_announced(folders, missing):
    _, fr = folders
    (fr / missing).unlink()
    status = summary()
    assert status['models']['francais_chat'] is False and status['chatbots']['francais_chat'] is None
    assert status['models']['moliere_chat'] is True
    assert Router.loads == []


def test_status_with_both_chatbots_reads_json_only(folders):
    _, fr = folders
    status = summary()
    assert status['models']['francais_chat'] is True and status['models']['moliere_chat'] is True
    french = status['chatbots']['francais_chat']
    assert french['format'] == 'tirets' and french['personas'] == [] and french['default_persona'] is None
    assert french['defaults'] == FRENCH_CONFIG and french['training'] is None
    # Le bloc « chatbot » reste celui de Molière (compatibilité), identique à chatbots.moliere_chat.
    assert status['chatbot'] == status['chatbots']['moliere_chat']
    assert status['chatbot']['format'] == 'repliques'
    assert [p['name'] for p in status['chatbot']['personas']] == ['MOLIÈRE', 'HARPAGON']
    (fr / 'training_report.json').write_text(json.dumps({'val_bpb': 1.5}), encoding='utf-8')
    assert summary()['chatbots']['francais_chat']['training'] == {'val_bpb': 1.5}
    assert Router.loads == []


def test_loading_error_names_the_french_chatbot(folders):
    _, fr = folders
    (fr / 'chat_config.json').write_text('{x', encoding='utf-8')
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            moliere.post(base, {'model': 'francais_chat', 'messages': HELLO})
        body = json.load(error.value)
    assert error.value.code == 503 and 'français moderne' in body['error'] and 'Expecting' not in body['error']
    # Le chatbot Molière n'est pas touché par la panne de l'autre.
    assert chat({'model': 'moliere_chat', 'messages': HELLO})['reply'] == 'Fort bien, Monsieur.'


def test_moliere_chat_is_unchanged_next_to_the_french_one(folders):
    result = chat({'messages': HELLO})  # route par défaut : toujours le chatbot Molière
    assert result['model'] == 'moliere_chat' and result['persona'] == 'MOLIÈRE'
    assert chat({'model': 'moliere_chat', 'persona': 'HARPAGON', 'messages': HELLO})['interlocutor'] == 'VALÈRE'
    with pytest.raises(ValueError, match='Personnage inconnu'):
        chat({'model': 'moliere_chat', 'persona': 'TARTUFFE', 'messages': HELLO})
    with pytest.raises(ValueError, match='quatre modèles'):
        chat({'model': 'gpt', 'messages': HELLO})


def test_completion_models_are_unchanged(folders):
    moliere.previous.test_explicit_seed_is_reported_and_reproduces_a_generation()
    assert Router.loads == []


def test_http_route_and_page(folders):
    with local_server() as base:
        with urlopen(base + '/api/status') as response:
            status = json.load(response)
        assert status['models']['francais_chat'] and status['chatbots']['francais_chat']['personas'] == []
        with moliere.post(base, {'model': 'francais_chat', 'messages': HELLO, 'seed': 3}) as response:
            result = json.load(response)
        assert result['model'] == 'francais_chat' and result['seed'] == 3 and result['persona'] == ''
        with pytest.raises(HTTPError) as error:
            moliere.post(base, {'model': 'francais_chat', 'persona': 'MOLIÈRE', 'messages': HELLO})
        assert error.value.code == 400 and 'pas de personnage' in json.load(error.value)['error']
        with urlopen(base + '/') as response:
            html = response.read().decode()
    order = [html.index(f'<option value="{name}">') for name in
             ('moliere_chat', 'francais_chat', 'competition', 'moliere')]
    assert order == sorted(order)
    assert 'Français moderne · chatbot' in html and "'Discutons.'" in html
    assert "francais_chat:'Assistant'" in html and "'chat-v4'" in html
    assert 'innerHTML' not in html and 'insertAdjacentHTML' not in html and 'outerHTML' not in html
    # Mémoire de conversation affichée pour les deux chatbots, avertissement seulement si tronqué.
    assert 'précédent' in html and "'Contexte : '" in html and 'if(meta.context_truncated===true)' in html
    # Personnage envoyé seulement au chatbot Molière ; anciennes conversations toujours relues.
    assert "if(chat.model==='moliere_chat'){chat.persona=personaOf(chat)" in html
    assert "KEY='ftb-chat-v1'" in html and 'v.version!==1' in html


@pytest.mark.skipif(not (server.ROOT / 'data/models/chatbot_francais/ckpt.pt').is_file(),
                    reason='Chatbot français moderne pas encore entraîné (data/models/chatbot_francais/ckpt.pt).')
def test_real_french_chatbot_replies_and_reproduces_with_seed():
    server.chatbot_engine.cache_clear()
    try:
        status = summary()['chatbots']['francais_chat']
        assert status['format'] == 'tirets' and status['personas'] == []
        payload = {'model': 'francais_chat', 'messages': HELLO, 'seed': 7, 'candidates': 2, 'max_new_tokens': 16}
        first, second = chat(payload), chat(payload)
        assert set(first) == moliere.REPLY_KEYS and isinstance(first['reply'], str) and first['reply']
        assert first['persona'] == '' and first['seed'] == 7 and first['reply'] == second['reply']
        assert 1 <= first['tokens_generated'] <= 16 and first['history_turns_used'] >= 1
    finally:
        server.chatbot_engine.cache_clear()
