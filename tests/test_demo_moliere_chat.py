"""Chatbot Molière dans la démo : routage, validations et statut, avec un faux moteur injecté."""
import json
import os
from pathlib import Path
import socket
import sys
import time
import types
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import demo.server as server
from demo.server import autograd, chat, generate, summary, tokenize
import test_demo_chat as previous
from test_demo import local_server

PERSONAS = [
    {'name': 'MOLIÈRE', 'label': 'Molière', 'interlocutor': 'LE PUBLIC', 'about': 'L’auteur lui-même.'},
    {'name': 'HARPAGON', 'label': "Harpagon (L'Avare)", 'interlocutor': 'VALÈRE', 'about': 'Un avare.'},
]
CONFIG = {'default_persona': 'MOLIÈRE', 'candidates': 8, 'temperature': 0.8, 'max_new_tokens': 120}
REPLY_KEYS = {'reply', 'persona', 'interlocutor', 'seed', 'tokens_generated', 'seconds', 'context_tokens',
              'context_limit', 'context_truncated', 'history_messages', 'history_turns_used', 'candidates',
              'anchor_mu', 'finish', 'model', 'device'}
HELLO = [{'role': 'user', 'content': 'Bonjour, comment vous portez-vous ?'}]


class FakeBot:
    """Même interface que chatbot.generate.ChatBot ; enregistre chaque appel.

    Comme le vrai ChatBot.load : lit chat_config.json (JSONDecodeError si illisible) et exige
    ckpt.pt et tokenizer.json. Un réglage absent (None) prend la valeur de la config chargée."""
    loads = []

    def __init__(self, config):
        self.personas = PERSONAS
        self.config = dict(CONFIG, **config)
        self.calls = []

    @classmethod
    def load(cls, model_dir, device='cpu'):
        folder = Path(model_dir)
        config = json.loads((folder / 'chat_config.json').read_text(encoding='utf-8'))
        for name in ('ckpt.pt', 'tokenizer.json'):
            if not (folder / name).is_file():
                raise FileNotFoundError(name)
        bot = cls(config)
        cls.loads.append((model_dir, device, bot))
        return bot

    def reply(self, messages, persona=None, seed=None, temperature=None, top_p=None,
              max_new_tokens=None, candidates=None):
        cfg = self.config
        persona = persona or cfg['default_persona']
        candidates = cfg['candidates'] if candidates is None else candidates
        self.calls.append({'messages': messages, 'persona': persona, 'seed': seed,
                           'temperature': cfg['temperature'] if temperature is None else temperature,
                           'max_new_tokens': cfg['max_new_tokens'] if max_new_tokens is None else max_new_tokens,
                           'candidates': candidates,
                           'sent': {'temperature': temperature, 'max_new_tokens': max_new_tokens}})
        interlocutor = next(p['interlocutor'] for p in PERSONAS if p['name'] == persona)
        return {'reply': 'Fort bien, Monsieur.', 'persona': persona, 'interlocutor': interlocutor,
                'seed': 4242 if seed is None else seed, 'tokens_generated': 5, 'seconds': 0.01,
                'context_tokens': 20, 'context_limit': 512, 'context_truncated': False,
                'history_turns_used': len(messages) // 2, 'candidates': candidates,
                'anchor_mu': cfg.get('anchor_mu', 0.0), 'finish': 'speaker', 'score': -1.5}


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    """Dossier modèle factice (aucun poids réel) et faux modules chatbot injectés."""
    folder = tmp_path / 'chatbot_moliere'
    folder.mkdir()
    (folder / 'ckpt.pt').write_bytes(b'faux poids')
    (folder / 'tokenizer.json').write_text('{}', encoding='utf-8')
    (folder / 'chat_config.json').write_text(json.dumps(CONFIG), encoding='utf-8')
    monkeypatch.setitem(server.CHATBOTS['moliere_chat'], 'dir', folder)
    generate = types.ModuleType('chatbot.generate')
    generate.ChatBot = FakeBot
    fmt = types.ModuleType('chatbot.format')
    fmt.PERSONAS = PERSONAS
    fmt.clean_text = real_clean_text()
    monkeypatch.setitem(sys.modules, 'chatbot.generate', generate)
    monkeypatch.setitem(sys.modules, 'chatbot.format', fmt)
    FakeBot.loads = []
    server.chatbot_engine.cache_clear()
    yield folder
    server.chatbot_engine.cache_clear()


def real_clean_text():
    """Le vrai chatbot.format.clean_text ; repli minimal s'il est momentanément cassé."""
    try:
        from chatbot.format import clean_text
        return clean_text
    except Exception:
        return lambda text: text.replace('<|endoftext|>', ' ').replace('\u200b', '').strip()


def last_call():
    return FakeBot.loads[-1][2].calls[-1]


def rewrite(path, text):
    """Réécrire un fichier et décaler sa date : la signature du modèle change à coup sûr."""
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text, encoding='utf-8')
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


def test_default_route_uses_chatbot_and_default_persona(model_dir):
    result = chat({'messages': HELLO})
    assert set(result) == REPLY_KEYS
    assert result['model'] == 'moliere_chat' and result['device'] == 'cpu'
    assert result['persona'] == 'MOLIÈRE' and result['interlocutor'] == 'LE PUBLIC'
    assert result['history_messages'] == 1 and result['reply'] == 'Fort bien, Monsieur.'
    assert last_call()['messages'] == HELLO and last_call()['persona'] == 'MOLIÈRE'


def test_engine_is_loaded_once_on_cpu(model_dir):
    chat({'model': 'moliere_chat', 'messages': HELLO})
    chat({'model': 'moliere_chat', 'messages': HELLO})
    assert len(FakeBot.loads) == 1
    assert FakeBot.loads[0][:2] == (model_dir, 'cpu')
    assert len(FakeBot.loads[0][2].calls) == 2


def test_history_is_forwarded_without_extra_fields(model_dir):
    messages = [{'role': 'user', 'content': 'Bonjour', 'id': 'x'},
                {'role': 'assistant', 'content': 'Serviteur.', 'meta': {}},
                {'role': 'user', 'content': 'Et les médecins ?'}]
    result = chat({'model': 'moliere_chat', 'persona': 'HARPAGON', 'messages': messages})
    assert last_call()['messages'] == [{k: m[k] for k in ('role', 'content')} for m in messages]
    assert result['persona'] == 'HARPAGON' and result['interlocutor'] == 'VALÈRE'
    assert result['history_messages'] == 3 and result['history_turns_used'] == 1


@pytest.mark.parametrize('persona', ['TARTUFFE', '', 'molière', 12, ['HARPAGON']])
def test_unknown_persona_is_rejected(model_dir, persona):
    with pytest.raises(ValueError):
        chat({'model': 'moliere_chat', 'persona': persona, 'messages': HELLO})


@pytest.mark.parametrize('messages', [
    None, [], [{'role': 'assistant', 'content': 'Réponse sans question'}],
    [{'role': 'user', 'content': 'Bonjour'}, {'role': 'assistant', 'content': 'Bonsoir'}],
    [{'role': 'user', 'content': ' '}], [{'role': 'user', 'content': 'a' * 2001}],
    [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'a' * 700} for i in range(19)],
    [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'Bonjour'} for i in range(21)],
])
def test_invalid_messages_never_reach_the_engine(model_dir, messages):
    with pytest.raises(ValueError):
        chat({'model': 'moliere_chat', 'messages': messages})
    assert FakeBot.loads == []


@pytest.mark.parametrize('extra', [
    {'candidates': 0}, {'candidates': 33}, {'candidates': True}, {'candidates': 2.5}, {'candidates': '8'},
    {'temperature': 0.05}, {'temperature': 1.6}, {'temperature': float('nan')}, {'temperature': True},
    {'temperature': float('inf')}, {'temperature': int('9' * 400)}, {'temperature': None},
    {'max_new_tokens': 0}, {'max_new_tokens': 257}, {'max_new_tokens': None},
    {'seed': -1}, {'seed': 2**32}, {'seed': True}, {'seed': '7'},
    {'model': 'gpt'},
])
def test_out_of_bounds_settings_are_rejected(model_dir, extra):
    with pytest.raises(ValueError):
        chat({'model': 'moliere_chat', 'messages': HELLO, **extra})


def test_settings_are_forwarded_and_defaults_left_to_engine(model_dir):
    chat({'model': 'moliere_chat', 'messages': HELLO})
    call = last_call()
    assert call['seed'] is None and call['candidates'] == 8 and call['max_new_tokens'] == 120
    # Rien d'imposé par la démo : température et longueur viennent de chat_config.json.
    assert call['sent'] == {'temperature': None, 'max_new_tokens': None} and call['temperature'] == 0.8
    result = chat({'model': 'moliere_chat', 'messages': HELLO, 'seed': 2547, 'candidates': 32,
                   'temperature': 1.2, 'max_new_tokens': 256})
    call = last_call()
    # 256 (« Longue » dans l'interface) est ramené au maximum du moteur : 255 < 512 // 2.
    assert (call['seed'], call['candidates'], call['temperature'], call['max_new_tokens']) == (2547, 32, 1.2, 255)
    assert result['seed'] == 2547 and result['candidates'] == 32


def test_missing_seed_is_drawn_by_engine_and_reported(model_dir):
    result = chat({'model': 'moliere_chat', 'messages': HELLO, 'seed': None})
    assert last_call()['seed'] is None and result['seed'] == 4242


def test_status_without_trained_chatbot(monkeypatch, tmp_path):
    monkeypatch.setitem(server.CHATBOTS['moliere_chat'], 'dir', tmp_path / 'absent')
    status = summary()
    assert status['ui_version'] == 'chat-v4'
    assert status['models']['moliere_chat'] is False and status['chatbot'] is None
    with pytest.raises(ValueError, match='pas encore entraîné'):
        chat({'model': 'moliere_chat', 'messages': HELLO})


def test_default_route_falls_back_to_competition_without_chatbot(monkeypatch, tmp_path):
    monkeypatch.setitem(server.CHATBOTS['moliere_chat'], 'dir', tmp_path / 'absent')
    seen = []

    def fake_complete(name, prompt, payload, max_tokens=128):
        seen.append((name, max_tokens))
        return {'prompt': prompt, 'continuation': 'Bonjour.', 'model': name}
    monkeypatch.setattr(server, 'complete_text', fake_complete)
    assert chat({'messages': HELLO})['model'] == 'competition'
    assert seen == [('competition', 256)]


def test_status_with_trained_chatbot_reads_json_only(model_dir):
    status = summary()
    assert status['models']['moliere_chat'] is True
    bot = status['chatbot']
    assert [p['name'] for p in bot['personas']] == ['MOLIÈRE', 'HARPAGON']
    assert bot['personas'][1]['label'] == "Harpagon (L'Avare)"
    assert bot['default_persona'] == 'MOLIÈRE' and bot['defaults'] == CONFIG and bot['training'] is None
    (model_dir / 'training_report.json').write_text(json.dumps({'steps': 10, 'val_bpb': 1.9}), encoding='utf-8')
    assert summary()['chatbot']['training'] == {'steps': 10, 'val_bpb': 1.9}
    assert FakeBot.loads == []


def test_status_falls_back_when_persona_module_is_missing(model_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, 'chatbot.format', None)
    (model_dir / 'chat_config.json').write_text(json.dumps({'personas': PERSONAS[1:]}), encoding='utf-8')
    bot = summary()['chatbot']
    assert [p['name'] for p in bot['personas']] == ['HARPAGON'] and bot['default_persona'] == 'HARPAGON'
    (model_dir / 'chat_config.json').write_text('{illisible', encoding='utf-8')
    assert summary()['chatbot']['personas'] == []


def test_completion_models_keep_their_behaviour(model_dir):
    """Le test existant de reproductibilité passe toujours, sans toucher au chatbot."""
    previous.test_explicit_seed_is_reported_and_reproduces_a_generation()
    assert FakeBot.loads == []


def post(base, payload):
    return urlopen(Request(base + '/api/chat', data=json.dumps(payload).encode(),
                           headers={'Content-Type': 'application/json'}))


def test_http_chatbot_route_status_and_errors(model_dir):
    with local_server() as base:
        with urlopen(base + '/api/status') as response:
            status = json.load(response)
        assert status['models']['moliere_chat'] and status['chatbot']['default_persona'] == 'MOLIÈRE'
        with post(base, {'model': 'moliere_chat', 'persona': 'HARPAGON', 'messages': HELLO,
                         'seed': 9, 'candidates': 3, 'max_new_tokens': 64, 'temperature': 0.7}) as response:
            result = json.load(response)
        assert set(result) == REPLY_KEYS and result['persona'] == 'HARPAGON' and result['seed'] == 9
        for extra, words in (({'persona': 'TARTUFFE'}, 'Personnage inconnu'), ({'candidates': 33}, 'candidats')):
            with pytest.raises(HTTPError) as error:
                post(base, {'model': 'moliere_chat', 'messages': HELLO, **extra})
            assert error.value.code == 400 and words in json.load(error.value)['error']
        with urlopen(base + '/') as response:
            html = response.read().decode()
        assert 'id="chat-persona"' in html and 'id="chat-model"' in html and 'Candidats comparés' in html
        assert "'chat-v4'" in html and 'innerHTML' not in html


@pytest.mark.skipif(not (server.ROOT / 'data/models/chatbot_moliere/ckpt.pt').is_file(),
                    reason='Chatbot Molière pas encore entraîné (data/models/chatbot_moliere/ckpt.pt).')
def test_real_chatbot_replies_and_reproduces_with_seed():
    server.chatbot_engine.cache_clear()
    status = summary()['chatbot']
    payload = {'model': 'moliere_chat', 'messages': HELLO, 'seed': 7, 'candidates': 2, 'max_new_tokens': 16}
    first, second = chat(payload), chat(payload)
    assert set(first) == REPLY_KEYS and isinstance(first['reply'], str)
    assert first['persona'] == status['default_persona'] and first['context_limit'] == 512
    assert first['seed'] == 7 and first['reply'] == second['reply']
    assert 1 <= first['tokens_generated'] <= 16


# --- Corrections des revues du 12/11 (revue_donnees_demo.json et revue_moteur.json) -------------

def test_chatbot_parameters_send_only_what_the_request_contains():
    assert server.chatbot_parameters({}) == {'seed': None}
    assert server.chatbot_parameters({'temperature': 0.6, 'max_new_tokens': 80}) == \
        {'seed': None, 'temperature': 0.6, 'max_new_tokens': 80}
    assert server.chatbot_parameters({'temperature': 1})['temperature'] == 1.0


def test_exported_temperature_and_length_reach_the_engine(model_dir):
    rewrite(model_dir / 'chat_config.json', json.dumps(dict(CONFIG, temperature=0.6, max_new_tokens=80)))
    chat({'model': 'moliere_chat', 'messages': HELLO})
    assert (last_call()['temperature'], last_call()['max_new_tokens']) == (0.6, 80)


def test_engine_is_reloaded_when_model_files_change(model_dir):
    chat({'model': 'moliere_chat', 'messages': HELLO})
    chat({'model': 'moliere_chat', 'messages': HELLO})
    assert len(FakeBot.loads) == 1
    rewrite(model_dir / 'chat_config.json', json.dumps(dict(CONFIG, candidates=3)))
    result = chat({'model': 'moliere_chat', 'messages': HELLO})  # sans cache_clear
    assert len(FakeBot.loads) == 2 and result['candidates'] == 3 and last_call()['candidates'] == 3


def test_default_persona_comes_from_the_loaded_engine(model_dir, monkeypatch):
    rewrite(model_dir / 'chat_config.json', json.dumps(dict(CONFIG, default_persona='HARPAGON')))

    def no_status():
        raise AssertionError('le personnage par défaut ne doit plus être relu dans /api/status')
    monkeypatch.setattr(server, 'chatbot_status', no_status)
    assert chat({'model': 'moliere_chat', 'messages': HELLO})['persona'] == 'HARPAGON'


@pytest.mark.parametrize('missing', ['ckpt.pt', 'tokenizer.json', 'chat_config.json'])
def test_incomplete_model_folder_is_not_announced(model_dir, monkeypatch, missing):
    (model_dir / missing).unlink()
    assert not server.chatbot_ready()
    status = summary()
    assert status['models']['moliere_chat'] is False and status['chatbot'] is None
    monkeypatch.setattr(server, 'complete_text',
                        lambda name, prompt, payload, max_tokens=128: {'prompt': prompt, 'continuation': 'Oui.'})
    assert chat({'messages': HELLO})['reply'] == 'Oui.'  # repli sur competition
    assert FakeBot.loads == []


def test_loading_error_is_a_clean_server_error(model_dir):
    (model_dir / 'chat_config.json').write_text('{x', encoding='utf-8')
    with pytest.raises(server.ChatbotUnavailable) as error:
        chat({'model': 'moliere_chat', 'messages': HELLO})
    assert 'Expecting' not in str(error.value) and 'pas pu être chargé' in str(error.value)
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            post(base, {'model': 'moliere_chat', 'messages': HELLO})
        body = json.load(error.value)
        assert error.value.code == 503 and 'Expecting' not in body['error'] and 'chatbot Molière' in body['error']
    rewrite(model_dir / 'chat_config.json', json.dumps(CONFIG))  # réparé : rechargé sans redémarrage
    assert chat({'model': 'moliere_chat', 'messages': HELLO})['reply'] == 'Fort bien, Monsieur.'


@pytest.mark.parametrize('model', ['moliere_chat', 'competition', 'moliere'])
def test_huge_integer_temperature_is_a_400_not_a_500(model_dir, model):
    with pytest.raises(ValueError):
        chat({'model': model, 'messages': HELLO, 'temperature': int('9' * 400)})
    with pytest.raises(ValueError):
        generate({'model': 'competition', 'prompt': 'Bonjour', 'temperature': int('9' * 400)})
    for x in (int('9' * 400), float('nan'), float('inf')):
        with pytest.raises(ValueError):
            autograd({'x': x})
    with local_server() as base:
        raw = ('{"model": "%s", "temperature": %s, "messages": [{"role": "user", "content": "Bonjour"}]}'
               % (model, '1' + '0' * 400)).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/chat', data=raw, headers={'Content-Type': 'application/json'}))
        assert error.value.code == 400 and 'température' in json.load(error.value)['error']
    assert FakeBot.loads == []


@pytest.mark.parametrize('model', ['moliere_chat', 'competition'])
def test_lone_surrogate_is_rejected_before_the_tokenizer(model_dir, model):
    with pytest.raises(ValueError, match='demi-paire'):
        chat({'model': model, 'messages': [{'role': 'user', 'content': 'Bon\ud800jour'}]})
    history = [{'role': 'user', 'content': 'Bonjour'}, {'role': 'assistant', 'content': '\udc00'},
               {'role': 'user', 'content': 'Et alors ?'}]
    with pytest.raises(ValueError, match='demi-paire'):
        chat({'model': model, 'messages': history})
    for action, key in ((generate, 'prompt'), (tokenize, 'text')):
        with pytest.raises(ValueError, match='demi-paire'):
            action({'model': 'competition', key: 'Bon\ud800jour'})
    with local_server() as base:
        with pytest.raises(HTTPError) as error:  # json.dumps écrit « \ud800 » tel quel
            post(base, {'model': model, 'messages': [{'role': 'user', 'content': 'Bon\ud800jour'}]})
        assert error.value.code == 400 and 'demi-paire' in json.load(error.value)['error']
    assert FakeBot.loads == []


@pytest.mark.parametrize('content', ['<|endoftext|>', '\u200b', ' <|endoftext|> \n\n '])
def test_message_empty_for_the_engine_is_rejected_by_the_server(model_dir, content):
    with pytest.raises(ValueError, match='texte visible'):
        chat({'model': 'moliere_chat', 'messages': [{'role': 'user', 'content': content}]})
    assert FakeBot.loads == []
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            post(base, {'model': 'moliere_chat', 'messages': [{'role': 'user', 'content': content}]})
        assert error.value.code == 400 and 'texte visible' in json.load(error.value)['error']


def test_empty_check_falls_back_when_format_module_is_missing(model_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, 'chatbot.format', None)
    assert server.chat_text_cleaner() is str.strip
    message = [{'role': 'user', 'content': '<|endoftext|>'}]
    assert server.checked_messages(message, same_as_engine=True) == message
    with pytest.raises(ValueError):
        server.checked_messages([{'role': 'user', 'content': ' '}], same_as_engine=True)


def test_completion_models_keep_the_old_check_of_empty_messages():
    """competition et moliere : même règle qu'avant (strip), « <|endoftext|> » reste accepté."""
    message = [{'role': 'user', 'content': '<|endoftext|>'}]
    assert server.checked_messages(message) == message


# Sorties produites par le serveur AVANT ces corrections (même machine, CPU), figées ici :
# à graine égale, competition et moliere doivent donner exactement le même texte.
FROZEN = [
    ('chat', 'competition', {'seed': 2547, 'max_new_tokens': 4}, 'Un chaise', 4),
    ('chat', 'competition', {'seed': 2547, 'max_new_tokens': 6, 'temperature': 1.2}, 'Un chaise\nCap', 6),
    ('chat', 'moliere', {'seed': 2547, 'max_new_tokens': 4}, 'Villez,', 4),
    ('chat', 'moliere', {'seed': 2547, 'max_new_tokens': 6, 'temperature': 1.2}, 'Villez les ann', 6),
    ('generate', 'competition', {'seed': 11, 'max_new_tokens': 5}, ' C’est une affaire de', 5),
    ('generate', 'moliere', {'seed': 11, 'max_new_tokens': 5}, " Ne m'en dis", 5),
]


@pytest.mark.parametrize('action,model,settings,text,count', FROZEN)
def test_completion_models_give_the_same_text_as_before(action, model, settings, text, count):
    weights = server.ROOT / ('minigpt/results/model.pt' if model == 'moliere' else 'submissions/2026-11-11_153839/ckpt.pt')
    if not weights.is_file():
        pytest.skip('Poids absents.')
    if action == 'chat':
        result = chat({'model': model, 'messages': [{'role': 'user', 'content': 'Une chaise'}], **settings})
        assert (result['reply'], result['tokens_generated']) == (text, count)
    else:
        result = generate({'model': model, 'prompt': 'SGANARELLE :', **settings})
        assert (result['continuation'], result['tokens_generated']) == (text, count)


def raw_post(base, body, length=None):
    """Envoie des octets bruts (corps volontairement mal formé)."""
    headers = {'Content-Type': 'application/json'}
    if length is not None:
        headers['Content-Length'] = length
    return urlopen(Request(base + '/api/chat', data=body, headers=headers))


BAD_JSON = 'Le corps de la requête n’est pas un JSON valide.'


@pytest.mark.parametrize('body', [
    b'[' * 20000 + b']' * 20000,                     # 40 Ko imbriqués : RecursionError
    b'{"model":',                                     # JSON coupé
    b'{"model": "\xff\xfe"}',                         # octets non UTF-8
    b'{"temperature": 1' + b'0' * 5000 + b'}',        # entier de 5 001 chiffres
])
def test_unreadable_body_is_a_french_400(body):
    """Corps illisible : 400 avec un message français fixe, jamais 500 ni texte brut de Python."""
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            raw_post(base, body)
        assert error.value.code == 400 and json.load(error.value)['error'] == BAD_JSON


def test_invalid_content_length_is_a_french_400():
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            raw_post(base, b'{}', length='abc')
        assert error.value.code == 400
        assert json.load(error.value)['error'] == 'En-tête Content-Length invalide.'


def socket_post(base, body, length, close_write):
    """POST brut par socket : Content-Length annoncé plus grand que le corps envoyé."""
    host, port = base.removeprefix('http://').split(':')
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(b'POST /api/chat HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: application/json\r\n'
                     b'Content-Length: %d\r\n\r\n' % (host.encode(), int(port), length) + body)
        if close_write:
            sock.shutdown(socket.SHUT_WR)  # le client a fini d'envoyer : corps tronqué
        answer = b''
        while chunk := sock.recv(4096):
            answer += chunk
    head, _, data = answer.partition(b'\r\n\r\n')
    return int(head.split()[1]), json.loads(data)['error']


def test_short_body_gets_a_clean_408_or_400(monkeypatch):
    """Corps de 7 octets pour un Content-Length de 500 : plus de 500 « Le calcul a échoué »."""
    monkeypatch.setattr(server.Handler, 'body_timeout', 0.5)
    with local_server() as base:
        started = time.monotonic()
        assert socket_post(base, b'{"a":1}', 500, close_write=False) == (
            408, 'Corps de la requête incomplet ou trop lent.')
        assert time.monotonic() - started < 5
        assert socket_post(base, b'{"a":1}', 500, close_write=True) == (400, BAD_JSON)


def test_page_fixes_are_present():
    """Contrôles statiques de la page (le comportement est vérifié dans le navigateur)."""
    html = (server.ROOT / 'demo/index.html').read_text(encoding='utf-8')
    assert 'innerHTML' not in html and 'insertAdjacentHTML' not in html
    assert "if(!c.messages.length)c.model='competition'" in html  # repli au démarrage : conversations vides
    assert "again.disabled=Boolean(pending)||!available(chat.model)" in html
    assert 'runGeneration(chat,previous)' in html and 'chat.messages.push(previous)' in html
    # Rechargement pendant une régénération : réponse remplacée gardée à part, puis relue validée.
    assert 'if(previous)chat.pendingPrevious=previous;save();' in html
    assert 'finally{delete chat.pendingPrevious;' in html
    assert "cleanMessage(item.pendingPrevious,'assistant',item.model)" in html
    assert 'id="model-warning"' in html and "$('model-warning').hidden=!info||available(chat.model)" in html
    assert 'Le serveur de démo ne répond pas' in html and 'response.json().catch(()=>({}))' in html
    assert "rawSeed===''?NaN" in html
    assert 'if(settings.chatTemperature!==null)body.temperature' in html
    assert 'if(settings.chatLength!==null)body.max_new_tokens' in html
    assert 'info?.chatbots?.[model]' in html  # chat-v4 : réglages du chatbot de la conversation
    # Démarrage : serveur arrêté ou /api/status illisible → message français, pas « Failed to fetch ».
    assert "catch{throw Error('Le serveur de démo ne répond pas." in html
    assert "throw Error('Réponse du serveur illisible.')" in html
    # Suppression de la dernière conversation : pas de recréation sur un modèle indisponible.
    assert 'newChat(available(chat.model)?chat.model:undefined)' in html and 'newChat(chat.model)' not in html


SUBMISSION_FILES = [server.SUBMISSION / 'ckpt.pt', server.SUBMISSION / 'tokenizer.json']


@pytest.mark.skipif(not all(path.is_file() for path in SUBMISSION_FILES),
                    reason='Poids de la soumission absents.')
def test_real_engine_on_a_temporary_model_folder(monkeypatch, tmp_path):
    """Vrai chatbot.generate.ChatBot (CPU) sur les poids de la soumission : chargement, reprise
    à graine égale, rechargement après réexport et erreurs propres, sans faux moteur."""
    pytest.importorskip('chatbot.generate')
    folder = tmp_path / 'chatbot_moliere'
    folder.mkdir()
    for path in SUBMISSION_FILES:
        (folder / path.name).symlink_to(path)
    rewrite(folder / 'chat_config.json', '{}')
    monkeypatch.setitem(server.CHATBOTS['moliere_chat'], 'dir', folder)
    server.chatbot_engine.cache_clear()
    try:
        payload = {'model': 'moliere_chat', 'messages': HELLO, 'seed': 7, 'candidates': 2, 'max_new_tokens': 4}
        first, second = chat(payload), chat(payload)
        assert set(first) == REPLY_KEYS and first['seed'] == 7 and first['reply'] == second['reply']
        assert first['persona'] in [p['name'] for p in summary()['chatbot']['personas']]
        assert 1 <= first['tokens_generated'] <= 4 and first['candidates'] == 2
        rewrite(folder / 'chat_config.json', json.dumps({'candidates': 3, 'max_new_tokens': 4}))
        assert chat({'model': 'moliere_chat', 'messages': HELLO, 'seed': 7})['candidates'] == 3
        rewrite(folder / 'chat_config.json', '{x')
        with local_server() as base:
            with pytest.raises(HTTPError) as error:
                post(base, payload)
            assert error.value.code == 503 and 'Expecting' not in json.load(error.value)['error']
        (folder / 'tokenizer.json').unlink()
        assert summary()['models']['moliere_chat'] is False
    finally:
        server.chatbot_engine.cache_clear()


def test_candidate_ceiling_is_32_and_message_follows_constant(model_dir):
    from demo import server
    assert server.MAX_CANDIDATES == 32
    assert chat({'model': 'moliere_chat', 'messages': HELLO, 'candidates': 32})['candidates'] == 32
    with pytest.raises(ValueError, match='entre 1 et 32'):
        chat({'model': 'moliere_chat', 'messages': HELLO, 'candidates': 33})
    assert server.summary()['chat_limits']['candidates'] == 32


def test_anchor_mu_is_returned_and_shown_in_details(model_dir):
    # Valeur 0 (modèle sans ancrage) puis valeur > 0 (S03 : anchor_mu 1.0 dans chat_config.json).
    result = chat({'model': 'moliere_chat', 'messages': HELLO, 'seed': 3})
    assert 'anchor_mu' in result and result['anchor_mu'] == 0.0
    rewrite(model_dir / 'chat_config.json', json.dumps(dict(CONFIG, anchor_mu=1.0)))
    assert chat({'model': 'moliere_chat', 'messages': HELLO, 'seed': 3})['anchor_mu'] == 1.0
    # L'affichage de la ligne « choix ancré » est testé en vrai JavaScript : tests/test_demo_page_js.py.
    html = (Path(__file__).resolve().parents[1] / 'demo/index.html').read_text(encoding='utf-8')
    assert 'info?.chat_limits?.candidates' in html and "$('candidates').max=String(maxCandidates())" in html
    assert 'chatbot-moliere-20261116' in html and 'docs/chatbot/FICHE_MODELE.md' in html
    assert 'modèle français' not in html and 'français courant' not in html


S03_REPORT = {'base': {'tokens_vus': 2000027648, 'pas': 30518, 'n_params': 54749440},
              'etape1': {'pas': 2747}, 'part_synthetique': {'tokens_synthetiques': 2720768}}


def test_status_names_the_published_release_only_for_the_published_model(model_dir):
    assert summary()['chatbot']['published_release'] is None  # pas de rapport
    rewrite(model_dir / 'training_report.json', json.dumps(S03_REPORT))
    assert summary()['chatbot']['published_release'] == 'chatbot-moliere-20261116'
    assert summary()['chatbots']['moliere_chat']['published_release'] == 'chatbot-moliere-20261116'
    for path, value in (('base.n_params', 29600000), ('base.tokens_vus', 400000000), ('etape1.pas', 2000),
                        ('part_synthetique.tokens_synthetiques', 897109)):
        other = json.loads(json.dumps(S03_REPORT))
        head, key = path.split('.')
        other[head][key] = value
        rewrite(model_dir / 'training_report.json', json.dumps(other))
        assert summary()['chatbot']['published_release'] is None, path
    rewrite(model_dir / 'training_report.json', json.dumps({'base': None}))
    assert summary()['chatbot']['published_release'] is None
    rewrite(model_dir / 'training_report.json', '[1, 2]')
    assert summary()['chatbot']['published_release'] is None


def test_real_installed_model_report_is_recognised_as_published_when_it_is_s03():
    report = server.optional_json(server.CHATBOTS['moliere_chat']['dir'] / 'training_report.json')
    if report is None:
        pytest.skip('modèle Molière non installé')
    assert server.published_release(report) in (None, 'chatbot-moliere-20261116')


def raw_status(base, host):
    """GET /api/status avec un en-tête Host choisi ; renvoie (code, corps JSON)."""
    import http.client
    address = base.removeprefix('http://')
    connection = http.client.HTTPConnection(address, timeout=5)
    try:
        connection.putrequest('GET', '/api/status', skip_host=True)
        if host is not None:
            connection.putheader('Host', host)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b'{}')
    finally:
        connection.close()


def test_get_refuses_a_foreign_host_header(model_dir):
    """Détournement de nom de domaine (DNS rebinding) : un Host étranger ne lit pas /api/status."""
    with local_server() as base:
        port = base.rsplit(':', 1)[1]
        for host in (f'evil.example:{port}', 'evil.example', f'127.0.0.1.evil.example:{port}',
                     f'localhost:{int(port) + 1}', '127.0.0.1', ''):
            code, body = raw_status(base, host)
            assert code == 403 and body == {'error': 'Cette démonstration est locale.'}, host
        for host in (f'127.0.0.1:{port}', f'localhost:{port}'):
            code, body = raw_status(base, host)
            assert code == 200 and body['ui_version'] == 'chat-v4', host


def test_post_refuses_a_foreign_host_header_before_any_work(model_dir):
    with local_server() as base:
        port = base.rsplit(':', 1)[1]
        request = Request(base + '/api/chat', data=json.dumps({'model': 'moliere_chat', 'messages': HELLO}).encode(),
                          headers={'Content-Type': 'application/json', 'Host': f'evil.example:{port}'})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403 and json.load(error.value)['error'] == 'Cette démonstration est locale.'
        assert FakeBot.loads == []  # le moteur n'a même pas été chargé
        request = Request(base + '/api/chat', data=json.dumps({'model': 'moliere_chat', 'messages': HELLO}).encode(),
                          headers={'Content-Type': 'application/json', 'Host': f'localhost:{port}'})
        assert json.load(urlopen(request))['model'] == 'moliere_chat'


@pytest.mark.skipif(not (server.SUBMISSION / 'tokenizer.json').is_file(), reason='tokenizer de la soumission absent')
def test_real_engine_accepts_the_servers_candidate_ceiling():
    """Le plafond du serveur (MAX_CANDIDATES) est accepté par la validation du vrai moteur, et un de plus est refusé."""
    torch = pytest.importorskip('torch')
    from chatbot import finetune as finetune_module
    from chatbot import generate as engine
    from ftb.model import build_model
    tiny = {'vocab_size': 8192, 'context_length': 64, 'd_model': 32, 'n_layer': 2, 'n_head': 2,
            'mlp_ratio': 4, 'dropout': 0.0, 'tie_embeddings': True}
    torch.manual_seed(0)
    bot = engine.ChatBot(build_model(tiny), finetune_module.open_tokenizer(server.SUBMISSION / 'tokenizer.json'),
                         {'max_new_tokens': 4, 'candidates': 3})
    result = bot.reply(HELLO, seed=1, candidates=server.MAX_CANDIDATES, max_new_tokens=4)
    assert result['candidates'] == server.MAX_CANDIDATES
    with pytest.raises(ValueError, match='hors bornes'):
        bot.reply(HELLO, seed=1, candidates=server.MAX_CANDIDATES + 1, max_new_tokens=4)


def test_engine_refusal_reaches_the_client_as_a_clear_french_message(model_dir, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise ValueError('max_new_tokens ou candidates hors bornes')
    monkeypatch.setattr(FakeBot, 'reply', refuse)
    with local_server() as base:
        with pytest.raises(HTTPError) as error:
            post(base, {'model': 'moliere_chat', 'messages': HELLO, 'candidates': 5})
        message = json.load(error.value)['error']
    assert error.value.code == 400 and 'refusé ces réglages' in message and 'hors bornes' not in message
    assert 'candidates' not in message and 'max_new_tokens' not in message
    # Une autre erreur du moteur n'est pas réécrite.
    monkeypatch.setattr(FakeBot, 'reply', lambda self, *a, **k: (_ for _ in ()).throw(ValueError('Autre erreur.')))
    with pytest.raises(ValueError, match='Autre erreur'):
        chat({'model': 'moliere_chat', 'messages': HELLO})
