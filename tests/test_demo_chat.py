"""Conversation réelle, conservation du contexte et bornes de l'interface locale."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.server import chat, chatbot_ready, clean_chat_reply, conversation_prompt
from test_demo import local_server


def test_context_preserves_turn_order():
    prompt = conversation_prompt([
        {'role': 'user', 'content': 'Un premier message.'},
        {'role': 'assistant', 'content': 'Une première réponse.'},
        {'role': 'user', 'content': 'Une nouvelle question.'},
    ])
    assert prompt.index('Un premier message.') < prompt.index('Une première réponse.') < prompt.index('Une nouvelle question.')
    assert prompt.endswith('Assistant :\n')


@pytest.mark.parametrize('messages', [
    None, [], {},
    [{'role': 'system', 'content': 'Instruction'}],
    [{'role': 'assistant', 'content': 'Réponse sans question'}],
    [{'role': 'user', 'content': 'Bonjour'}, {'role': 'assistant', 'content': 'Bonsoir'}],
    [{'role': 'user', 'content': ' '}],
    [{'role': 'user', 'content': 12}],
    [{'role': 'user', 'content': 'a' * 2001}],
    [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'a' * 700} for i in range(19)],
    [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'Bonjour'} for i in range(21)],
])
def test_invalid_conversations_are_rejected(messages):
    with pytest.raises(ValueError):
        conversation_prompt(messages)


def test_reply_does_not_invent_another_user_turn():
    assert clean_chat_reply('Assistant : Bonjour.\nUtilisateur : Question inventée') == 'Bonjour.'
    assert clean_chat_reply('Un texte.<|endoftext|>Une autre page') == 'Un texte.'
    assert clean_chat_reply('SGANARELLE.\nUne réplique.') == 'SGANARELLE.\nUne réplique.'


def test_real_model_uses_bounded_conversation_context():
    result = chat({'model': 'competition', 'max_new_tokens': 4, 'messages': [
        {'role': 'user', 'content': 'bonjour ' * 240},
        {'role': 'assistant', 'content': 'une réponse ' * 150},
        {'role': 'user', 'content': 'Peux-tu continuer ?'},
    ]})
    assert isinstance(result['reply'], str)
    assert 1 <= result['tokens_generated'] <= 4
    assert result['history_messages'] == 3
    assert result['context_truncated']
    assert result['context_tokens'] == result['context_limit'] == 512
    assert 'prompt' not in result


def test_chat_http_and_workshop_are_available():
    with local_server() as base:
        with urlopen(base + '/atelier') as response:
            assert 'Retour au chat' in response.read().decode()
        with urlopen(base + '/') as response:
            html = response.read().decode()
            assert 'Ton message' in html and 'sessionStorage' in html
        payload = json.dumps({'max_new_tokens': 4, 'messages': [{'role': 'user', 'content': 'Bonjour'}]}).encode()
        with urlopen(Request(base + '/api/chat', data=payload, headers={'Content-Type': 'application/json'})) as response:
            result = json.load(response)
            # Sans modèle précisé : le chatbot s'il est entraîné, sinon le modèle français.
            expected = 'moliere_chat' if chatbot_ready() else 'competition'
            assert result['model'] == expected and result['history_messages'] == 1
        for extra in ({'max_new_tokens': 257}, {'temperature': float('nan')}):
            data = json.dumps({'messages': [{'role': 'user', 'content': 'Bonjour'}], **extra}).encode()
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + '/api/chat', data=data, headers={'Content-Type': 'application/json'}))
            assert error.value.code == 400


def test_explicit_seed_is_reported_and_reproduces_a_generation():
    payload = {'model': 'competition', 'max_new_tokens': 4, 'seed': 2547,
               'messages': [{'role': 'user', 'content': 'Une chaise'}]}
    first = chat(payload)
    second = chat(payload)
    assert first['seed'] == second['seed'] == 2547
    assert first['reply'] == second['reply']
