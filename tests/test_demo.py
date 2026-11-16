"""Vrais calculs et limites de la démo locale, sans modifier les modèles entraînés."""
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from demo.server import Handler, autograd, clean, generate, summary, tokenize


@pytest.mark.parametrize('value', [-2, -1, 0, 3])
def test_shared_operand_gradient(value):
    result = autograd({'x': value})
    assert result['y'] == (value + 1) ** 2
    assert result['gradient'] == 2 * value + 2
    assert result['agreement']


@pytest.mark.parametrize('name', ['moliere', 'competition'])
def test_real_tokenizers_preserve_utf8(name):
    text = 'L’été à Noël ☕\nÉlève, 42 000 ! <|endoftext|>'
    result = tokenize({'model': name, 'text': text})
    assert result['decoded'] == text
    assert result['roundtrip'] and result['tokens'] == len(result['ids'])
    assert result['bytes'] == len(text.encode('utf-8'))


def test_masking_preserves_dates_and_population():
    result = clean({'text': 'Le musÃ©e de 1876 reçoit 42 000 visiteurs. Contact : contact@example.org. Téléphone : 01 00 00 00 00.'})
    assert 'musée' in result['masked']
    assert '1876' in result['masked'] and '42 000' in result['masked']
    assert 'contact@example.org' not in result['masked']
    assert '01 00 00 00 00' not in result['masked']
    assert result['masks'] == {'EMAIL': 1, 'TELEPHONE': 1}
    assert result['same_normalized_hash']


@pytest.mark.parametrize('name,prompt', [('moliere', 'SGANARELLE :'), ('competition', 'La ville est située')])
def test_real_trained_models_generate(name, prompt):
    if name == 'moliere' and not Path('minigpt/results/model.pt').is_file():
        pytest.skip('Intégration Molière : recréer les poids avec minigpt.train_moliere.')
    result = generate({'model': name, 'prompt': prompt, 'max_new_tokens': 8})
    assert 1 <= result['tokens_generated'] <= 8
    assert isinstance(result['continuation'], str)
    assert result['device'] == 'cpu'


@contextmanager
def local_server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_server_rejects_remote_origin_and_path_traversal():
    with local_server() as base:
        request = Request(base + '/api/generate', data=b'{}', headers={
            'Content-Type': 'application/json', 'Origin': 'https://example.org'})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(base + '/../../AGENTS.md')
        assert error.value.code == 404


def test_server_bounds_generation_and_request_size():
    with local_server() as base:
        for count in (0, 129, True):
            payload = json.dumps({'prompt': 'Bonjour', 'max_new_tokens': count}).encode()
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + '/api/generate', data=payload,
                                headers={'Content-Type': 'application/json'}))
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/tokenize', data=b'x' * 16385,
                            headers={'Content-Type': 'application/json'}))
        assert error.value.code == 413


def test_displayed_numbers_come_from_submitted_metadata():
    status = summary()
    assert status['dataset']['n_documents'] == 72612
    assert status['dataset']['n_tokens'] >= status['final']['tokens_seen']
    assert status['ablation'][1]['bpb'] < status['ablation'][0]['bpb']
    assert status['models']['competition']
    assert status['ui_version'] == 'chat-v4'
    assert set(status['models']) == {'competition', 'moliere', 'moliere_chat', 'francais_chat'}
    assert set(status['chatbots']) == {'moliere_chat', 'francais_chat'}
