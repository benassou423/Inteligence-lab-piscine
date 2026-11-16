"""Démo locale : uv run python -m demo --port 8766. Aucun fichier d'entraînement modifié."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / 'submissions/2026-11-11_153839'
# Les deux chatbots : identifiant de modèle → dossier exporté et nom utilisé dans les messages.
# Un dossier absent = chatbot non installé (les deux sont entraînés sur la RTX 5090 et exportés ici).
# Une petite table plutôt qu'un copier-coller : le même code sert les deux chatbots.
CHATBOTS = {
    'moliere_chat': {'dir': ROOT / 'data/models/chatbot_moliere', 'name': 'Le chatbot Molière'},
    'francais_chat': {'dir': ROOT / 'data/models/chatbot_francais',
                      'name': 'Le chatbot en français moderne'},
}
GENERATION_LOCK = threading.Lock()
MAX_CHAT_MESSAGES = 19
MAX_CHAT_CHARACTERS = 12000
MAX_CANDIDATES = 32  # le réglage par défaut du chatbot Molière (S03) est 32
# Le modèle publié « S03 » (release GitHub) : aucun identifiant n'est écrit dans le dossier installé, donc
# on le reconnaît à cinq valeurs de son training_report.json (tokens et pas du pré-entraînement, pas de
# l'affinage, paramètres, tokens de synthèse). Une empreinte de lecture, pas une preuve cryptographique.
PUBLISHED_RELEASE = 'chatbot-moliere-20261116'
PUBLISHED_FINGERPRINT = (2000027648, 30518, 2747, 54749440, 2720768)
# Les trois fichiers écrits par chatbot.generate.export_model (chat_config.json en dernier).
CHATBOT_FILES = ('ckpt.pt', 'tokenizer.json', 'chat_config.json')


class ChatbotUnavailable(Exception):
    """Panne côté serveur (modèle incomplet ou illisible) : 503, jamais présentée comme une faute du client."""


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def optional_json(path):
    """Lire un JSON s'il existe ; None si le fichier manque ou est illisible."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def chatbot_dir(model='moliere_chat'):
    """Dossier du chatbot, relu dans la table à chaque appel (les tests le remplacent)."""
    return CHATBOTS[model]['dir']


def chatbot_ready(model='moliere_chat'):
    """Prêt seulement si les trois fichiers existent : un export à moitié écrit n'est pas annoncé."""
    return all((chatbot_dir(model) / name).is_file() for name in CHATBOT_FILES)


def chatbot_signature(model='moliere_chat'):
    """Date de modification et taille des trois fichiers : elles changent à chaque nouvel export."""
    signature = []
    for name in CHATBOT_FILES:
        try:
            info = (chatbot_dir(model) / name).stat()
            signature.append((name, info.st_mtime_ns, info.st_size))
        except OSError:
            signature.append((name, None, None))
    return tuple(signature)


def persona_list(config):
    """Personnages lus dans chatbot.format (import paresseux), sinon dans chat_config.json."""
    try:
        from chatbot.format import PERSONAS
    except Exception:
        PERSONAS = config.get('personas', [])
    items = list(PERSONAS.values()) if isinstance(PERSONAS, dict) else PERSONAS
    personas = []
    for item in items if isinstance(items, (list, tuple)) else []:
        if isinstance(item, dict) and isinstance(item.get('name'), str):
            personas.append({'name': item['name'], 'label': str(item.get('label') or item['name']),
                             'interlocutor': item.get('interlocutor'), 'about': item.get('about', '')})
    return personas


def published_release(training):
    """Étiquette de release si le rapport d'entraînement est celui du modèle publié S03, sinon None."""
    try:
        found = (training['base']['tokens_vus'], training['base']['pas'], training['etape1']['pas'],
                 training['base']['n_params'], training['part_synthetique']['tokens_synthetiques'])
    except (KeyError, TypeError):
        return None
    return PUBLISHED_RELEASE if found == PUBLISHED_FINGERPRINT else None


def chatbot_status(model='moliere_chat'):
    """Résumé d'un chatbot pour /api/status : lit seulement les JSON, jamais les poids."""
    if not chatbot_ready(model):
        return None
    folder = chatbot_dir(model)
    config = optional_json(folder / 'chat_config.json')
    config = config if isinstance(config, dict) else {}
    # Format « tirets » (français moderne) : deux interlocuteurs sans nom, donc aucun personnage.
    style = config.get('format', 'repliques')
    personas = persona_list(config) if style == 'repliques' else []
    default = config.get('default_persona') or (personas[0]['name'] if personas else None)
    training = optional_json(folder / 'training_report.json')
    return {'format': style, 'personas': personas, 'default_persona': default if style == 'repliques' else None,
            'defaults': config, 'training': training,
            # Release publiée si ce dossier contient le modèle S03 (la page n'affiche ses chiffres de jury que dans ce cas).
            'published_release': published_release(training) if model == 'moliere_chat' else None}


def summary():
    """Les chiffres affichés viennent des résultats du projet, pas de valeurs simulées."""
    mini = read_json('minigpt/results/metrics.json')
    final = read_json('experiments/submission_20261111.json')
    prepared = read_json('submissions/2026-11-11_153839/data_meta.json')
    ablation = read_json('experiments/ablation_complete_20261110.json')['average']
    return {
        'ui_version': 'chat-v4',
        'mini': {k: mini[k] for k in ('n_params', 'steps', 'vocab_size', 'bpb_val_transformer',
                                      'bpb_val_bigram_tokens', 'history')},
        'final': final,
        'dataset': {k: prepared[k] for k in ('n_documents', 'n_tokens', 'n_bytes')},
        'ablation': [{'recipe': k, 'bpb': ablation[k]['bpb']}
                     for k in ('brut', 'nettoye', 'complet')],
        'models': {'moliere': (ROOT / 'minigpt/results/model.pt').is_file(),
                   'competition': (SUBMISSION / 'ckpt.pt').is_file(),
                   **{name: chatbot_ready(name) for name in CHATBOTS}},
        # « chatbot » : bloc du chatbot Molière, gardé tel quel pour les anciennes pages et tests.
        'chatbot': chatbot_status('moliere_chat'),
        'chatbots': {name: chatbot_status(name) for name in CHATBOTS},
        'chat_limits': {'messages': MAX_CHAT_MESSAGES, 'characters': MAX_CHAT_CHARACTERS,
                        'max_new_tokens': 256, 'candidates': MAX_CANDIDATES},
    }


def ensure_utf8(value):
    """Refuser une demi-paire UTF-16 isolée (« \\ud800 ») : JSON l'accepte, le tokenizer plante."""
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError('Le texte contient un caractère invalide (demi-paire UTF-16).') from None
    return value


def checked_text(payload, key='text'):
    value = payload.get(key, '')
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError('Saisir un texte de 1 à 2 000 caractères.')
    return ensure_utf8(value)


def model_name(payload, default='moliere'):
    name = payload.get('model', default)
    if name not in ('moliere', 'competition'):
        raise ValueError('Choisir un des deux modèles proposés.')
    return name


@lru_cache(maxsize=2)
def tokenizer(name):
    if name == 'moliere':
        from minigpt.bpe import BPETokenizer
        return BPETokenizer().load(ROOT / 'minigpt/results/tokenizer.json')
    from ftb.tokenizer import load_tokenizer
    return load_tokenizer(SUBMISSION / 'tokenizer.json')


def autograd(payload):
    from minigpt.micrograd import Value
    import torch
    value = payload.get('x', 3)
    # Comparaison en chaîne : rejette NaN et ±inf, et un entier immense sans conversion en float.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -100 <= value <= 100:
        raise ValueError('Choisir x entre -100 et 100.')
    x = Value(float(value))
    y = x * x + 2 * x + 1
    y.backward()
    reference = torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
    expected = reference * reference + 2 * reference + 1
    expected.backward()
    return {'x': value, 'y': y.data, 'gradient': x.grad,
            'pytorch_gradient': reference.grad.item(),
            'agreement': math.isclose(x.grad, reference.grad.item(), abs_tol=1e-10)}


def tokenize(payload):
    text = checked_text(payload)
    name = model_name(payload)
    tok = tokenizer(name)
    ids = tok.encode(text)
    if name == 'moliere':
        pieces = [repr(tok.vocab[i]) for i in ids]
        size = len(tok.vocab)
    else:
        pieces = [tok.tk.id_to_token(i) for i in ids]
        size = tok.vocab_size
    return {'ids': ids, 'pieces': pieces, 'decoded': tok.decode(ids),
            'roundtrip': tok.decode(ids) == text, 'bytes': len(text.encode('utf-8')),
            'tokens': len(ids), 'vocab_size': size}


def clean(payload):
    from pipeline.clean.encoding import clean_text
    from pipeline.clean.privacy import mask_text
    from pipeline.dedup.exact import text_key
    text = checked_text(payload)
    repaired = clean_text(text)
    masked, counts = mask_text(repaired, source='commoncrawl')
    duplicate = '  ' + masked.upper().replace(' ', '   ') + '\n'
    return {'repaired': repaired, 'masked': masked, 'masks': dict(counts),
            'normalized_hash': text_key(masked).hex(),
            'duplicate': duplicate,
            'same_normalized_hash': text_key(masked) == text_key(duplicate)}


@lru_cache(maxsize=2)
def trained_model(name):
    import torch
    torch.set_num_threads(4)
    if name == 'moliere':
        from minigpt.gpt import GPT, GPTConfig
        path = ROOT / 'minigpt/results/model.pt'
        if not path.is_file():
            raise ValueError('Le modèle Molière doit être recréé avec la commande du guide.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = GPT(GPTConfig(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        from ftb.model import load_checkpoint
        model, checkpoint = load_checkpoint(SUBMISSION / 'ckpt.pt', map_location='cpu')
    return model.eval()


def valid_temperature(value):
    """Nombre entre 0,1 et 1,5. Pas de math.isfinite : sur un entier de 400 chiffres, il lève
    OverflowError (erreur 500) ; la comparaison en chaîne rejette aussi NaN et l'infini."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0.1 <= value <= 1.5


def sampling_parameters(payload, max_tokens=128):
    count = payload.get('max_new_tokens', 64)
    temperature = payload.get('temperature', 0.8)
    seed = payload.get('seed', 1337)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= max_tokens:
        raise ValueError(f'Choisir entre 1 et {max_tokens} tokens à ajouter.')
    if not valid_temperature(temperature):
        raise ValueError('Choisir une température entre 0,1 et 1,5.')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('La graine doit être un entier entre 0 et 4 294 967 295.')
    return count, temperature, seed


def complete_text(name, prompt, payload, max_tokens=128):
    """Même modèle et mêmes poids ; seul le contexte fourni varie entre texte et chat."""
    import torch
    count, temperature, seed = sampling_parameters(payload, max_tokens)
    start = time.monotonic()
    # Une génération à la fois : graine et calcul restent isolés des autres clics.
    with GENERATION_LOCK, torch.inference_mode():
        model = trained_model(name)
        tok = tokenizer(name)
        all_ids = tok.encode(prompt)
        context_limit = model.config.block_size if name == 'moliere' else model.context_length
        ids = all_ids[-context_limit:]
        torch.manual_seed(seed)
        arguments = {'temperature': temperature, 'top_k': 40}
        if name == 'competition':
            arguments['eot_id'] = tok.eot_id
        output = model.generate(torch.tensor([ids], dtype=torch.long), count, **arguments)[0].tolist()
        new_ids = output[len(ids):]
        visible_ids = new_ids[:-1] if name == 'competition' and new_ids and new_ids[-1] == tok.eot_id else new_ids
    return {'prompt': prompt, 'continuation': tok.decode(visible_ids),
            'tokens_generated': len(new_ids), 'seconds': round(time.monotonic() - start, 2),
            'device': 'cpu', 'model': name, 'seed': seed,
            'context_tokens': len(ids), 'context_limit': context_limit,
            'context_truncated': len(all_ids) > context_limit}


def generate(payload):
    return complete_text(model_name(payload), checked_text(payload, 'prompt'), payload)


def chat_text_cleaner():
    """clean_text du chatbot (import paresseux) ; repli sur str.strip si le module manque."""
    try:
        from chatbot.format import clean_text as clean_text_chat
        return clean_text_chat
    except Exception:
        return str.strip


def checked_messages(messages, same_as_engine=False):
    """Valider les tours (alternance, nombre, tailles) ; renvoie une copie réduite à role/content.

    same_as_engine=True (chatbot Molière) : un message utilisateur est « vide » avec la même règle
    que le moteur (chatbot.format.clean_text), par exemple « <|endoftext|> » seul."""
    cleaner = chat_text_cleaner() if same_as_engine else str.strip
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_CHAT_MESSAGES or len(messages) % 2 == 0:
        raise ValueError('La conversation doit finir par un message utilisateur et contenir au plus 19 messages.')
    checked = []
    total = 0
    for index, message in enumerate(messages):
        expected_role = 'user' if index % 2 == 0 else 'assistant'
        if not isinstance(message, dict) or message.get('role') != expected_role:
            raise ValueError('Les messages doivent alterner utilisateur et assistant.')
        content = message.get('content')
        if not isinstance(content, str) or len(content) > 2000 or (expected_role == 'user' and not content.strip()):
            raise ValueError('Chaque message utilisateur doit contenir de 1 à 2 000 caractères.')
        ensure_utf8(content)
        if expected_role == 'user' and not cleaner(content):
            raise ValueError('Chaque message utilisateur doit contenir du texte visible.')
        total += len(content)
        if total > MAX_CHAT_CHARACTERS:
            raise ValueError('Le contexte envoyé dépasse 12 000 caractères.')
        checked.append({'role': expected_role, 'content': content})
    return checked


def conversation_prompt(messages):
    """Valider les tours et construire un contexte borné, sans historique côté serveur."""
    parts = ['Conversation en français.']
    for message in checked_messages(messages):
        label = 'Utilisateur' if message['role'] == 'user' else 'Assistant'
        parts.append(f"{label} :\n{message['content']}")
    parts.append('Assistant :\n')
    return '\n\n'.join(parts)


def clean_chat_reply(text):
    """Arrêter l'affichage avant un nouveau rôle que le modèle aurait inventé."""
    text = re.sub(r'^\s*Assistant\s*:\s*', '', text, count=1, flags=re.I)
    text = text.split('<|endoftext|>', 1)[0]
    text = re.split(r'(?m)^\s*(?:Utilisateur|User|Humain|Assistant)\s*:', text, maxsplit=1, flags=re.I)[0]
    return text.strip()


# Un moteur chargé par dossier : {dossier: (signature des fichiers, moteur)}.
ENGINES = {}


def chatbot_engine(model_dir, signature):
    """Charger une seule fois chaque chatbot ; import paresseux pour que la démo démarre sans lui.

    Passer d'un chatbot à l'autre ne recharge rien (un moteur par dossier). Après un nouvel
    export, la signature change : l'ancien moteur de CE dossier est libéré puis le nouveau chargé.
    Une erreur n'est jamais mise en cache : la requête suivante réessaie."""
    cached = ENGINES.get(model_dir)
    if cached is not None and cached[0] == signature:
        return cached[1]
    ENGINES.pop(model_dir, None)  # libérer la mémoire de l'ancien modèle avant de charger le nouveau
    import torch
    torch.set_num_threads(4)
    from chatbot.generate import ChatBot
    bot = ChatBot.load(model_dir, device='cpu')
    ENGINES[model_dir] = (signature, bot)
    return bot


# Même nom que l'ancien cache (lru_cache) : vider tous les moteurs chargés.
chatbot_engine.cache_clear = ENGINES.clear


def chatbot_parameters(payload):
    """Réglages facultatifs du chatbot ; un champ absent laisse la valeur de chat_config.json."""
    options = {}
    if 'temperature' in payload:
        if not valid_temperature(payload['temperature']):
            raise ValueError('Choisir une température entre 0,1 et 1,5.')
        options['temperature'] = float(payload['temperature'])  # sans risque : déjà bornée
    seed = payload.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise ValueError('La graine doit être un entier entre 0 et 4 294 967 295.')
    options['seed'] = seed  # None : le moteur tire une graine et la renvoie
    if 'max_new_tokens' in payload:
        count = payload['max_new_tokens']
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
            raise ValueError('Choisir entre 1 et 256 tokens à ajouter.')
        # Le moteur garde au moins la moitié des 512 tokens pour le contexte : 255 au plus.
        options['max_new_tokens'] = min(count, 255)
    if payload.get('candidates') is not None:
        candidates = payload['candidates']
        if isinstance(candidates, bool) or not isinstance(candidates, int) or not 1 <= candidates <= MAX_CANDIDATES:
            raise ValueError(f'Choisir entre 1 et {MAX_CANDIDATES} candidats à comparer.')
        options['candidates'] = candidates
    return options


def chatbot_chat(payload, model='moliere_chat'):
    """Un chatbot répond au dernier message : Molière dans la voix du personnage choisi,
    le français moderne sans personnage (deux interlocuteurs sans nom, format « tirets »)."""
    title = CHATBOTS[model]['name']
    messages = checked_messages(payload.get('messages'), same_as_engine=True)
    options = chatbot_parameters(payload)
    persona = payload.get('persona')
    if persona is not None and not isinstance(persona, str):
        raise ValueError('Le personnage doit être désigné par son nom.')
    if not chatbot_ready(model):
        raise ValueError(f'{title} n’est pas encore entraîné : choisir un autre modèle.')
    with GENERATION_LOCK:
        try:
            bot = chatbot_engine(chatbot_dir(model), chatbot_signature(model))
        except Exception as exc:
            # Fichier manquant, JSON illisible, tokenizer différent : panne du serveur, pas du client.
            raise ChatbotUnavailable(f'{title} n’a pas pu être chargé (modèle incomplet, '
                                     'illisible ou en cours d’export). Réessayer ou choisir un autre modèle.') from exc
        names = [p['name'] for p in bot.personas]
        if not names:
            # Pas de personnage (format « tirets ») : un nom envoyé est refusé plutôt qu'ignoré
            # en silence, pour que le client sache que le champ ne sert à rien.
            if persona is not None:
                raise ValueError('Ce chatbot n’a pas de personnage : ne pas envoyer de champ « persona ».')
        elif persona is None:
            # Même source que le moteur : la configuration qu'il a chargée.
            default = getattr(bot, 'config', {}).get('default_persona')
            persona = default if default in names else (names[0] if names else None)
        if names and persona not in names:
            raise ValueError('Personnage inconnu : choisir un des personnages proposés.')
        try:
            out = bot.reply(messages, persona=persona, **options)
        except ValueError as exc:
            # Le moteur a ses propres bornes (longueur, candidats) : message clair, pas son texte brut.
            if 'hors bornes' in str(exc):
                raise ValueError('Le modèle a refusé ces réglages (longueur de réponse ou nombre de candidats hors '
                                 'de ses limites). Reviens aux valeurs du modèle dans les réglages.') from None
            raise
    keys = ('reply', 'persona', 'interlocutor', 'seed', 'tokens_generated', 'seconds', 'context_tokens',
            'context_limit', 'context_truncated', 'history_turns_used', 'candidates', 'anchor_mu', 'finish')
    result = {key: out.get(key) for key in keys}
    result.update(history_messages=len(messages), model=model, device='cpu')
    return result


def chat(payload):
    name = payload.get('model', 'moliere_chat' if chatbot_ready() else 'competition')
    if name not in ('moliere_chat', 'francais_chat', 'competition', 'moliere'):
        raise ValueError('Choisir un des quatre modèles proposés.')
    if name in CHATBOTS:
        return chatbot_chat(payload, name)
    # Complétion (« deuxième test ») : comportement inchangé pour competition et moliere.
    messages = payload.get('messages')
    prompt = conversation_prompt(messages)
    result = complete_text(name, prompt, payload, max_tokens=256)
    result.pop('prompt')
    result['reply'] = clean_chat_reply(result.pop('continuation'))
    result['history_messages'] = len(messages)
    return result


ACTIONS = {'autograd': autograd, 'tokenize': tokenize, 'clean': clean, 'generate': generate, 'chat': chat}


class Handler(BaseHTTPRequestHandler):
    """Surface HTTP limitée : ni chemin libre, ni commande, ni écriture de données."""
    body_timeout = 10  # secondes accordées pour recevoir le corps (réduit dans les tests)
    actions = ACTIONS  # table des actions POST : une sous-classe (application Molière) la remplace
    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def host_refused(self):
        """Refuser un en-tête Host étranger (détournement de nom de domaine) : 403 et True.
        Un client sans Host (HTTP/1.0) passe : un navigateur en envoie toujours un."""
        port = self.server.server_port
        if self.headers.get('Host') not in (None, f'127.0.0.1:{port}', f'localhost:{port}'):
            self.reply(403, {'error': 'Cette démonstration est locale.'})
            return True
        return False

    def do_GET(self):
        if self.host_refused():
            return
        if self.path == '/':
            self.reply(200, (ROOT / 'demo/index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path in ('/atelier', '/atelier/'):
            self.reply(200, (ROOT / 'demo/atelier.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/status':
            self.reply(200, summary())
        elif self.path == '/curve.png':
            self.reply(200, (ROOT / 'minigpt/results/loss_curve.png').read_bytes(), 'image/png')
        else:
            self.reply(404, {'error': 'Page inconnue.'})

    def do_POST(self):
        port = self.server.server_port
        if self.host_refused():
            return
        if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
            self.reply(403, {'error': 'Cette démonstration est locale.'})
            return
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self.reply(415, {'error': 'Un objet JSON est attendu.'})
            return
        try:
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise ValueError('En-tête Content-Length invalide.') from None
            limit = 65536 if self.path == '/api/chat' else 16384
            if not 0 < length <= limit:
                self.reply(413, {'error': 'Requête trop volumineuse.'})
                return
            self.connection.settimeout(self.body_timeout)
            # Corps annoncé plus long que celui reçu : soit le client attend (délai dépassé,
            # TimeoutError est un OSError) → 408, soit il a fermé l'envoi → corps tronqué → 400.
            try:
                body = self.rfile.read(length)
            except OSError:
                self.reply(408, {'error': 'Corps de la requête incomplet ou trop lent.'})
                self.close_connection = True
                return
            # Corps illisible (JSON invalide, octets non UTF-8, imbrication trop profonde,
            # entier géant, corps tronqué) : message français unique, sans détail interne.
            try:
                if len(body) < length:
                    raise ValueError
                payload = json.loads(body)
            except (ValueError, RecursionError):
                raise ValueError('Le corps de la requête n’est pas un JSON valide.') from None
            if not isinstance(payload, dict):
                raise ValueError('Un objet JSON est attendu.')
            action = self.actions.get(self.path.removeprefix('/api/')) if self.path.startswith('/api/') else None
            if action is None:
                self.reply(404, {'error': 'Action inconnue.'})
                return
            self.reply(200, action(payload))
        except ChatbotUnavailable as exc:
            self.reply(503, {'error': str(exc)})
            raise  # la cause exacte part dans le journal local
        except (ValueError, UnicodeError) as exc:
            self.reply(400, {'error': str(exc)})
        except Exception:
            self.reply(500, {'error': 'Le calcul a échoué. Consulter le journal local de la démo.'})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Démo disponible : http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
