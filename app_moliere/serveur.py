"""Serveur local de l'application Molière (127.0.0.1 seulement).

Il réutilise le serveur de la démo (`demo.server`) : mêmes contrôles HTTP (Origin local, taille,
délai, JSON) et même moteur de chat. On y ajoute la page, l'état du modèle et son téléchargement.
"""
import io
import logging
import logging.handlers
import sys
import threading
import traceback
from http.server import ThreadingHTTPServer

import demo.server as demo
from app_moliere import VERSION, modele

log = logging.getLogger('moliere')
MAX_CARACTERES_MESSAGE = 2000  # la démo l'écrit en dur (demo.server.checked_messages) ; un test les compare
ARRET = threading.Event()      # posé par « Quitter » (mode navigateur seulement)
MODE_NAVIGATEUR = False        # vrai quand l'application s'ouvre dans le navigateur (pas de fenêtre native)
ETATS = ('absent', 'telechargement', 'verification', 'chargement', 'pret', 'erreur')


class FluxJournal(io.TextIOBase):
    """Flux de remplacement pour sys.stderr/sys.stdout : écrit dans le journal, jamais dans un fichier fermé."""

    encoding = 'utf-8'

    def __init__(self, niveau):
        self.niveau = niveau
        self._local = threading.local()  # garde de réentrance, une par fil

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):  # comportement standard d'un flux sans descripteur
        raise io.UnsupportedOperation('fileno')

    def write(self, texte):
        # Si l'écriture du journal échoue, logging écrit l'erreur sur sys.stderr, donc ici : sans cette garde,
        # on rebouclerait sans fin. Dans ce cas on jette le texte.
        if getattr(self._local, 'actif', False):
            return len(texte)
        self._local.actif = True
        try:
            for ligne in str(texte).splitlines():
                if ligne.strip():
                    log.log(self.niveau, ligne.rstrip())
        finally:
            self._local.actif = False
        return len(texte)

    def flush(self):
        pass


def installer_journal():
    """Journal dans un fichier : une application fenêtrée n'a ni sortie standard ni erreur standard."""
    dossier = modele.dossier_donnees()
    try:
        dossier.mkdir(parents=True, exist_ok=True)
        gestionnaire = logging.handlers.RotatingFileHandler(
            dossier / 'journal.log', maxBytes=500_000, backupCount=2, encoding='utf-8')
    except OSError:
        gestionnaire = logging.NullHandler()
    gestionnaire.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    for ancien in log.handlers:
        ancien.close()
    log.handlers[:] = [gestionnaire]
    log.setLevel(logging.INFO)
    log.propagate = False
    logging.raiseExceptions = False  # une panne du journal ne doit jamais s'écrire sur le journal lui-même
    # Les flux absents (application fenêtrée) vont dans le journal. On ne leur donne jamais le flux du
    # gestionnaire : après une rotation du fichier, ce flux serait fermé (ValueError au premier warning).
    if sys.stderr is None or modele.est_gele():
        sys.stderr = FluxJournal(logging.WARNING)
    if sys.stdout is None:
        sys.stdout = FluxJournal(logging.INFO)


class Gestionnaire:
    """État du modèle (absent, téléchargement, vérification, chargement, prêt, erreur) et ses transitions."""

    def __init__(self):
        self.verrou = threading.Lock()
        self.annuler = threading.Event()
        self.etat = 'absent'
        self.recus = 0
        self.total = modele.TAILLE_ZIP
        self.message = ''
        self.dossier = None
        self.fil = None
        self.forcer = False
        self.nettoyer_restes()
        dossier = modele.chercher_modele()
        if dossier is not None:
            self.charger_en_fond(dossier)

    def nettoyer_restes(self):
        """Au démarrage : supprime les restes d'un téléchargement interrompu (sous le verrou, motifs précis)."""
        try:
            with modele.verrou_donnees():
                retires = modele.nettoyer_orphelins()
            if retires:
                log.info('restes supprimés au démarrage : %s', ', '.join(retires))
        except (modele.ErreurModele, OSError) as exc:  # autre fenêtre qui télécharge, dossier illisible : on laisse
            log.info('nettoyage au démarrage ignoré : %s', exc)

    def instantane(self):
        with self.verrou:
            return {'etat': self.etat, 'recus': self.recus, 'total': self.total, 'message': self.message}

    def _poser(self, etat, message=''):
        with self.verrou:
            self.etat, self.message = etat, message

    def charger_en_fond(self, dossier):
        """Pointe la démo sur le modèle et le précharge, pour que la première réponse soit rapide."""
        self.dossier = dossier
        demo.CHATBOTS['moliere_chat']['dir'] = dossier
        self._poser('chargement')
        self.fil = threading.Thread(target=self._charger, args=(dossier,), daemon=True)
        self.fil.start()

    def _charger(self, dossier):
        try:
            with demo.GENERATION_LOCK:
                demo.chatbot_engine(dossier, demo.chatbot_signature('moliere_chat'))
            log.info('modèle chargé depuis %s', dossier)
            self._poser('pret')
        except Exception:
            log.error('chargement impossible : %s', traceback.format_exc())
            self._poser('erreur', f'Le modèle trouvé dans « {dossier} » n’a pas pu être chargé. '
                                  '« Réessayer » le télécharge à nouveau (environ 240 Mo). '
                                  f'Détails dans le journal : {journal()}')

    def telecharger(self):
        """Lance le téléchargement dans un fil. Sans effet s'il est déjà en cours ou si le modèle est prêt."""
        with self.verrou:
            if self.etat in ('telechargement', 'verification', 'chargement', 'pret'):
                return
            # Après une panne de chargement, « Réessayer » doit vraiment retélécharger.
            self.forcer = self.etat == 'erreur' and self.dossier is not None
            self.etat, self.message, self.recus = 'telechargement', '', 0
            self.annuler.clear()
        self.fil = threading.Thread(target=self._telecharger, daemon=True)
        self.fil.start()

    def _progression(self, recus, total):
        with self.verrou:
            self.recus = recus
            self.total = total or modele.TAILLE_ZIP

    def _telecharger(self):
        dest = modele.dossier_modele_installe()
        while True:
            try:
                log.info('téléchargement du modèle vers %s', dest)
                # Sous le verrou, on regarde une dernière fois si une autre fenêtre vient d'installer le modèle.
                deja = None if self.forcer else (lambda: modele.chercher_modele() is not None)
                if modele.telecharger_et_installer(dest, self._progression, self.annuler,
                                                   etape=lambda e: self._poser(e), url=modele.URL,
                                                   deja_installe=deja) is None:
                    trouve = modele.chercher_modele()
                    if trouve is not None:
                        self.charger_en_fond(trouve)
                        return
            except modele.Annule:
                log.info('téléchargement annulé')
                self._poser('absent')
                return
            except modele.AutreInstance:
                # Une autre fenêtre télécharge : on attend qu'elle ait fini, puis on cherche le modèle.
                log.info('une autre fenêtre télécharge déjà le modèle : attente')
                self._poser('telechargement', modele.MSG_AUTRE_FENETRE)
                trouve = None
                while not self.annuler.wait(1.0):
                    trouve = modele.chercher_modele()
                    if trouve is not None or modele.verrou_libre():
                        break
                if self.annuler.is_set():
                    self._poser('absent')
                    return
                if trouve is not None:
                    self.charger_en_fond(trouve)
                    return
                self._poser('telechargement')  # le verrou est libre et rien n'est installé : on réessaie
                continue
            except modele.ErreurModele as exc:
                log.warning('téléchargement échoué : %s', exc)
                self._poser('erreur', str(exc))
                return
            except Exception:
                log.error('erreur inattendue : %s', traceback.format_exc())
                self._poser('erreur', 'Une erreur inattendue est survenue pendant le téléchargement. '
                                      f'Détails dans le journal : {journal()}')
                return
            break
        self.charger_en_fond(dest)

    def annuler_telechargement(self):
        if self.etat in ('telechargement', 'verification'):
            self.annuler.set()

    def fermer(self, delai=2.0):
        """À la fermeture de la fenêtre : demande l'annulation puis attend le fil au plus `delai` secondes."""
        self.annuler.set()
        fil = self.fil
        if fil is not None and fil.is_alive() and fil is not threading.current_thread():
            fil.join(delai)


def journal():
    return modele.dossier_donnees() / 'journal.log'


GESTION = None  # créé par creer_serveur() ; les actions POST (fonctions de payload) le lisent


def etat():
    """Contenu de GET /api/etat."""
    instant = GESTION.instantane()
    dossier = GESTION.dossier if instant['etat'] in ('chargement', 'pret') else None
    rapport = demo.optional_json(dossier / 'training_report.json') if dossier else None
    config = demo.optional_json(dossier / 'chat_config.json') if dossier else None
    chiffres = None
    if isinstance(rapport, dict):
        base = rapport.get('base') or {}
        panel = (rapport.get('evaluation') or {}).get('panel_commun') or {}
        chiffres = {'parametres': base.get('n_params'), 'tokens_vus': base.get('tokens_vus'),
                    'bpb_validation': base.get('meilleur_bpb_validation'),
                    'secondes_par_reponse': panel.get('seconds_per_reply')}
    config = config if isinstance(config, dict) else {}
    try:
        from chatbot.format import PERSONAS
        personas = [{'name': p['name'], 'label': p['label'], 'about': p.get('about', '')} for p in PERSONAS]
    except Exception:
        personas = demo.persona_list(config)
    return {**instant, 'version': VERSION, 'personas': personas,
            'persona_defaut': config.get('default_persona') or 'MOLIÈRE',
            'reglages': {k: config.get(k) for k in ('temperature', 'max_new_tokens', 'candidates')},
            'limites': {'messages': demo.MAX_CHAT_MESSAGES, 'caracteres_message': MAX_CARACTERES_MESSAGE,
                        'caracteres_total': demo.MAX_CHAT_CHARACTERS},
            'chiffres': chiffres, 'taille_telechargement': modele.TAILLE_ZIP, 'source': modele.URL,
            'journal': str(journal()), 'mode_navigateur': MODE_NAVIGATEUR}


def action_chat(payload):
    if GESTION.instantane()['etat'] != 'pret':
        raise ValueError('Le modèle n’est pas encore prêt : attendez la fin du chargement.')
    demo.CHATBOTS['moliere_chat']['dir'] = GESTION.dossier
    return demo.chatbot_chat(payload, 'moliere_chat')


def action_telecharger(payload):
    GESTION.telecharger()
    return etat()


def action_annuler(payload):
    GESTION.annuler_telechargement()
    return etat()


def action_quitter(payload):
    """Seulement quand l'application est ouverte dans le navigateur : sans fenêtre, rien d'autre ne la ferme."""
    if not MODE_NAVIGATEUR:
        raise ValueError('Cette action n’est disponible que dans le navigateur.')
    ARRET.set()
    return {'ok': True}


class Handler(demo.Handler):
    """Même surface que la démo, mais seulement la page, l'état et trois actions."""
    actions = {'chat': action_chat, 'telecharger': action_telecharger, 'annuler': action_annuler,
               'quitter': action_quitter}

    def log_message(self, format, *args):  # la version de base écrit sur sys.stderr, qui peut valoir None
        if args and isinstance(args[0], str) and args[0].startswith('GET /api/etat'):
            return  # le sondage de la page (toutes les 4 s) remplirait le journal
        log.info('%s %s', self.address_string(), format % args)

    def hote_local(self):
        port = self.server.server_port
        return self.headers.get('Host') in (None, f'127.0.0.1:{port}', f'localhost:{port}')

    def do_POST(self):
        if not self.hote_local():  # protège contre le rebinding DNS, aussi pour les POST
            self.reply(403, {'error': 'Cette application est locale.'})
            return
        super().do_POST()

    def do_GET(self):
        if not self.hote_local():
            self.reply(403, {'error': 'Cette application est locale.'})
            return
        chemin = self.path.split('?', 1)[0]
        ressources = modele.dossier_ressources()
        if chemin in ('/', '/index.html'):
            self.reply(200, (ressources / 'page.html').read_bytes(), 'text/html; charset=utf-8')
        elif chemin == '/icone.png':
            self.reply(200, (ressources / 'ressources' / 'icone.png').read_bytes(), 'image/png')
        elif chemin == '/api/etat':
            self.reply(200, etat())
        else:
            self.reply(404, {'error': 'Page inconnue.'})


class Serveur(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        log.error('erreur de requête de %s : %s', client_address, traceback.format_exc())


def creer_serveur(port=0):
    """Serveur lié à 127.0.0.1 ; port=0 : le système choisit un port libre."""
    global GESTION
    GESTION = Gestionnaire()
    return Serveur(('127.0.0.1', port), Handler)
