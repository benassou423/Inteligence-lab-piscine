"""Application Molière : recherche, téléchargement et extraction sûre du modèle, serveur local.

Rapide, sans réseau externe ni vrai modèle : un petit serveur HTTP local sert un faux zip,
et un faux moteur remplace chatbot.generate.ChatBot."""
import errno
import hashlib
import http.client
import io
import json
import logging
import os
import re
import shutil
import ssl
import sys
import threading
import time
import types
import urllib.error
import warnings
import zipfile
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import demo.server as demo
from app_moliere import modele, serveur

CONFIG = {'default_persona': 'MOLIÈRE', 'candidates': 8, 'temperature': 0.8, 'max_new_tokens': 120}
CONTENU = {'ckpt.pt': b'faux poids ' * 50, 'tokenizer.json': b'{}',
           'chat_config.json': json.dumps(CONFIG).encode(),
           'training_report.json': json.dumps({'base': {'n_params': 123}}).encode()}


def fabriquer_zip(entrees=None, extra=None):
    """Zip en mémoire : par défaut les 4 fichiers attendus sous chatbot_moliere/."""
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, 'w') as zf:
        for nom, data in (entrees if entrees is not None else {f'chatbot_moliere/{k}': v for k, v in CONTENU.items()}).items():
            zf.writestr(nom, data)
        for info, data in (extra or []):
            zf.writestr(info, data)
    return tampon.getvalue()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def constantes(monkeypatch):
    """Les empreintes du vrai modèle sont remplacées par celles du faux zip."""
    data = fabriquer_zip()
    monkeypatch.setattr(modele, 'SHA_ZIP', sha(data))
    monkeypatch.setattr(modele, 'SHA_CKPT', sha(CONTENU['ckpt.pt']))
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(data))
    return data


@pytest.fixture
def donnees(monkeypatch, tmp_path):
    monkeypatch.setenv('MOLIERE_DONNEES', str(tmp_path / 'donnees'))
    monkeypatch.delenv('MOLIERE_MODELE', raising=False)
    return tmp_path / 'donnees'


def faux_modele(dossier):
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in modele.FICHIERS_REQUIS:
        (dossier / nom).write_bytes(CONTENU[nom])
    return dossier


# --- recherche du modèle --------------------------------------------------------------------------
def test_ordre_de_recherche(monkeypatch, donnees, tmp_path):
    env, embarque, installe = tmp_path / 'env', tmp_path / 'meipass' / 'modele_embarque', donnees / 'modele'
    monkeypatch.setattr(modele, 'est_gele', lambda: True)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / 'meipass'), raising=False)
    assert modele.chercher_modele() is None
    faux_modele(installe)
    assert modele.chercher_modele() == installe
    faux_modele(embarque)
    assert modele.chercher_modele() == embarque          # l'embarqué passe avant les données
    monkeypatch.setenv('MOLIERE_MODELE', str(env))
    assert modele.chercher_modele() == embarque          # variable pointant sur un dossier vide : ignorée
    faux_modele(env)
    assert modele.chercher_modele() == env               # puis la variable d'environnement passe avant tout


def test_depot_seulement_depuis_les_sources(monkeypatch, donnees):
    assert modele.emplacements()[-1][0] == 'dépôt'
    monkeypatch.setattr(modele, 'est_gele', lambda: True)
    assert all(nom != 'dépôt' for nom, _ in modele.emplacements())


def test_modele_incomplet_non_present(tmp_path):
    d = faux_modele(tmp_path / 'm')
    (d / 'tokenizer.json').unlink()
    assert not modele.modele_present(d)
    (d / 'tokenizer.json').write_bytes(b'')              # fichier vide : pas un modèle non plus
    assert not modele.modele_present(d)


# --- extraction sûre ------------------------------------------------------------------------------
def tentative(tmp_path, data, constantes_ok=True):
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    dest = tmp_path / 'sortie' / 'modele'
    with pytest.raises(modele.ErreurModele) as exc:
        modele.installer_archive(archive, dest, sha_zip=sha(data), sha_ckpt=sha(CONTENU['ckpt.pt']))
    assert not dest.exists()                              # rien d'installé après un refus
    assert [p.name for p in dest.parent.iterdir()] == []  # ni dossier temporaire oublié
    return str(exc.value)


def test_chemin_dangereux_refuse(tmp_path):
    entrees = {f'chatbot_moliere/{k}': v for k, v in CONTENU.items()}
    entrees['chatbot_moliere/../evil.txt'] = b'x'
    assert 'chemin' in tentative(tmp_path, fabriquer_zip(entrees))
    entrees = {'/etc/passwd': b'x'}
    assert 'chemin' in tentative(tmp_path, fabriquer_zip(entrees))


def test_fichier_inattendu_refuse(tmp_path):
    entrees = {f'chatbot_moliere/{k}': v for k, v in CONTENU.items()}
    entrees['chatbot_moliere/autre.bin'] = b'x'
    assert 'inattendu' in tentative(tmp_path, fabriquer_zip(entrees))


def test_lien_symbolique_refuse(tmp_path):
    entrees = {f'chatbot_moliere/{k}': v for k, v in CONTENU.items() if k != 'tokenizer.json'}
    lien = zipfile.ZipInfo('chatbot_moliere/tokenizer.json')
    lien.create_system = 3
    lien.external_attr = (0o120777) << 16
    assert 'lien' in tentative(tmp_path, fabriquer_zip(entrees, extra=[(lien, b'/etc/passwd')]))


def test_taille_anormale_refusee(tmp_path, monkeypatch):
    monkeypatch.setitem(modele.TAILLE_MAX, 'tokenizer.json', 1)
    assert 'taille' in tentative(tmp_path, fabriquer_zip())


def test_fichier_requis_manquant(tmp_path):
    entrees = {f'chatbot_moliere/{k}': v for k, v in CONTENU.items() if k != 'ckpt.pt'}
    assert 'incomplète' in tentative(tmp_path, fabriquer_zip(entrees))


def test_sha_zip_faux_refuse(tmp_path):
    data = fabriquer_zip()
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    with pytest.raises(modele.ErreurModele, match='empreinte'):
        modele.installer_archive(archive, tmp_path / 'm', sha_zip='0' * 64, sha_ckpt=sha(CONTENU['ckpt.pt']))
    assert not (tmp_path / 'm').exists()


def test_sha_ckpt_faux_refuse(tmp_path):
    data = fabriquer_zip()
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    with pytest.raises(modele.ErreurModele, match='poids'):
        modele.installer_archive(archive, tmp_path / 'm', sha_zip=sha(data), sha_ckpt='0' * 64)
    assert not (tmp_path / 'm').exists()


def test_installation_et_remplacement(tmp_path):
    data = fabriquer_zip()
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    dest = tmp_path / 'modele'
    dest.mkdir()
    (dest / 'ckpt.pt').write_text('ancien')              # un vieux modèle : seuls des noms de fichiers du modèle
    modele.installer_archive(archive, dest, sha_zip=sha(data), sha_ckpt=sha(CONTENU['ckpt.pt']))
    assert modele.modele_present(dest) and (dest / 'ckpt.pt').read_bytes() == CONTENU['ckpt.pt']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.zip', 'modele']


def test_remplacer_refuse_un_dossier_etranger(tmp_path, monkeypatch, capsys):
    """Défaut 14 : `--dest ~/Documents --remplacer` ne doit jamais effacer un dossier quelconque."""
    data = fabriquer_zip()
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    dest = tmp_path / 'Documents'
    dest.mkdir()
    (dest / 'vieux.txt').write_text('precieux')
    with pytest.raises(modele.ErreurModele, match='vieux.txt'):
        modele.installer_archive(archive, dest, sha_zip=sha(data), sha_ckpt=sha(CONTENU['ckpt.pt']))
    assert (dest / 'vieux.txt').read_text() == 'precieux'
    # la commande refuse avant tout téléchargement
    monkeypatch.setattr(modele, 'telecharger', lambda *a, **k: pytest.fail('aucun téléchargement attendu'))
    assert modele.main(['--dest', str(dest), '--remplacer']) == 1
    assert 'vieux.txt' in capsys.readouterr().err and (dest / 'vieux.txt').exists()
    lien = tmp_path / 'lien'
    lien.symlink_to(dest)
    with pytest.raises(modele.ErreurModele):
        modele.verifier_remplacable(lien)


# --- téléchargement -------------------------------------------------------------------------------
@pytest.fixture
def serveur_http():
    """Petit serveur local. GET /bon.zip sert `corps` par morceaux (pause, longueur annoncée, coupure
    et absence de Content-Length réglables) ; GET /bloque.zip envoie 10 octets puis se tait."""
    etat = {'corps': b'', 'pause': 0.0, 'annonce': None, 'envoie': None, 'sans_longueur': False,
            'hits': 0, 'libere': threading.Event()}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/bloque.zip':
                self.send_response(200)
                self.send_header('Content-Length', '1000')
                self.end_headers()
                self.wfile.write(b'0123456789')
                self.wfile.flush()
                etat['libere'].wait(20)
                return
            if self.path != '/bon.zip':
                self.send_response(404)
                self.end_headers()
                return
            etat['hits'] += 1
            corps = etat['corps']
            self.send_response(200)
            if not etat['sans_longueur']:
                self.send_header('Content-Length', str(len(corps) if etat['annonce'] is None else etat['annonce']))
            self.end_headers()
            envoye = corps if etat['envoie'] is None else corps[:etat['envoie']]
            try:
                for i in range(0, len(envoye), 4096):
                    self.wfile.write(envoye[i:i + 4096])
                    if etat['pause']:
                        time.sleep(etat['pause'])
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    etat['url'] = f'http://127.0.0.1:{srv.server_port}/bon.zip'
    etat['base'] = f'http://127.0.0.1:{srv.server_port}'
    yield etat
    etat['libere'].set()
    srv.shutdown()
    srv.server_close()


def test_telechargement_progression_et_installation(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes
    suivi, etapes = [], []
    dest = tmp_path / 'd' / 'modele'
    modele.telecharger_et_installer(dest, lambda r, t: suivi.append((r, t)), etape=etapes.append, url=serveur_http['url'])
    assert modele.modele_present(dest)
    assert suivi[0] == (0, len(constantes)) and suivi[-1] == (len(constantes), len(constantes))
    assert etapes == ['verification']
    assert not list(dest.parent.glob('*.part')) and not (dest.parent / modele.NOM_ZIP).exists()


def test_telechargement_annule(constantes, serveur_http, tmp_path, monkeypatch):
    serveur_http['corps'] = constantes * 20
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(constantes) * 20)
    serveur_http['pause'] = 0.01
    stop = threading.Event()
    dest = tmp_path / 'd' / 'modele'
    with pytest.raises(modele.Annule):
        modele.telecharger_et_installer(dest, lambda r, t: stop.set() if r > 10000 else None, stop, url=serveur_http['url'])
    assert not dest.exists() and [x.name for x in dest.parent.iterdir() if x.name != '.verrou'] == []


def test_erreur_http_et_serveur_absent(constantes, serveur_http, tmp_path):
    with pytest.raises(modele.ErreurModele, match='404'):
        modele.telecharger(serveur_http['base'] + '/absent.zip', tmp_path / 'x.zip')
    assert not list(tmp_path.iterdir())
    with pytest.raises(modele.ErreurModele, match='connexion'):
        modele.telecharger('http://127.0.0.1:9/x.zip', tmp_path / 'x.zip', delai=2)


def test_archive_alteree_ecartee(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes[:-1] + bytes([constantes[-1] ^ 1])
    dest = tmp_path / 'd' / 'modele'
    with pytest.raises(modele.ErreurModele):
        modele.telecharger_et_installer(dest, url=serveur_http['url'])
    assert not dest.exists() and not (dest.parent / modele.NOM_ZIP).exists()


def test_commande_refuse_decraser(tmp_path, capsys):
    d = tmp_path / 'm'
    d.mkdir()
    assert modele.main(['--dest', str(d)]) == 2
    assert 'remplacer' in capsys.readouterr().err


def test_contexte_tls_utilise_certifi(monkeypatch):
    """Le contexte doit être créé avec cafile=certifi.where() (et pas seulement exister)."""
    certifi = pytest.importorskip('certifi')
    appels = []
    vrai = ssl.create_default_context

    def espion(*args, **kw):
        appels.append(kw)
        return vrai(*args, **kw)
    monkeypatch.setattr(modele.ssl, 'create_default_context', espion)
    ctx = modele.contexte_tls()
    assert appels == [{'cafile': certifi.where()}]
    assert ctx.verify_mode == ssl.CERT_REQUIRED


# --- téléchargement : erreurs distinguées, plafonds, annulation ----------------------------------
def test_erreur_d_ecriture_n_est_pas_une_panne_internet(constantes, serveur_http, tmp_path, monkeypatch):
    """Défaut 3 : disque plein / dossier protégé ≠ « vérifiez votre connexion »."""
    serveur_http['corps'] = constantes
    ouvrir = open

    class Plein:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, bloc):
            raise OSError(errno.ENOSPC, 'No space left on device')

    monkeypatch.setattr(modele, 'open', lambda f, *a, **k: Plein() if str(f).endswith('.part') else ouvrir(f, *a, **k),
                        raising=False)
    with pytest.raises(modele.ErreurModele) as exc:
        modele.telecharger(serveur_http['url'], tmp_path / 'x.zip')
    assert 'Écriture impossible' in str(exc.value) and 'Internet' not in str(exc.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignore les permissions')
def test_dossier_protege_ecriture_impossible(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes
    protege = tmp_path / 'ro'
    protege.mkdir()
    protege.chmod(0o500)
    try:
        with pytest.raises(modele.ErreurModele, match='Écriture impossible'):
            modele.telecharger(serveur_http['url'], protege / 'x.zip')
    finally:
        protege.chmod(0o700)


def test_content_length_different_de_la_taille_attendue(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes
    serveur_http['annonce'] = len(constantes) + 10
    with pytest.raises(modele.ErreurModele, match='annonce'):
        modele.telecharger_et_installer(tmp_path / 'd' / 'modele', url=serveur_http['url'])
    assert not list((tmp_path / 'd').glob('*.part')) and not (tmp_path / 'd' / modele.NOM_ZIP).exists()


def test_flux_coupe_trop_tot(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes
    serveur_http['envoie'] = len(constantes) // 2          # annonce la bonne taille, n'en envoie que la moitié
    with pytest.raises(modele.ErreurModele, match='trop tôt'):
        modele.telecharger_et_installer(tmp_path / 'd' / 'modele', url=serveur_http['url'])
    assert not list((tmp_path / 'd').glob('*.part'))


def test_fichier_plus_gros_que_prevu(constantes, serveur_http, tmp_path):
    serveur_http['corps'] = constantes + b'x' * 5000
    serveur_http['sans_longueur'] = True                   # pas de Content-Length : le plafond fait foi
    with pytest.raises(modele.ErreurModele, match='plus gros'):
        modele.telecharger_et_installer(tmp_path / 'd' / 'modele', url=serveur_http['url'])
    assert not list((tmp_path / 'd').glob('*.part'))


def test_espace_disque_insuffisant(constantes, serveur_http, tmp_path, monkeypatch):
    serveur_http['corps'] = constantes
    Usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(shutil, 'disk_usage', lambda d: Usage(10**9, 10**9 - 5 * 10**6, 5 * 10**6))
    with pytest.raises(modele.ErreurModele, match=r'Espace disque insuffisant : \d+ Mo nécessaires, 5 Mo') as exc:
        modele.telecharger_et_installer(tmp_path / 'd' / 'modele', url=serveur_http['url'])
    assert serveur_http['hits'] == 0, 'on vérifie la place avant de télécharger'
    assert 'Mo' in str(exc.value)


def test_annulation_pendant_une_lecture_bloquee(serveur_http, tmp_path):
    """Défaut 11 : annuler une connexion muette donne « annulé », vite, pas une erreur réseau."""
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    debut = time.time()
    with pytest.raises(modele.Annule):
        modele.telecharger(serveur_http['base'] + '/bloque.zip', tmp_path / 'x.zip', annuler=stop, delai=30)
    assert time.time() - debut < 8
    assert not list(tmp_path.iterdir())


def test_delai_depasse_avec_annulation_demandee_donne_annule(serveur_http, tmp_path):
    stop = threading.Event()
    stop.set()
    with pytest.raises(modele.Annule):                     # l'erreur réseau est traduite en « annulé »
        modele.telecharger('http://127.0.0.1:9/x.zip', tmp_path / 'x.zip', annuler=stop, delai=2)


def test_certificat_refuse_message_dedie(tmp_path, monkeypatch):
    """Défaut 12 : erreur de certificat ≠ panne de connexion."""
    def refuse(*a, **k):
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, 'certificate verify failed'))
    monkeypatch.setattr(modele.urllib.request, 'urlopen', refuse)
    with pytest.raises(modele.ErreurModele) as exc:
        modele.telecharger('https://exemple.invalid/x.zip', tmp_path / 'x.zip')
    assert 'Connexion sécurisée refusée' in str(exc.value) and 'date et l’heure' in str(exc.value)
    assert 'Internet' not in str(exc.value)


# --- verrou, nettoyage des restes ---------------------------------------------------------------
pytestmark_verrou = pytest.mark.skipif(modele.fcntl is None, reason='pas de fcntl')


@pytestmark_verrou
def test_verrou_deja_pris(tmp_path):
    with modele.verrou_donnees(tmp_path):
        assert not modele.verrou_libre(tmp_path)
        with pytest.raises(modele.AutreInstance, match='autre fenêtre'):
            with modele.verrou_donnees(tmp_path):
                pass
        with pytest.raises(modele.AutreInstance):          # et l'installation ne touche à rien
            modele.telecharger_et_installer(tmp_path / 'modele', url='http://127.0.0.1:9/x')
    assert modele.verrou_libre(tmp_path)                   # relâché à la sortie


def test_nettoyage_des_orphelins_seulement_les_motifs(tmp_path):
    d = tmp_path / 'donnees'
    d.mkdir()
    for nom in (modele.NOM_ZIP, modele.NOM_ZIP + '.part', 'journal.log', 'autre.zip', 'x.part'):
        (d / nom).write_bytes(b'1')
    (d / '.moliere-install-abc').mkdir()
    (d / '.moliere-install-abc' / 'ckpt.pt').write_bytes(b'1')
    (d / 'modele.ancien-42').mkdir()
    faux_modele(d / 'modele')
    retires = modele.nettoyer_orphelins(d)
    assert sorted(p.name for p in d.iterdir()) == ['autre.zip', 'journal.log', 'modele', 'x.part']
    assert len(retires) == 4
    # arrêt en plein remplacement : le modèle manque, un « ancien » complet existe → on le remet
    shutil.rmtree(d / 'modele')
    faux_modele(d / 'modele.ancien-7')
    modele.nettoyer_orphelins(d)
    assert modele.modele_present(d / 'modele') and not (d / 'modele.ancien-7').exists()


@pytestmark_verrou
def test_demarrage_nettoie_les_restes(appli, donnees):
    donnees.mkdir(parents=True)
    (donnees / (modele.NOM_ZIP + '.part')).write_bytes(b'1')
    (donnees / modele.NOM_ZIP).write_bytes(b'1')
    (donnees / '.moliere-install-zz').mkdir()
    appli()
    assert sorted(p.name for p in donnees.iterdir() if p.name != '.verrou') == []


@pytestmark_verrou
def test_demarrage_ne_nettoie_pas_pendant_qu_une_autre_fenetre_telecharge(appli, donnees):
    donnees.mkdir(parents=True)
    (donnees / (modele.NOM_ZIP + '.part')).write_bytes(b'1')
    with modele.verrou_donnees(donnees):                   # l'autre fenêtre
        appli()
    assert (donnees / (modele.NOM_ZIP + '.part')).exists()


@pytestmark_verrou
def test_seconde_fenetre_attend_puis_trouve_le_modele(appli, donnees):
    _, base = appli()
    with modele.verrou_donnees(donnees):                   # une autre fenêtre télécharge
        assert post(base, '/api/telecharger', {})[0] == 200
        fin = time.time() + 10
        while time.time() < fin:
            e = json.loads(get(base, '/api/etat')[1])
            if e['message'] == modele.MSG_AUTRE_FENETRE:
                break
            time.sleep(0.05)
        assert e['etat'] == 'telechargement' and e['message'] == modele.MSG_AUTRE_FENETRE
        faux_modele(donnees / 'modele')                    # l'autre fenêtre a fini
    attendre(base, 'pret', delai=20)


@pytestmark_verrou
def test_fermeture_annule_et_nettoie(appli, donnees, constantes, serveur_http, monkeypatch):
    """Défaut 5 : à la fermeture, annulation demandée puis attente du fil (2 s au plus)."""
    serveur_http['corps'] = constantes * 200
    serveur_http['pause'] = 0.05
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(constantes) * 200)
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    _, base = appli()
    post(base, '/api/telecharger', {})
    attendre(base, 'telechargement')
    debut = time.time()
    serveur.GESTION.fermer(2.0)
    assert time.time() - debut < 3
    assert not serveur.GESTION.fil.is_alive()
    assert [p.name for p in donnees.iterdir() if p.name != '.verrou'] == []


# --- journal ------------------------------------------------------------------------------------
@pytest.fixture
def journal_propre(donnees, monkeypatch):
    monkeypatch.setattr(sys, 'stderr', sys.stderr)         # mémorise pour restaurer
    monkeypatch.setattr(sys, 'stdout', sys.stdout)
    yield
    for h in serveur.log.handlers:
        h.close()
    serveur.log.handlers[:] = []
    serveur.log.propagate = True


def test_journal_apres_rotation_stderr_reste_utilisable(journal_propre, donnees, monkeypatch):
    """Défaut 6 : sys.stderr=None, rotation du fichier, puis un avertissement : aucune exception."""
    monkeypatch.setattr(sys, 'stderr', None)
    monkeypatch.setattr(sys, 'stdout', None)
    serveur.installer_journal()
    assert sys.stderr is not None and sys.stdout is not None
    assert not isinstance(sys.stderr, logging.StreamHandler)
    serveur.log.handlers[0].doRollover()                   # le flux du gestionnaire est fermé puis rouvert
    print('affichage standard', file=sys.stdout)
    sys.stderr.write('texte sur la sortie d’erreur\n')
    # ce que fait warnings.showwarning par défaut : écrire sur sys.stderr (pytest le remplace, on le fait à la main)
    sys.stderr.write(warnings.formatwarning(UserWarning('avertissement apres rotation'), UserWarning, 'f.py', 1))
    for h in serveur.log.handlers:
        h.flush()
    texte = (donnees / 'journal.log').read_text(encoding='utf-8')
    assert 'texte sur la sortie d’erreur' in texte and 'affichage standard' in texte
    assert 'avertissement apres rotation' in texte


def test_sondage_etat_non_journalise(journal_propre, appli, donnees):
    serveur.installer_journal()
    _, base = appli()
    get(base, '/api/etat')
    get(base, '/')
    for h in serveur.log.handlers:
        h.flush()
    texte = (donnees / 'journal.log').read_text(encoding='utf-8')
    assert 'GET /api/etat' not in texte and 'GET / ' in texte


# --- HTTP : Host / Origin, réponses complètes ----------------------------------------------------
def test_post_avec_hote_etranger_refuse(appli):
    _, base = appli()
    port = int(base.rsplit(':', 1)[1])
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    conn.request('POST', '/api/telecharger', body=b'{}', headers={'Host': 'evil.example', 'Content-Type': 'application/json'})
    rep = conn.getresponse()
    assert rep.status == 403
    conn.close()
    assert json.loads(get(base, '/api/etat')[1])['etat'] == 'absent'      # rien n'a été lancé


def test_origine_null_refusee(appli):
    _, base = appli()
    assert post(base, '/api/telecharger', {}, {'Origin': 'null'})[0] == 403
    assert json.loads(get(base, '/api/etat')[1])['etat'] == 'absent'


def test_reponses_post_completes_comme_etat(appli, constantes, serveur_http, monkeypatch):
    """Défaut d : /api/telecharger et /api/annuler renvoient le même objet que /api/etat (version incluse)."""
    serveur_http['corps'] = constantes * 20
    serveur_http['pause'] = 0.02
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(constantes) * 20)
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    _, base = appli()
    code, rep = post(base, '/api/telecharger', {})
    assert code == 200 and rep['version'] and len(rep['personas']) == 11 and rep['journal']
    code, rep = post(base, '/api/annuler', {})
    assert code == 200 and rep['version'] and 'limites' in rep
    attendre(base, 'absent')


def test_double_post_telecharger_simultane(appli, constantes, serveur_http, monkeypatch):
    serveur_http['corps'] = constantes * 200
    serveur_http['pause'] = 0.02
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(constantes) * 200)
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    _, base = appli()
    barriere = threading.Barrier(2)
    codes = []

    def tir():
        barriere.wait()
        codes.append(post(base, '/api/telecharger', {})[0])
    fils = [threading.Thread(target=tir) for _ in range(2)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    attendre(base, 'telechargement')
    assert codes == [200, 200]
    post(base, '/api/annuler', {})
    attendre(base, 'absent')
    assert serveur_http['hits'] == 1                       # un seul téléchargement réel


def test_etat_verification_observable(appli, constantes, serveur_http, monkeypatch):
    serveur_http['corps'] = constantes
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    vrai = modele.installer_archive

    def lent(*a, **k):
        time.sleep(0.8)
        return vrai(*a, **k)
    monkeypatch.setattr(modele, 'installer_archive', lent)
    _, base = appli()
    post(base, '/api/telecharger', {})
    vus = set()
    fin = time.time() + 15
    while time.time() < fin:
        e = json.loads(get(base, '/api/etat')[1])
        vus.add(e['etat'])
        if e['etat'] == 'pret':
            break
        time.sleep(0.02)
    assert 'verification' in vus and 'pret' in vus


def test_limites_publiees_et_alignees_sur_la_demo(appli, donnees):
    faux_modele(donnees / 'modele')
    _, base = appli()
    e = attendre(base, 'pret')
    assert e['limites'] == {'messages': demo.MAX_CHAT_MESSAGES, 'caracteres_message': 2000,
                            'caracteres_total': demo.MAX_CHAT_CHARACTERS}
    long = 'a' * (e['limites']['caracteres_message'] + 1)    # la valeur codée ici doit rester celle de la démo
    code, _ = post(base, '/api/chat', {'messages': [{'role': 'user', 'content': long}]})
    assert code == 400
    code, _ = post(base, '/api/chat', {'messages': [{'role': 'user', 'content': long[:-1]}]})
    assert code == 200
    reste = e['limites']['caracteres_total']               # le total est bien refusé au-delà
    msgs = []
    for i in range(7):
        msgs += [{'role': 'user', 'content': 'u' * 1000}, {'role': 'assistant', 'content': 'a' * 1000}]
    msgs.append({'role': 'user', 'content': 'u' * 1000})
    assert sum(len(m['content']) for m in msgs) > reste
    assert post(base, '/api/chat', {'messages': msgs})[0] == 400


def test_quitter_seulement_en_mode_navigateur(appli, monkeypatch):
    _, base = appli()
    serveur.ARRET.clear()
    code, rep = post(base, '/api/quitter', {})
    assert code == 400 and not serveur.ARRET.is_set()
    monkeypatch.setattr(serveur, 'MODE_NAVIGATEUR', True)
    assert json.loads(get(base, '/api/etat')[1])['mode_navigateur'] is True
    code, rep = post(base, '/api/quitter', {})
    assert code == 200 and serveur.ARRET.is_set()
    serveur.ARRET.clear()


def test_message_de_chargement_donne_le_chemin(appli, donnees, monkeypatch):
    faux_modele(donnees / 'modele')
    demo.chatbot_engine.cache_clear()

    class Casse(FauxBot):
        @classmethod
        def load(cls, *a, **k):
            raise RuntimeError('poids illisibles')
    sys.modules['chatbot.generate'].ChatBot = Casse
    _, base = appli()
    e = attendre(base, 'erreur')
    assert str(donnees / 'modele') in e['message'] and 'Réessayer' in e['message'] and 'journal' in e['message']
    assert 'Supprimez' not in e['message']


# --- script de construction ----------------------------------------------------------------------
def test_construction_modele_embarque_ne_pose_pas_modele_env(tmp_path, monkeypatch):
    """Défaut 8 : en mode embarqué, MOLIERE_MODELE est retiré ; sinon il passerait avant l'emplacement embarqué."""
    from tools import build_app_macos as b
    monkeypatch.setenv('MOLIERE_MODELE', '/ailleurs')
    env = b.env_autotest(tmp_path / 'm', tmp_path / 'd', embarque=str(tmp_path / 'emb'))
    assert 'MOLIERE_MODELE' not in env and env['MOLIERE_DONNEES'] == str(tmp_path / 'd')
    env = b.env_autotest(tmp_path / 'm', tmp_path / 'd', embarque=None)
    assert env['MOLIERE_MODELE'] == str((tmp_path / 'm').resolve())
    ok = {'ok': True, 'dossier_modele': '/x/Moliere.app/Contents/Frameworks/modele_embarque'}
    b.verifier_rapport_autotest(ok, 'emb')
    with pytest.raises(RuntimeError, match='embarqué'):
        b.verifier_rapport_autotest({'ok': True, 'dossier_modele': '/ailleurs/chatbot_moliere'}, 'emb')
    with pytest.raises(RuntimeError, match='échec'):
        b.verifier_rapport_autotest({'ok': False, 'erreur': 'x'}, None)
    b.verifier_rapport_autotest({'ok': True, 'dossier_modele': '/ailleurs/chatbot_moliere'}, None)


def test_construction_modele_autotest_absent_s_arrete(tmp_path):
    from tools import build_app_macos as b
    with pytest.raises(SystemExit, match='autotest-telecharge'):
        b.verifier_entrees(None, str(tmp_path / 'absent'), False)
    b.verifier_entrees(None, str(tmp_path / 'absent'), True)       # autorisé explicitement
    faux_modele(tmp_path / 'm')
    b.verifier_entrees(None, str(tmp_path / 'm'), False)
    with pytest.raises(SystemExit, match='embarquer'):
        b.verifier_entrees(str(tmp_path / 'vide'), None, False)


def test_construction_detecte_icloud(monkeypatch, tmp_path):
    from tools import build_app_macos as b
    assert b.sous_icloud(Path.home() / 'Library' / 'Mobile Documents' / 'x')
    assert b.sous_icloud(Path.home() / 'Desktop' / 'projet' / 'data')
    assert not b.sous_icloud('/private/tmp/x')


def test_construction_ordre_rapport_ecrit_en_dernier():
    """Défaut 9 : lecture du script — anciens rapport/DMG supprimés d'abord, autotest avant publication du DMG,
    rapport écrit en dernier."""
    src = (Path(__file__).resolve().parents[1] / 'tools' / 'build_app_macos.py').read_text(encoding='utf-8')
    i_suppr = src.index('for ancien in (chemin_rapport')
    i_autotest = src.index("rapport['autotest_secondes']")
    i_publie = src.index('shutil.move(dmg, final)')
    i_ecrit = src.index('chemin_rapport.write_text')
    assert i_suppr < i_autotest < i_publie < i_ecrit


# --- serveur local --------------------------------------------------------------------------------
class FauxBot:
    def __init__(self, config):
        self.config = dict(CONFIG, **config)
        from chatbot.format import PERSONAS
        self.personas = PERSONAS

    @classmethod
    def load(cls, model_dir, device='cpu'):
        return cls(json.loads((Path(model_dir) / 'chat_config.json').read_text(encoding='utf-8')))

    def reply(self, messages, persona=None, **kw):
        return {'reply': 'Fort bien, Monsieur.', 'persona': persona, 'interlocutor': 'LA GRANGE', 'seed': 1,
                'tokens_generated': 5, 'seconds': 0.01, 'context_tokens': 9, 'context_limit': 512,
                'context_truncated': False, 'history_turns_used': 1, 'candidates': 8, 'finish': 'eot'}


@pytest.fixture
def appli(monkeypatch, donnees):
    """Fabrique le serveur de l'application (faux moteur) ; renvoie une fonction de démarrage."""
    faux = types.ModuleType('chatbot.generate')
    faux.ChatBot = FauxBot
    monkeypatch.setitem(sys.modules, 'chatbot.generate', faux)
    monkeypatch.setitem(demo.CHATBOTS['moliere_chat'], 'dir', Path('/nulle/part'))
    demo.chatbot_engine.cache_clear()
    monkeypatch.setattr(modele, 'emplacements', lambda: [('données', modele.dossier_modele_installe())])
    lances = []

    def demarrer():
        srv = serveur.creer_serveur(0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        lances.append(srv)
        return srv, f'http://127.0.0.1:{srv.server_port}'
    yield demarrer
    for srv in lances:
        srv.shutdown()
        srv.server_close()
    demo.chatbot_engine.cache_clear()


def get(base, chemin, entetes=None):
    with urlopen(Request(base + chemin, headers=entetes or {}), timeout=10) as r:
        return r.status, r.read(), r.headers


def post(base, chemin, corps, entetes=None):
    h = {'Content-Type': 'application/json', **(entetes or {})}
    try:
        with urlopen(Request(base + chemin, data=json.dumps(corps).encode(), headers=h), timeout=10) as r:
            return r.status, json.loads(r.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def attendre(base, etat, delai=15):
    fin = time.time() + delai
    while time.time() < fin:
        e = json.loads(get(base, '/api/etat')[1])
        if e['etat'] == etat:
            return e
        time.sleep(0.05)
    raise AssertionError(f'état {etat} non atteint : {e}')


def test_etat_absent_puis_chat_refuse(appli):
    _, base = appli()
    e = json.loads(get(base, '/api/etat')[1])
    assert e['etat'] == 'absent' and e['total'] == modele.TAILLE_ZIP
    assert len(e['personas']) == 11 and e['persona_defaut'] == 'MOLIÈRE'
    code, rep = post(base, '/api/chat', {'messages': [{'role': 'user', 'content': 'Bonjour'}]})
    assert code == 400 and 'pas encore prêt' in rep['error']


def test_etat_pret_et_chat(appli, donnees):
    faux_modele(donnees / 'modele')
    (donnees / 'modele' / 'training_report.json').write_bytes(CONTENU['training_report.json'])
    _, base = appli()
    e = attendre(base, 'pret')
    assert e['chiffres']['parametres'] == 123 and e['version']
    code, rep = post(base, '/api/chat', {'messages': [{'role': 'user', 'content': 'Bonjour'}], 'persona': 'HARPAGON'})
    assert code == 200 and rep['reply'] == 'Fort bien, Monsieur.' and rep['persona'] == 'HARPAGON'
    code, _ = post(base, '/api/chat', {'messages': [{'role': 'assistant', 'content': 'x'}]})
    assert code == 400                                   # les contrôles de la démo s'appliquent


def test_origine_etrangere_et_hote_refuses(appli):
    _, base = appli()
    code, rep = post(base, '/api/telecharger', {}, {'Origin': 'http://evil.example'})
    assert code == 403
    with pytest.raises(HTTPError) as exc:
        get(base, '/api/etat', {'Host': 'evil.example'})
    assert exc.value.code == 403
    assert get(base, '/api/etat')[0] == 200


def test_premier_lancement_telechargement_complet(appli, constantes, serveur_http, monkeypatch, donnees):
    serveur_http['corps'] = constantes
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    _, base = appli()
    assert post(base, '/api/telecharger', {})[0] == 200
    attendre(base, 'pret')
    assert modele.modele_present(donnees / 'modele')
    code, rep = post(base, '/api/chat', {'messages': [{'role': 'user', 'content': 'Bonjour'}]})
    assert code == 200 and rep['reply']


def test_telechargement_en_erreur_puis_annulation(appli, constantes, serveur_http, monkeypatch):
    monkeypatch.setattr(modele, 'URL', serveur_http['base'] + '/absent.zip')
    _, base = appli()
    post(base, '/api/telecharger', {})
    e = attendre(base, 'erreur')
    assert '404' in e['message'] and 'Traceback' not in e['message']
    serveur_http['corps'] = constantes * 20
    monkeypatch.setattr(modele, 'TAILLE_ZIP', len(constantes) * 20)
    serveur_http['pause'] = 0.01
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    post(base, '/api/telecharger', {})                   # « Réessayer » repart de zéro
    e = attendre(base, 'telechargement')
    post(base, '/api/annuler', {})
    attendre(base, 'absent')


def test_page_sans_ressource_externe(appli):
    _, base = appli()
    code, corps, entetes = get(base, '/')
    html = corps.decode('utf-8')
    assert code == 200 and 'text/html' in entetes['Content-Type']
    sans_liens = re.sub(r'<a\s[^>]*>', '', html)
    assert not re.search(r'https?://', sans_liens, re.I)  # adresses permises seulement dans des liens <a>
    for balise in re.findall(r'<(?:script|link|img|iframe|source|video|audio)\b[^>]*>', html, re.I):
        assert 'http://' not in balise and 'https://' not in balise and '//cdn' not in balise
    assert not re.search(r'@import|url\(\s*[\'"]?https?:', html)
    assert 'innerHTML' not in html and 'insertAdjacentHTML' not in html  # texte jamais inséré comme HTML
    assert get(base, '/icone.png')[1].startswith(b'\x89PNG')


# --- corrections de la relecture finale ---------------------------------------------------------------
def test_journal_en_panne_ne_boucle_pas(journal_propre, donnees, monkeypatch):
    """Défaut 1 : écriture du journal impossible + sys.stderr remplacé par FluxJournal : la main est rendue."""
    monkeypatch.setattr(logging, 'raiseExceptions', True)   # mémorisé, restauré par monkeypatch
    monkeypatch.setattr(modele, 'est_gele', lambda: True)
    serveur.installer_journal()
    assert logging.raiseExceptions is False and isinstance(sys.stderr, serveur.FluxJournal)
    assert sys.stderr.encoding == 'utf-8' and sys.stderr.isatty() is False
    with pytest.raises(io.UnsupportedOperation):
        sys.stderr.fileno()

    def panne(*a, **k):
        raise OSError('disque plein')
    gestionnaire = serveur.log.handlers[0]
    vrai_flush = gestionnaire.flush
    gestionnaire.flush = panne
    flux = sys.stderr
    fini = threading.Event()

    def travail():
        serveur.log.info('premier message')
        print('sur stderr', file=flux)
        logging.raiseExceptions = True                      # même avec l'option par défaut : la garde suffit
        serveur.log.info('second message')
        print('encore', file=flux)
        fini.set()
    fil = threading.Thread(target=travail, daemon=True)
    fil.start()
    fil.join(2)
    gestionnaire.flush = vrai_flush                          # le nettoyage de la fixture doit pouvoir fermer
    assert fini.is_set() and not fil.is_alive()


def test_ds_store_ne_bloque_pas_reessayer(tmp_path):
    """Défaut 2 : .DS_Store et ._* créés par le Finder ne sont pas des fichiers étrangers, et disparaissent."""
    data = fabriquer_zip()
    archive = tmp_path / 'a.zip'
    archive.write_bytes(data)
    dest = tmp_path / 'modele'
    faux_modele(dest)
    (dest / '.DS_Store').write_bytes(b'x')
    (dest / '._ckpt.pt').write_bytes(b'x')
    modele.verifier_remplacable(dest)
    modele.installer_archive(archive, dest, sha_zip=sha(data), sha_ckpt=sha(CONTENU['ckpt.pt']))
    assert sorted(p.name for p in dest.iterdir()) == sorted(modele.FICHIERS)
    (dest / 'autre.txt').write_text('x')
    with pytest.raises(modele.ErreurModele, match='autre.txt'):
        modele.verifier_remplacable(dest)


@pytestmark_verrou
def test_lien_casse_ne_fait_pas_planter_le_demarrage(appli, donnees):
    """Défaut 3 : « modele » est un lien cassé à côté d'un « ancien » complet : le serveur démarre quand même."""
    donnees.mkdir(parents=True)
    (donnees / 'modele').symlink_to(donnees / 'nulle-part')
    faux_modele(donnees / 'modele.ancien-9')
    assert modele.nettoyer_orphelins(donnees) == []         # ni restauré ni supprimé
    assert (donnees / 'modele.ancien-9').exists()
    _, base = appli()
    assert json.loads(get(base, '/api/etat')[1])['etat'] in ('absent', 'erreur')


@pytestmark_verrou
def test_nettoyage_oserror_est_journalise(appli, donnees, monkeypatch):
    def casse(*a, **k):
        raise NotADirectoryError('boum')
    monkeypatch.setattr(modele, 'nettoyer_orphelins', casse)
    _, base = appli()
    assert get(base, '/api/etat')[0] == 200


def test_fermeture_de_la_fenetre_ne_renvoie_jamais_false(appli, donnees, monkeypatch):
    """Défaut 4 : a_la_fermeture appelle GESTION.fermer, ne renvoie que None ; fermer est idempotente."""
    from app_moliere import __main__ as principal
    appli()
    appelees = []
    vraie = serveur.GESTION.fermer
    monkeypatch.setattr(serveur.GESTION, 'fermer', lambda d=2.0: appelees.append(d) or vraie(d))
    assert principal.a_la_fermeture() is None and appelees == [2.0]
    assert principal.a_la_fermeture() is None              # double appel sans danger
    monkeypatch.setattr(serveur.GESTION, 'fermer', lambda d=2.0: 1 / 0)
    assert principal.a_la_fermeture() is None              # même en cas d'erreur


def test_fenetre_branche_l_evenement_closing(appli, donnees, monkeypatch):
    """Défaut 4 : main() branche a_la_fermeture sur events.closing (faux pywebview)."""
    from app_moliere import __main__ as principal
    vus = {}

    class Evenement:
        def __init__(self):
            self.items = []

        def __iadd__(self, f):
            self.items.append(f)
            return self

    faux = types.ModuleType('webview')

    def create_window(*a, **k):
        w = types.SimpleNamespace(events=types.SimpleNamespace(closing=Evenement()))
        vus['w'] = w
        return w
    faux.create_window = create_window
    faux.start = lambda: vus.update(items=list(vus['w'].events.closing.items))
    monkeypatch.setitem(sys.modules, 'webview', faux)
    monkeypatch.setattr(sys, 'stderr', sys.stderr)
    monkeypatch.setattr(sys, 'stdout', sys.stdout)
    appli()                                                 # monkeypatchs du faux moteur
    try:
        assert principal.main([]) == 0
    finally:
        for h in serveur.log.handlers:
            h.close()
        serveur.log.handlers[:] = []
        serveur.log.propagate = True
    assert vus['items'] == [principal.a_la_fermeture]


def test_verifier_fenetre(monkeypatch, tmp_path, capsys):
    """Défaut 8 : l'option importe pywebview et son module de plateforme, code ≠ 0 sinon."""
    from app_moliere import __main__ as principal
    sortie = tmp_path / 'r.json'
    assert principal.main(['--verifier-fenetre', '--rapport', str(sortie)]) == 0
    assert json.loads(sortie.read_text())['ok'] is True
    monkeypatch.setitem(sys.modules, 'webview', None)       # import impossible
    assert principal.main(['--verifier-fenetre', '--rapport', str(sortie)]) == 1
    assert 'erreur' in json.loads(sortie.read_text())


@pytestmark_verrou
def test_seconde_fenetre_ne_retelecharge_pas_si_le_modele_vient_d_arriver(appli, donnees, constantes, serveur_http,
                                                                         monkeypatch):
    """Défaut 5 : le modèle apparaît entre l'état « absent » et l'obtention du verrou : aucun téléchargement."""
    serveur_http['corps'] = constantes
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    _, base = appli()
    assert json.loads(get(base, '/api/etat')[1])['etat'] == 'absent'
    faux_modele(donnees / 'modele')                         # l'autre fenêtre a fini
    assert post(base, '/api/telecharger', {})[0] == 200
    attendre(base, 'pret', delai=20)
    assert serveur_http['hits'] == 0


@pytestmark_verrou
def test_reessayer_apres_panne_de_chargement_retelecharge(appli, donnees, constantes, serveur_http, monkeypatch):
    """Défaut 5 (exception) : après une panne de chargement, « Réessayer » télécharge bien."""
    serveur_http['corps'] = constantes
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    faux_modele(donnees / 'modele')
    vrai = sys.modules['chatbot.generate'].ChatBot

    class Casse:
        @classmethod
        def load(cls, *a, **k):
            raise RuntimeError('poids illisibles')
    sys.modules['chatbot.generate'].ChatBot = Casse
    _, base = appli()
    attendre(base, 'erreur')
    sys.modules['chatbot.generate'].ChatBot = vrai
    demo.chatbot_engine.cache_clear()
    post(base, '/api/telecharger', {})
    attendre(base, 'pret', delai=20)
    assert serveur_http['hits'] == 1


def test_commande_ne_touche_pas_au_zip_voisin(constantes, serveur_http, tmp_path, monkeypatch):
    """Défaut 9 : un chatbot_moliere_s03.zip déjà présent à côté de la destination reste intact ; rien ne traîne."""
    serveur_http['corps'] = constantes
    monkeypatch.setattr(modele, 'URL', serveur_http['url'])
    voisin = tmp_path / 'sortie' / modele.NOM_ZIP
    voisin.parent.mkdir()
    voisin.write_bytes(b'mon zip a moi')
    dest = tmp_path / 'sortie' / 'modele'
    assert modele.main(['--dest', str(dest)]) == 0
    assert modele.modele_present(dest) and voisin.read_bytes() == b'mon zip a moi'
    assert sorted(p.name for p in voisin.parent.iterdir()) == ['chatbot_moliere_s03.zip', 'modele']


def test_construction_publie_le_dmg_apres_verification():
    """Défaut 7 : DMG vérifié dans le dossier temporaire avant d'être déplacé ; ancienne app supprimée ;
    délai de l'autotest supérieur à son attente maximale ; fenêtre importable notée au rapport."""
    src = (Path(__file__).resolve().parents[1] / 'tools' / 'build_app_macos.py').read_text(encoding='utf-8')
    assert src.index('verifier_signature_dans_dmg(dmg, scratch)') < src.index('shutil.move(dmg, final)')
    assert src.index("shutil.rmtree(BUILD / f'{NOM}.app'") < src.index('brut = construire(')
    assert "'--attente-max', '1200'" in src and 'timeout=1800' in src
    assert "rapport['fenetre_importable'] = fenetre_importable(app)" in src
