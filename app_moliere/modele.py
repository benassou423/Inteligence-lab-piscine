"""Trouver, télécharger et vérifier le modèle du chatbot Molière (sans dépendance réseau tierce).

Le modèle (240 Mo) est une pièce jointe publique d'une release GitHub. On le télécharge dans un
fichier « .part », on vérifie son empreinte sha256, on extrait seulement les 4 fichiers attendus,
on revérifie l'empreinte des poids, puis on met le dossier en place d'un seul geste (renommage).
Rien n'est jamais pris pour un modèle tant que tout n'est pas vérifié.

Commande : python -m app_moliere.modele [--dest DOSSIER] [--remplacer]
"""
import argparse
import contextlib
import hashlib
import http.client
import os
import shutil
import socket
import ssl
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    import fcntl  # verrou entre deux fenêtres (macOS, Linux) ; absent sous Windows
except ImportError:  # pragma: no cover
    fcntl = None

# --- Le modèle publié « S03 » (54 749 440 paramètres) -----------------------------------------
DEPOT = 'benassou423/Inteligence-lab-piscine'
ETIQUETTE = 'chatbot-moliere-20261116'
NOM_ZIP = 'chatbot_moliere_s03.zip'
URL = f'https://github.com/{DEPOT}/releases/download/{ETIQUETTE}/{NOM_ZIP}'
TAILLE_ZIP = 240_087_338
SHA_ZIP = '6d62aa50f4917fbb23d24e24a8d64807bde55f273c9cb4203c6970c32c886417'
SHA_CKPT = '01b83112db9152df00f138d6ee6a5bb4c075fc2406cda834e9484b6def7d0577'
DOSSIER_ZIP = 'chatbot_moliere'
FICHIERS = ('ckpt.pt', 'tokenizer.json', 'chat_config.json', 'training_report.json')
FICHIERS_REQUIS = ('ckpt.pt', 'tokenizer.json', 'chat_config.json')
# Taille maximale acceptée par fichier une fois décompressé (garde-fou contre une archive piégée).
TAILLE_MAX = {'ckpt.pt': 300_000_000, 'tokenizer.json': 20_000_000,
              'chat_config.json': 1_000_000, 'training_report.json': 5_000_000}
DELAI = 30  # secondes sans réponse du serveur avant d'abandonner
USER_AGENT = 'Moliere-app/1.0 (+https://github.com/benassou423/Inteligence-lab-piscine)'


class ErreurModele(Exception):
    """Erreur avec un message en français, affichable tel quel à l'utilisateur."""


class Annule(ErreurModele):
    """Téléchargement annulé par l'utilisateur."""


class AutreInstance(ErreurModele):
    """Une autre fenêtre de Molière est déjà en train de télécharger ou d'installer le modèle."""


MSG_ECRITURE = 'Écriture impossible (disque plein ou dossier protégé).'
MSG_AUTRE_FENETRE = 'Une autre fenêtre de Molière télécharge déjà le modèle.'
MARGE_DISQUE = 200_000_000  # octets laissés libres en plus du zip et de la copie extraite


# --- Où chercher le modèle ----------------------------------------------------------------------
def dossier_donnees():
    """Dossier des données de l'utilisateur (modèle et journal). MOLIERE_DONNEES le déplace (tests)."""
    force = os.environ.get('MOLIERE_DONNEES')
    if force:
        return Path(force)
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'Moliere'
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        return Path(base) / 'Moliere'
    return Path.home() / '.local' / 'share' / 'moliere'


def dossier_modele_installe():
    """Où l'application range le modèle qu'elle télécharge."""
    return dossier_donnees() / 'modele'


def est_gele():
    return bool(getattr(sys, 'frozen', False))


def dossier_ressources():
    """Dossier de page.html et des images, dans l'application gelée comme depuis les sources."""
    if est_gele():
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'app_moliere'
    return Path(__file__).resolve().parent


def modele_present(dossier):
    """Présent seulement si les fichiers nécessaires existent (et ne sont pas vides)."""
    d = Path(dossier)
    return all((d / nom).is_file() and (d / nom).stat().st_size > 0 for nom in FICHIERS_REQUIS)


def emplacements():
    """Les endroits où chercher, dans l'ordre : variable d'environnement, embarqué, données, dépôt."""
    liste = []
    env = os.environ.get('MOLIERE_MODELE')
    if env:
        liste.append(('variable d’environnement', Path(env)))
    if est_gele():
        liste.append(('embarqué', Path(getattr(sys, '_MEIPASS', '.')) / 'modele_embarque'))
    liste.append(('données utilisateur', dossier_modele_installe()))
    if not est_gele():
        liste.append(('dépôt', Path(__file__).resolve().parents[1] / 'data' / 'models' / 'chatbot_moliere'))
    return liste


def chercher_modele():
    """Dossier du premier modèle complet, ou None."""
    for _, dossier in emplacements():
        if modele_present(dossier):
            return dossier
    return None


# --- Téléchargement -----------------------------------------------------------------------------
def contexte_tls():
    """Contexte TLS avec les certificats de certifi : une application gelée ne trouve pas ceux du système."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def sha256_fichier(chemin, morceau=1 << 20):
    h = hashlib.sha256()
    with open(chemin, 'rb') as f:
        while True:
            bloc = f.read(morceau)
            if not bloc:
                break
            h.update(bloc)
    return h.hexdigest()


def traduire_erreur_reseau(exc, annule):
    """Transforme une panne réseau en message français (ou en Annule si l'utilisateur a annulé)."""
    if annule:
        return Annule('Téléchargement annulé.')
    if isinstance(exc, urllib.error.HTTPError):
        return ErreurModele(f'Le serveur de téléchargement a répondu « {exc.code} ». Réessayez plus tard.')
    raison = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(raison, ssl.SSLError):
        return ErreurModele('Connexion sécurisée refusée : vérifiez la date et l’heure du Mac, '
                            'ou un antivirus/proxy.')
    if isinstance(exc, http.client.IncompleteRead):
        return ErreurModele('Le téléchargement s’est arrêté trop tôt. Vérifiez la connexion et réessayez.')
    return ErreurModele('Impossible de joindre le serveur de téléchargement. '
                        'Vérifiez votre connexion Internet et réessayez.')


def verifier_espace(dossier, taille_zip):
    """Il faut de la place pour le zip, la copie extraite (même ordre de grandeur) et une marge."""
    besoin = 2 * taille_zip + MARGE_DISQUE
    try:
        libre = shutil.disk_usage(dossier).free
    except OSError:
        return  # impossible de mesurer : on tente quand même, l'écriture dira si c'est plein
    if libre < besoin:
        raise ErreurModele(f'Espace disque insuffisant : {besoin / 1e6:.0f} Mo nécessaires, '
                           f'{libre / 1e6:.0f} Mo disponibles.')


def _surveiller_annulation(rep, annuler, fini):
    """Coupe la connexion dès que l'annulation est demandée : une lecture bloquée se réveille tout de suite."""
    def boucle():
        while not fini.is_set():
            if annuler.wait(0.2):
                if not fini.is_set():
                    try:
                        rep.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except Exception:  # noqa: BLE001 : au pire, le délai de lecture finira par la réveiller
                        pass
                return
    threading.Thread(target=boucle, daemon=True).start()


def telecharger(url, cible, progression=None, annuler=None, delai=None, taille_attendue=None):
    """Télécharge `url` vers `cible` en flux. `progression(recus, total)` ; `annuler` : threading.Event.

    Écrit dans « cible.part » puis renomme : un fichier à moitié reçu n'a jamais le vrai nom.
    `taille_attendue` : le serveur doit annoncer exactement cette taille et ne pas en envoyer davantage."""
    cible = Path(cible)
    part = cible.with_name(cible.name + '.part')
    delai = DELAI if delai is None else delai

    def annule():
        return annuler is not None and annuler.is_set()
    try:
        part.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ErreurModele(MSG_ECRITURE) from exc
    if taille_attendue:
        verifier_espace(part.parent, taille_attendue)
    requete = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    contexte = contexte_tls() if url.lower().startswith('https') else None
    try:
        try:
            rep = urllib.request.urlopen(requete, timeout=delai, context=contexte)
        except Exception as exc:  # noqa: BLE001
            raise traduire_erreur_reseau(exc, annule()) from exc
        with rep:
            try:
                total = int(rep.headers.get('Content-Length') or 0)
            except ValueError:
                total = 0
            if taille_attendue and total and total != taille_attendue:
                raise ErreurModele(f'Le serveur annonce {total} octets au lieu de {taille_attendue} : '
                                   'téléchargement refusé.')
            fini = threading.Event()
            if annuler is not None:
                _surveiller_annulation(rep, annuler, fini)
            recus = 0
            try:
                try:
                    sortie = open(part, 'wb')
                except OSError as exc:
                    raise ErreurModele(MSG_ECRITURE) from exc
                with sortie:
                    if progression:
                        progression(0, total)
                    while True:
                        if annule():
                            raise Annule('Téléchargement annulé.')
                        try:
                            bloc = rep.read(1 << 16)
                        except Exception as exc:  # noqa: BLE001
                            raise traduire_erreur_reseau(exc, annule()) from exc
                        if not bloc:
                            break
                        recus += len(bloc)
                        if taille_attendue and recus > taille_attendue:
                            raise ErreurModele('Le fichier reçu est plus gros que prévu : téléchargement refusé.')
                        try:
                            sortie.write(bloc)
                        except OSError as exc:
                            raise ErreurModele(MSG_ECRITURE) from exc
                        if progression:
                            progression(recus, total)
            finally:
                fini.set()
            if annule():
                raise Annule('Téléchargement annulé.')
            if (total and recus != total) or (taille_attendue and recus != taille_attendue):
                raise ErreurModele('Le téléchargement s’est arrêté trop tôt. Vérifiez la connexion et réessayez.')
        try:
            os.replace(part, cible)
        except OSError as exc:
            raise ErreurModele(MSG_ECRITURE) from exc
    finally:
        part.unlink(missing_ok=True)  # après un succès, le fichier a déjà été renommé
    return cible


# --- Verrou entre fenêtres et nettoyage ---------------------------------------------------------
@contextlib.contextmanager
def verrou_donnees(dossier=None):
    """Verrou exclusif non bloquant sur « .verrou » : une seule fenêtre télécharge/installe à la fois.

    Lève AutreInstance si une autre fenêtre le tient. Sans fcntl (Windows) : pas de verrou."""
    dossier = Path(dossier) if dossier else dossier_donnees()
    try:
        dossier.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ErreurModele(MSG_ECRITURE) from exc
    if fcntl is None:
        yield
        return
    try:
        fichier = open(dossier / '.verrou', 'a+')
    except OSError as exc:
        raise ErreurModele(MSG_ECRITURE) from exc
    try:
        try:
            fcntl.flock(fichier, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AutreInstance(MSG_AUTRE_FENETRE) from exc
        try:
            yield
        finally:
            fcntl.flock(fichier, fcntl.LOCK_UN)
    finally:
        fichier.close()


def verrou_libre(dossier=None):
    """Vrai si aucune autre fenêtre ne télécharge en ce moment."""
    try:
        with verrou_donnees(dossier):
            return True
    except AutreInstance:
        return False
    except ErreurModele:
        return True


def nettoyer_orphelins(dossier=None):
    """Supprime les restes d'un téléchargement interrompu (à appeler sous le verrou), dans `dossier` seulement.

    Motifs : « chatbot_moliere_s03.zip(.part) », « .moliere-install-* », « modele.ancien-* ».
    Si le modèle manque mais qu'un « ancien » complet existe (arrêt en plein remplacement), on le remet."""
    d = Path(dossier) if dossier else dossier_donnees()
    retires = []
    if not d.is_dir():
        return retires
    for p in sorted(d.iterdir()):
        n = p.name
        if n in (NOM_ZIP, NOM_ZIP + '.part') or n.startswith('.moliere-install-'):
            pass
        elif n.startswith('modele.ancien-'):
            if (d / 'modele').is_symlink():
                continue  # « modele » est un lien (même cassé) : on ne touche ni au lien ni à l'ancien dossier
            if not (d / 'modele').exists() and p.is_dir() and not p.is_symlink() and modele_present(p):
                os.replace(p, d / 'modele')
                retires.append(n + ' (restauré)')
                continue
        else:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        retires.append(n)
    return retires


def verifier_remplacable(dest):
    """On ne remplace qu'un dossier qui ne contient QUE des fichiers du modèle (jamais un dossier quelconque)."""
    dest = Path(dest)
    if not dest.exists() and not dest.is_symlink():
        return
    if dest.is_symlink() or not dest.is_dir():
        raise ErreurModele(f'« {dest} » n’est pas un dossier de modèle : il n’est pas remplacé.')
    # Le Finder crée « .DS_Store » (et « ._* ») dès qu'on ouvre le dossier : ce ne sont pas des fichiers étrangers.
    inconnus = sorted(p.name for p in dest.iterdir()
                      if p.name not in FICHIERS and p.name != '.DS_Store' and not p.name.startswith('._'))
    if inconnus:
        raise ErreurModele(f'Le dossier « {dest} » contient des fichiers qui ne sont pas ceux du modèle '
                           f'({", ".join(inconnus[:3])}) : il n’est pas remplacé. Déplacez-le puis réessayez.')


# --- Extraction sûre ----------------------------------------------------------------------------
def extraire_sur(archive, destination):
    """Extrait les seuls 4 fichiers attendus de `archive` dans `destination` (dossier existant).

    Refuse : chemin absolu ou « .. », nom inattendu, lien symbolique, doublon, taille anormale."""
    destination = Path(destination)
    attendus = {f'{DOSSIER_ZIP}/{nom}': nom for nom in FICHIERS}
    vus = set()
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ErreurModele('Le fichier téléchargé est illisible (archive corrompue).') from exc
    with zf:
        for info in zf.infolist():
            nom = info.filename
            if nom.startswith('/') or '\\' in nom or '..' in nom.split('/') or ':' in nom:
                raise ErreurModele('Archive refusée : chemin dangereux.')
            if nom == f'{DOSSIER_ZIP}/' and info.is_dir():
                continue
            if nom not in attendus:
                raise ErreurModele('Archive refusée : elle contient un fichier inattendu.')
            genre = stat.S_IFMT(info.external_attr >> 16)  # 0 : l'archive ne dit rien (fichier ordinaire)
            if genre not in (0, stat.S_IFREG):
                raise ErreurModele('Archive refusée : lien ou fichier spécial.')
            court = attendus[nom]
            if nom in vus:
                raise ErreurModele('Archive refusée : fichier en double.')
            vus.add(nom)
            if info.file_size > TAILLE_MAX[court]:
                raise ErreurModele('Archive refusée : taille anormale.')
            copie = 0
            with zf.open(info) as entree, open(destination / court, 'wb') as sortie:
                while True:
                    bloc = entree.read(1 << 20)
                    if not bloc:
                        break
                    copie += len(bloc)
                    if copie > TAILLE_MAX[court]:  # l'en-tête mentait sur la taille
                        raise ErreurModele('Archive refusée : taille anormale.')
                    sortie.write(bloc)
    manque = [n for n in FICHIERS_REQUIS if f'{DOSSIER_ZIP}/{n}' not in vus]
    if manque:
        raise ErreurModele('Archive incomplète : il manque ' + ', '.join(manque) + '.')


def installer_archive(archive, dest, sha_zip=None, sha_ckpt=None):
    """Vérifie l'archive, l'extrait, vérifie les poids, puis met `dest` en place de façon atomique."""
    dest = Path(dest)
    verifier_remplacable(dest)
    sha_zip = SHA_ZIP if sha_zip is None else sha_zip
    sha_ckpt = SHA_CKPT if sha_ckpt is None else sha_ckpt
    if sha256_fichier(archive) != sha_zip:
        raise ErreurModele('L’empreinte du fichier téléchargé ne correspond pas : fichier abîmé ou modifié. '
                           'Il a été écarté, réessayez.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.moliere-install-', dir=dest.parent))
    ancien = None
    try:
        extraire_sur(archive, tmp)
        if sha256_fichier(tmp / 'ckpt.pt') != sha_ckpt:
            raise ErreurModele('L’empreinte des poids du modèle ne correspond pas. Rien n’a été installé.')
        if dest.exists():  # remplacement : l'ancien dossier est mis de côté, puis supprimé
            ancien = dest.with_name(dest.name + f'.ancien-{os.getpid()}')
            os.replace(dest, ancien)
        os.replace(tmp, dest)
    except BaseException:
        if ancien is not None and not dest.exists():
            os.replace(ancien, dest)  # retour en arrière
            ancien = None
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if ancien is not None:
            shutil.rmtree(ancien, ignore_errors=True)
    return dest


def telecharger_et_installer(dest, progression=None, annuler=None, etape=None, url=None, deja_installe=None,
                             dossier_travail=None):
    """Téléchargement, vérification, installation. `etape('verification')` signale la fin du téléchargement.

    Verrou exclusif sur le dossier de travail (par défaut celui qui contient `dest`) : AutreInstance si une autre
    fenêtre travaille déjà. `deja_installe()` est appelée sous le verrou : si elle renvoie vrai (une autre fenêtre
    vient de finir), rien n'est téléchargé et la fonction renvoie None."""
    dest = Path(dest)
    verifier_remplacable(dest)  # avant de télécharger 240 Mo pour rien
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ErreurModele(MSG_ECRITURE) from exc
    travail = Path(dossier_travail) if dossier_travail else dest.parent
    with verrou_donnees(travail):
        if deja_installe is not None and deja_installe():
            return None
        archive = travail / NOM_ZIP
        try:
            telecharger(url or URL, archive, progression, annuler, taille_attendue=TAILLE_ZIP)
            if annuler is not None and annuler.is_set():
                raise Annule('Téléchargement annulé.')
            if etape:
                etape('verification')
            installer_archive(archive, dest)
        except ErreurModele:
            archive.unlink(missing_ok=True)  # une archive refusée ne doit pas être réutilisée
            raise
        except OSError as exc:
            archive.unlink(missing_ok=True)
            raise ErreurModele(MSG_ECRITURE) from exc
        archive.unlink(missing_ok=True)
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dest', type=Path, default=None,
                   help='dossier d’installation (défaut : dossier des données de l’utilisateur)')
    p.add_argument('--remplacer', action='store_true', help='remplacer un dossier existant')
    args = p.parse_args(argv)
    dest = args.dest or dossier_modele_installe()
    if dest.exists() and not args.remplacer:
        print(f'{dest} existe déjà : rien n’est écrasé (ajouter --remplacer pour le remplacer).', file=sys.stderr)
        return 2
    debut = time.time()
    dernier = [0.0]

    def avance(recus, total):
        if time.time() - dernier[0] > 1 or (total and recus == total):
            dernier[0] = time.time()
            print(f'\r{recus / 1e6:7.1f} / {total / 1e6:.1f} Mo', end='', file=sys.stderr, flush=True)
    # L'archive et le verrou vont dans un dossier temporaire à côté de la destination : on n'écrase jamais
    # un « chatbot_moliere_s03.zip » voisin et rien ne reste après la commande.
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        travail = Path(tempfile.mkdtemp(dir=dest.parent, prefix='.moliere-telechargement-'))
    except OSError:
        print(f'\nÉchec : {MSG_ECRITURE}', file=sys.stderr)
        return 1
    try:
        telecharger_et_installer(dest, avance, dossier_travail=travail)
    except ErreurModele as exc:
        print(f'\nÉchec : {exc}', file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(travail, ignore_errors=True)
    print(f'\nModèle installé et vérifié dans {dest} ({time.time() - debut:.0f} s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
