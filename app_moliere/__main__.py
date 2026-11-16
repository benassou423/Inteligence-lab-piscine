"""Lanceur de l'application Molière : fenêtre native (pywebview) sur un serveur local.

    python -m app_moliere                    fenêtre
    python -m app_moliere --navigateur       ouvre le navigateur par défaut
    python -m app_moliere --sans-fenetre     serveur seul
    python -m app_moliere --autotest --rapport r.json   essai complet puis sortie (code 0 si réussi)
"""
import argparse
import json
import multiprocessing
import sys
import threading
import time
import urllib.error
import urllib.request

from app_moliere import VERSION, modele, serveur


def appel(port, chemin, corps=None, delai=60):
    """Appel HTTP local (utilisé par l'autotest) ; renvoie le JSON de la réponse."""
    url = f'http://127.0.0.1:{port}{chemin}'
    if corps is None:
        requete = urllib.request.Request(url)
    else:
        requete = urllib.request.Request(url, data=json.dumps(corps).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(requete, timeout=delai) as rep:
        return json.loads(rep.read())


def autotest(port, rapport, attente_max):
    """Attend le modèle (le télécharge au besoin), pose une vraie question, écrit le résultat."""
    debut = time.time()
    resultat = {'ok': False, 'version': VERSION, 'gele': modele.est_gele(), 'telecharge': False}
    try:
        etat = appel(port, '/api/etat')
        resultat['etat_initial'] = etat['etat']
        while True:
            etat = appel(port, '/api/etat')
            if etat['etat'] == 'pret':
                break
            if etat['etat'] == 'erreur':
                raise RuntimeError(etat['message'])
            if etat['etat'] == 'absent':
                resultat['telecharge'] = True
                resultat['debut_telechargement'] = time.time() - debut
                appel(port, '/api/telecharger', {})
            if time.time() - debut > attente_max:
                raise TimeoutError('le modèle n’est pas prêt dans le délai imparti')
            time.sleep(1)
        resultat['secondes_attente_modele'] = round(time.time() - debut, 1)
        resultat['dossier_modele'] = str(serveur.GESTION.dossier)
        question = 'Bonjour, Monsieur, comment vous portez-vous ?'
        debut_chat = time.time()
        rep = appel(port, '/api/chat', {'messages': [{'role': 'user', 'content': question}],
                                        'persona': 'MOLIÈRE', 'seed': 1337}, delai=300)
        resultat.update(question=question, reponse=rep.get('reply'), personnage=rep.get('persona'),
                        secondes_reponse=round(time.time() - debut_chat, 2))
        resultat['ok'] = bool(isinstance(rep.get('reply'), str) and rep['reply'].strip())
    except Exception as exc:  # noqa: BLE001 : toute panne est reportée dans le rapport
        resultat['erreur'] = f'{type(exc).__name__}: {exc}'
    texte = json.dumps(resultat, ensure_ascii=False, indent=2)
    print(texte, flush=True)
    if rapport:
        with open(rapport, 'w', encoding='utf-8') as f:
            f.write(texte + '\n')
    return 0 if resultat['ok'] else 1


def a_la_fermeture():
    """Appelée par pywebview à la fermeture de la fenêtre (Cmd+Q compris : Cocoa ne repasse pas par `finally`).

    Ne renvoie jamais False : cela annulerait la fermeture. `fermer` peut être appelée deux fois sans danger."""
    try:
        serveur.log.info('fermeture')
        serveur.GESTION.fermer(2.0)
    except Exception:  # noqa: BLE001 : rien ne doit empêcher la fenêtre de se fermer
        pass
    return None


def verifier_fenetre(rapport=None):
    """Importe pywebview et son module de plateforme SANS ouvrir de fenêtre ; code 0 si tout s'importe."""
    module = {'darwin': 'webview.platforms.cocoa', 'win32': 'webview.platforms.winforms'}.get(
        sys.platform, 'webview.platforms.gtk')
    resultat = {'ok': False, 'module': module}
    try:
        import importlib
        importlib.import_module('webview')
        importlib.import_module(module)
        resultat['ok'] = True
    except Exception as exc:  # noqa: BLE001
        resultat['erreur'] = f'{type(exc).__name__}: {exc}'
    texte = json.dumps(resultat, ensure_ascii=False)
    if sys.stdout is not None:
        print(texte, flush=True)
    if rapport:
        with open(rapport, 'w', encoding='utf-8') as f:
            f.write(texte + '\n')
    return 0 if resultat['ok'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--port', type=int, default=0, help='port local (défaut : un port libre)')
    p.add_argument('--navigateur', action='store_true', help='ouvrir le navigateur au lieu d’une fenêtre')
    p.add_argument('--sans-fenetre', action='store_true', help='serveur seul')
    p.add_argument('--autotest', action='store_true', help='essai complet puis sortie')
    p.add_argument('--rapport', help='fichier JSON de résultat de l’autotest')
    p.add_argument('--verifier-fenetre', action='store_true',
                   help='vérifie que la fenêtre native (pywebview) s’importe, sans l’ouvrir')
    p.add_argument('--attente-max', type=int, default=1800, help='secondes d’attente du modèle (autotest)')
    args = p.parse_args(argv)
    if args.verifier_fenetre:
        return verifier_fenetre(args.rapport)

    serveur.installer_journal()
    import torch
    torch.set_num_threads(4)
    srv = serveur.creer_serveur(args.port)
    port = srv.server_port
    url = f'http://127.0.0.1:{port}/'
    serveur.log.info('Molière %s démarré sur %s (gelé : %s)', VERSION, url, modele.est_gele())
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    code = 0
    try:
        if args.autotest:
            code = autotest(port, args.rapport, args.attente_max)
        elif args.sans_fenetre:
            print(f'Serveur disponible : {url}', flush=True)
            while True:
                time.sleep(3600)
        else:
            fenetre = False
            if not args.navigateur:
                try:
                    import webview
                    w = webview.create_window('Molière', url, width=980, height=760, min_size=(420, 560))
                    w.events.closing += a_la_fermeture
                    fenetre = True
                    webview.start()  # rend la main quand la fenêtre est fermée
                except Exception:  # pywebview absent ou refusé : on se rabat sur le navigateur
                    serveur.log.exception('fenêtre native indisponible, repli sur le navigateur')
                    fenetre = False
            if not fenetre:
                import webbrowser
                serveur.MODE_NAVIGATEUR = True  # la page affiche alors un bouton « Quitter »
                webbrowser.open(url)
                print(f'Molière est ouvert dans le navigateur : {url} (bouton Quitter ou Ctrl+C pour arrêter)', flush=True)
                serveur.ARRET.wait()
                time.sleep(0.3)  # laisse partir la réponse de « Quitter »
    except KeyboardInterrupt:
        pass
    finally:
        # Fermeture : on demande l'annulation d'un téléchargement en cours et on attend le fil 2 s au plus
        # (le fil est un démon : sans cela ses nettoyages ne tourneraient pas).
        serveur.GESTION.fermer(2.0)
        srv.shutdown()
        srv.server_close()
        serveur.log.info('arrêt propre')
    return code


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
