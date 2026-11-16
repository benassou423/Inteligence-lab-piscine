"""Construit Moliere.app (PyInstaller), la signe (ad hoc), crée le .dmg et teste le binaire gelé.

    uv run --group app python tools/build_app_macos.py [--embarquer-modele DOSSIER] [--sans-dmg]
                                                       [--modele-autotest DOSSIER | --autotest-telecharge]

Ordre : anciens rapport, DMG et application supprimés ; construction ; signature ; contrôle de la fenêtre
importable ; autotest du binaire gelé ; DMG monté et signature vérifiée dans le dossier temporaire ;
seulement ensuite le DMG est publié dans le dossier de sortie ;
le rapport est écrit en dernier (un échec ne laisse donc jamais un DMG neuf à côté d'un ancien rapport).

Tout ce qui est lourd est écrit sous data/stages/app_moliere_20261116/ (lien vers un dossier hors iCloud).
"""
import argparse
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUILD = ROOT / 'data' / 'stages' / 'app_moliere_20261116'
NOM = 'Moliere'
VERSION = '1.0.0'
IDENTIFIANT = 'io.github.benassou423.moliere'
MACOS_MINI = '14.0'  # la roue torch arm64 utilisée est compilée pour macOS 14 au minimum
# Modules lourds que le moteur n'importe pas : on les exclut pour garder l'application petite.
EXCLUS = ['matplotlib', 'pandas', 'datasets', 'pyarrow', 'fasttext', 'huggingface_hub', 'transformers',
          'pytest', 'tkinter', 'IPython', 'notebook', 'numba', 'scipy', 'sklearn', 'PIL', 'lxml', 'warcio',
          'trafilatura', 'datasketch', 'zstandard', 'mwparserfromhell', 'ftfy', 'pipeline', 'minigpt', 'bench',
          # inutiles à l'exécution (l'autotest du binaire gelé le prouve)
          'aiohttp', 'setuptools', 'pkg_resources', '_pytest', 'pluggy', 'iniconfig']
IMPORTS_CACHES = ['demo.server', 'chatbot.generate', 'chatbot.format', 'chatbot.ancrage', 'ftb.model',
                  'ftb.tokenizer', 'webview.platforms.cocoa', 'certifi']


def lancer(cmd, **kw):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, 'rb') as f:
        for bloc in iter(lambda: f.read(1 << 20), b''):
            h.update(bloc)
    return h.hexdigest()


def taille_dossier(dossier):
    return sum(p.stat().st_size for p in Path(dossier).rglob('*') if p.is_file() and not p.is_symlink())


def sous_icloud(chemin):
    """Vrai si le chemin est (probablement) synchronisé par iCloud : Bureau, Documents ou iCloud Drive."""
    texte = str(Path(chemin).expanduser().absolute())
    home = str(Path.home())
    return ('Mobile Documents' in texte or 'com~apple~CloudDocs' in texte
            or texte.startswith((home + '/Desktop/', home + '/Documents/')))


def preparer_sortie():
    """Crée le dossier de sortie sans casser un lien existant ; renvoie les avertissements éventuels."""
    avertissements = []
    stages = ROOT / 'data' / 'stages'
    if stages.is_symlink() and not stages.exists():
        raise SystemExit(f'« {stages} » est un lien cassé : réparez-le (jamais recréé ici, pour ne pas '
                         'casser les liens vers les dossiers *.nosync).')
    if not stages.exists():
        stages.mkdir(parents=True)
        avertissements.append(f'« {stages} » n’existait pas : créé comme dossier ordinaire.')
        if sous_icloud(stages):
            avertissements.append('Ce dossier est probablement synchronisé par iCloud : la construction fait ~1 Go '
                                  'de fichiers temporaires. Déplacez-le hors iCloud (dossier *.nosync + lien).')
    BUILD.mkdir(parents=True, exist_ok=True)
    return avertissements


def verifier_entrees(embarquer, modele_autotest, autotest_telecharge):
    """Arrête avant toute construction si le modèle de l'autotest manque (jamais de téléchargement muet)."""
    from app_moliere import modele
    if embarquer:
        if not modele.modele_present(embarquer):
            raise SystemExit(f'Le dossier à embarquer « {embarquer} » ne contient pas un modèle complet.')
        return
    if modele_autotest and modele.modele_present(modele_autotest):
        return
    if not autotest_telecharge:
        raise SystemExit(f'Le modèle de l’autotest « {modele_autotest} » est absent ou incomplet. '
                         'Donnez --modele-autotest DOSSIER, ou ajoutez --autotest-telecharge pour autoriser '
                         'l’autotest à télécharger le modèle (240 Mo) depuis GitHub.')


def env_autotest(modele_local, donnees, embarque):
    """Variables d'environnement de l'autotest. Modèle embarqué : MOLIERE_MODELE n'est PAS posé,
    sinon il passerait avant l'emplacement embarqué qu'on veut justement tester."""
    env = dict(os.environ, MOLIERE_DONNEES=str(donnees))
    env.pop('MOLIERE_MODELE', None)
    if not embarque and modele_local:
        env['MOLIERE_MODELE'] = str(Path(modele_local).resolve())
    return env


def verifier_rapport_autotest(rapport, embarque):
    """Lève si l'autotest n'a pas réussi, ou si (modèle embarqué) il n'a pas utilisé l'emplacement embarqué."""
    if not rapport.get('ok'):
        raise RuntimeError(f"autotest en échec : {rapport.get('erreur')}")
    if embarque and 'modele_embarque' not in Path(rapport.get('dossier_modele', '')).parts:
        raise RuntimeError('l’autotest n’a pas utilisé le modèle embarqué : '
                           f"dossier_modele = {rapport.get('dossier_modele')}")


def verifier_signature_dans_dmg(dmg, scratch):
    """Monte le DMG en lecture seule et vérifie la signature de l'application du volume."""
    point = scratch / 'volume_dmg'
    point.mkdir()
    lancer(['hdiutil', 'attach', '-readonly', '-nobrowse', '-noautoopen', '-mountpoint', point, dmg])
    try:
        res = subprocess.run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', point / f'{NOM}.app'],
                             capture_output=True, text=True)
        lignes = (res.stdout + res.stderr).strip().splitlines()
        return {'ok': res.returncode == 0, 'sortie': '\n'.join(lignes[-2:])}  # les deux dernières lignes suffisent
    finally:
        subprocess.run(['hdiutil', 'detach', point, '-force'], capture_output=True)


def construire(embarquer):
    for sous in ('work', 'dist', 'spec'):
        shutil.rmtree(BUILD / sous, ignore_errors=True)
    entree = BUILD / 'entree_moliere.py'
    entree.write_text('import multiprocessing\nfrom app_moliere.__main__ import main\n\n'
                      'if __name__ == "__main__":\n    multiprocessing.freeze_support()\n    raise SystemExit(main())\n',
                      encoding='utf-8')
    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
           '--name', NOM, '--icon', ROOT / 'app_moliere/ressources/Moliere.icns',
           '--osx-bundle-identifier', IDENTIFIANT,
           '--distpath', BUILD / 'dist', '--workpath', BUILD / 'work', '--specpath', BUILD / 'spec',
           '--paths', ROOT,
           '--add-data', f'{ROOT / "app_moliere/page.html"}:app_moliere',
           '--add-data', f'{ROOT / "app_moliere/ressources"}:app_moliere/ressources',
           # ftb/__init__.py lit config/official.json à l'import : sans lui, le moteur ne démarre pas
           '--add-data', f'{ROOT / "config/official.json"}:config']
    for module in IMPORTS_CACHES:
        cmd += ['--hidden-import', module]
    for module in EXCLUS:
        cmd += ['--exclude-module', module]
    if embarquer:
        cmd += ['--add-data', f'{Path(embarquer).resolve()}:modele_embarque']
    cmd.append(entree)
    lancer(cmd, cwd=BUILD)
    return BUILD / 'dist' / f'{NOM}.app'


def corriger_plist(app):
    chemin = app / 'Contents' / 'Info.plist'
    with open(chemin, 'rb') as f:
        info = plistlib.load(f)
    info.update({'CFBundleName': 'Molière', 'CFBundleDisplayName': 'Molière', 'CFBundleIdentifier': IDENTIFIANT,
                 'CFBundleShortVersionString': VERSION, 'CFBundleVersion': VERSION,
                 'NSHighResolutionCapable': True, 'LSMinimumSystemVersion': MACOS_MINI,
                 'NSHumanReadableCopyright': 'Code Apache 2.0, poids du modèle CC BY-SA 4.0',
                 'LSApplicationCategoryType': 'public.app-category.education'})
    with open(chemin, 'wb') as f:
        plistlib.dump(info, f)


def signer(app):
    """Signature ad hoc (aucune identité Developer ID sur cette machine) : pas de notarisation."""
    lancer(['xattr', '-cr', app])  # codesign refuse FinderInfo et autres attributs étendus
    lancer(['codesign', '--force', '--deep', '-s', '-', app])
    lancer(['codesign', '--verify', '--deep', '--strict', '--verbose=2', app])


def creer_dmg(app, scratch):
    dmg = scratch / f'{NOM}-{VERSION}-macos-arm64.dmg'
    dmg.unlink(missing_ok=True)
    prep = scratch / 'dmg_contenu'
    shutil.rmtree(prep, ignore_errors=True)
    prep.mkdir()
    lancer(['ditto', app, prep / app.name])  # ditto garde les liens symboliques et les signatures
    os.symlink('/Applications', prep / 'Applications')
    lancer(['hdiutil', 'create', '-volname', 'Molière', '-srcfolder', prep, '-ov', '-format', 'UDZO', dmg])
    shutil.rmtree(prep, ignore_errors=True)
    return dmg


def fenetre_importable(app):
    """Lance le binaire emballé avec --verifier-fenetre : pywebview et son module de plateforme doivent s'importer
    (sinon l'application se rabattrait en silence sur le navigateur)."""
    binaire = app / 'Contents' / 'MacOS' / NOM
    sortie = BUILD / 'fenetre_importable.json'
    sortie.unlink(missing_ok=True)
    res = subprocess.run([str(binaire), '--verifier-fenetre', '--rapport', str(sortie)],
                         capture_output=True, text=True, timeout=300)
    detail = json.loads(sortie.read_text(encoding='utf-8')) if sortie.exists() else {'erreur': res.stderr[-500:]}
    if res.returncode != 0 or not detail.get('ok'):
        raise RuntimeError(f"la fenêtre native n'est pas importable dans l'application emballée : {detail}")
    return True


def autotest_gele(app, modele_local, rapport, embarque):
    binaire = app / 'Contents' / 'MacOS' / NOM
    donnees = BUILD / 'donnees_autotest'
    shutil.rmtree(donnees, ignore_errors=True)
    env = env_autotest(modele_local, donnees, embarque)
    debut = time.time()
    # Délai du sous-processus strictement supérieur à l'attente maximale de l'autotest (1200 s).
    lancer([binaire, '--autotest', '--rapport', rapport, '--attente-max', '1200'], env=env, timeout=1800)
    return round(time.time() - debut, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--embarquer-modele', metavar='DOSSIER', help='copie le modèle dans l’application (hors ligne, plus lourde)')
    p.add_argument('--sans-dmg', action='store_true')
    p.add_argument('--modele-autotest', default=str(ROOT / 'data/models/chatbot_moliere'),
                   help='modèle local utilisé par l’autotest du binaire gelé')
    p.add_argument('--autotest-telecharge', action='store_true',
                   help='autorise l’autotest à télécharger le modèle (240 Mo) si le dossier local est absent')
    args = p.parse_args()
    debut = time.time()
    avertissements = preparer_sortie()
    for a in avertissements:
        print('ATTENTION :', a, flush=True)
    verifier_entrees(args.embarquer_modele, args.modele_autotest, args.autotest_telecharge)
    modele_local = args.modele_autotest if modele_present_local(args.modele_autotest) else None
    # 1. L'ancien rapport et l'ancien DMG disparaissent d'abord : rien de périmé ne reste en cas d'échec.
    chemin_rapport = BUILD / 'rapport_construction.json'
    chemin_autotest = BUILD / 'autotest_gele.json'
    for ancien in (chemin_rapport, chemin_autotest, *BUILD.glob(f'{NOM}-*.dmg')):
        ancien.unlink(missing_ok=True)
    shutil.rmtree(BUILD / f'{NOM}.app', ignore_errors=True)  # l'ancienne copie ne doit pas survivre à un échec
    brut = construire(args.embarquer_modele)
    # Le Bureau est géré par iCloud (File Provider) : il rajoute sans cesse l'attribut FinderInfo au
    # dossier .app, ce que codesign refuse. On signe donc dans un dossier temporaire hors iCloud.
    scratch = Path(tempfile.mkdtemp(prefix='moliere_build_'))
    try:
        app = scratch / f'{NOM}.app'
        lancer(['ditto', '--noextattr', '--norsrc', brut, app])
        corriger_plist(app)
        signer(app)
        rapport = {'version': VERSION, 'identifiant': IDENTIFIANT, 'macos_minimum': MACOS_MINI,
                   'modele_embarque': bool(args.embarquer_modele), 'signature': 'ad hoc (non notarisée)',
                   'taille_application_octets': taille_dossier(app), 'avertissements': avertissements}
        dmg = None if args.sans_dmg else creer_dmg(app, scratch)  # créé dans le dossier temporaire
        # 2. Fenêtre native importable, puis autotest du binaire gelé AVANT toute publication ; lèvent en cas d'échec.
        rapport['fenetre_importable'] = fenetre_importable(app)
        rapport['autotest_secondes'] = autotest_gele(app, modele_local, chemin_autotest, args.embarquer_modele)
        rapport['autotest'] = json.loads(chemin_autotest.read_text(encoding='utf-8'))
        verifier_rapport_autotest(rapport['autotest'], args.embarquer_modele)
        # 3. Signature de l'application DANS le volume monté (c'est lui qui fait foi), vérifiée dans le dossier
        #    temporaire : un échec du montage ne laisse rien dans le dossier de sortie. Publication ensuite.
        rapport['application_reference'] = 'celle du DMG'
        if dmg is not None:
            verif = verifier_signature_dans_dmg(dmg, scratch)
            rapport['signature_application_du_dmg'] = verif
            if not verif['ok']:
                raise RuntimeError('signature invalide dans le DMG : ' + verif['sortie'])
            final = BUILD / dmg.name
            shutil.move(dmg, final)
            rapport.update(dmg=str(final), dmg_octets=final.stat().st_size, dmg_sha256=sha256(final))
        copie = BUILD / f'{NOM}.app'
        shutil.rmtree(copie, ignore_errors=True)
        lancer(['ditto', app, copie])  # copie de travail ; iCloud y rajoute FinderInfo : codesign peut la refuser
        rapport['application'] = str(copie)
        rapport['application_note'] = ('copie locale sous data/stages, non fiable pour codesign --verify '
                                       '(attributs iCloud) : l’application du DMG fait foi')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    rapport['construction_secondes'] = round(time.time() - debut, 1)
    # 4. Le rapport est écrit en dernier.
    chemin_rapport.write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in rapport.items() if k != 'autotest'}, ensure_ascii=False, indent=2))


def modele_present_local(dossier):
    from app_moliere import modele
    return bool(dossier) and modele.modele_present(dossier)


if __name__ == '__main__':
    main()
