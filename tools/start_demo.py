"""Lancer la démo en préservant les liens nosync : uv run --no-project python tools/start_demo.py."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reproduce import ensure_links, check_kit


def health(url):
    try:
        with urlopen(url + '/api/status', timeout=1) as response:
            data = json.load(response)
        return data['final']['dataset_sha256'] == 'b88033ad2fbad01d1e1da101da4de4ce07a39c9a2060170ba0712705ab4ba8ee'
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Lancer la commande depuis Inteligence-lab-piscine/.')
    ensure_links(ROOT)
    check_kit(ROOT)
    url = f'http://127.0.0.1:{args.port}'
    if health(url):
        print(f'Démo déjà disponible : {url}')
        return
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise SystemExit('Ce port est occupé par un autre service ; choisir --port avec un autre numéro.')
    log = ROOT / 'data/stages/logs/demo-20261111.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = ['uv', 'run', 'python', '-m', 'demo', '--port', str(args.port)]
    if os.name != 'nt':
        command.insert(0, 'nohup')
    with log.open('a') as out:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=os.name != 'nt')
    state = ROOT / 'data/stages/demo_20261111'
    state.mkdir(exist_ok=True)
    (state / 'server.json').write_text(json.dumps({'pid': process.pid, 'url': url, 'log': str(log)}, indent=2) + '\n')
    for _ in range(30):
        if health(url):
            print(f'Démo disponible : {url}\nJournal : {log}')
            return
        if process.poll() is not None:
            raise SystemExit(f'Le démarrage a échoué. Consulter {log}.')
        time.sleep(1)
    print(f'Installation ou démarrage encore en cours, PID {process.pid}. Journal : {log}')


if __name__ == '__main__':
    main()
