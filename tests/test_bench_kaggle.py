"""bench.kaggle : push → run coupé en plein entraînement → nouvelle session (dossier vide) → reprise → pull.

Un dossier local remplace Hugging Face (`--local-hub`) ; le kit est celui du repo ; budget de mise
au point (8 pas), donc run non soumissible : on teste la mécanique, pas le BPB.
"""
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from bench.common import REPO
from bench.kaggle import RUN_FILES
from bench.run import run

_spec = importlib.util.spec_from_file_location("bench_fx", Path(__file__).parent / "fixtures/bench/make_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def test_push_run_cut_resume_pull(tmp_path):
    from ftb.tokenizer import train_hf
    tok = tmp_path / "tokenizer.json"
    train_hf([str(REPO / "corpus" / "moliere_train.txt")]).save(str(tok))
    val = tmp_path / "val.jsonl"
    val.write_text("".join(json.dumps({"id": g, "genre": g, "text": f"Un texte du genre {g}, assez court."}) + "\n"
                           for g in ("encyclopedique", "litteraire", "presse", "dialogue")), encoding="utf-8")
    fx.make_corpus(tmp_path / "data", layout="stage")
    cfg = fx.run_config(tmp_path, run="full", tokenizer={"path": str(tok)}, train={"mode": "full", "device": "cpu"})
    run(cfg, ("mix", "tokenizer", "prepare"))
    cfg_path = tmp_path / "full.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    hub = tmp_path / "hub"
    base = [sys.executable, "-m", "bench.kaggle", "--local-hub", str(hub)]
    push = base + ["push", "--config", str(cfg_path), "--dataset", "ben/ds", "--runs", "ben/runs"]

    # le mini-mix serait relu des centaines de fois en 100 M tokens : refusé sans --force
    r = subprocess.run(push, cwd=REPO, capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "relu" in r.stdout
    r = subprocess.run(push + ["--force"], cwd=REPO, check=True, capture_output=True, text=True)
    import tokenizers
    assert f'tokenizers=={tokenizers.__version__}' in r.stdout and "kaggle.py run --dataset ben/ds" in r.stdout
    assert (hub / "dataset/ben/ds/train.jsonl.gz").exists() and (hub / "model/ben/runs/kaggle.py").exists()

    # « Kaggle » : le script seul, hors du repo ; le kit est un dossier à part
    kag = tmp_path / "kaggle"
    kag.mkdir()
    shutil.copy(hub / "model/ben/runs/kaggle.py", kag / "kaggle.py")
    work = tmp_path / "working"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cmd = [sys.executable, str(kag / "kaggle.py"), "--local-hub", str(hub), "run", "--dataset", "ben/ds",
           "--runs", "ben/runs", "--kit", str(REPO), "--work", str(work), "--no-compile", "--device", device,
           "--debug-budget", str(8 * 32768), "--validation", str(val), "--poll", "0.1", "--micro-batch", "16"]
    state = hub / "model/ben/runs/full/state.pt"
    p = subprocess.Popen(cmd, cwd=kag, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 600
    while not state.exists():
        assert p.poll() is None and time.time() < deadline, "aucun point de reprise envoyé"
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGKILL)  # la session coupe (script ET ftb.train)
    p.wait()
    assert not (hub / "model/ben/runs/full/ckpt.pt").exists()
    shutil.rmtree(work)  # nouvelle session Kaggle : /kaggle/working est vide

    r = subprocess.run(cmd, cwd=kag, capture_output=True, text=True, timeout=900, check=False)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "point de reprise récupéré" in r.stdout
    meta = torch.load(hub / "model/ben/runs/full/ckpt.pt", map_location="cpu", weights_only=True)["meta"]
    assert meta["resumed_from_steps"] and meta["tokens_seen"] == 8 * 32768

    out = tmp_path / "pulled"
    subprocess.run(base + ["pull", "--runs", "ben/runs", "--run", "full", "--out", str(out)], cwd=REPO, check=True,
                   capture_output=True)
    assert all((out / f).exists() for f in RUN_FILES)
    r = subprocess.run([sys.executable, "-m", "ftb.submit", "--run", str(out), "--participant", "Test",
                        "--dest", str(tmp_path / "sub")], cwd=REPO, capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "non soumissible" in r.stdout + r.stderr  # le kit refuse un run de mise au point
