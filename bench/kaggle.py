"""Run COMPLET (100 M tokens, soumissible) sur Kaggle (GPU T4), avec reprise si la session coupe.

Marche à suivre (remplacer <hf> par ton nom d'utilisateur Hugging Face) :
1. Mac : dataset ordonné, tokenizer, train.bin. Aucune ligne « ATTENTION flux » ne doit sortir.
     uv run python -m bench.run --config configs/bench/full.yaml --until prepare
2. Mac : une fois `uv run huggingface-cli login` (jeton « write »), puis envoi sur deux dépôts PRIVÉS :
     uv run python -m bench.kaggle push --config configs/bench/full.yaml --dataset <hf>/ftb-dataset --runs <hf>/ftb-runs
   dataset : train.jsonl.gz, tokenizer.json, data_meta.json, manifest.json ;
   runs : ce script et l'empreinte du kit (refus si le flux de 100 M tokens est coupé ou relu).
3. Kaggle : notebook, Accelerator « GPU T4 », Internet activé, secret HF_TOKEN (Add-ons > Secrets).
   Une seule cellule (`push` l'affiche déjà remplie), à relancer telle quelle si la session coupe :
     import os; from kaggle_secrets import UserSecretsClient
     os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
     !test -d kit || (git clone -q https://github.com/Meuleur/Piscine.git kit && git -C kit checkout -q 3709c85)
     !pip install -q "tokenizers==<version de data_meta.json>"
     !python -c "import shutil, huggingface_hub as h; shutil.copy(h.hf_hub_download('<hf>/ftb-runs', 'kaggle.py'), 'kaggle.py')"
     !python kaggle.py run --dataset <hf>/ftb-dataset --runs <hf>/ftb-runs --kit kit --run full
   Le script refait `ftb.prepare` (comme l'orga) et exige la même empreinte, envoie `state.pt` sur le
   dépôt des runs à chaque sauvegarde du kit (tous les 500 pas) et, à la relance, le récupère et
   ajoute `--resume`. `--compile` est mis par défaut (T4) ; `--no-compile` s'il échoue.
4. Mac : rapatrier le run et soumettre (2 par jour) :
     uv run python -m bench.kaggle pull --runs <hf>/ftb-runs --run full
     uv run python -m ftb.submit --run runs/full --participant "Benjamin Assouline" \\
         --dataset-url https://huggingface.co/datasets/<hf>/ftb-dataset
Le côté Kaggle (`run`) n'importe rien de `bench/` : bibliothèque standard, huggingface_hub et le kit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA_FILES = ("train.jsonl.gz", "tokenizer.json", "data_meta.json", "manifest.json")
RUN_FILES = ("ckpt.pt", "tokenizer.json", "train_log.jsonl", "data_meta.json", "samples.txt", "eval.json")
KIT_FILES = ("config/official.json", "ftb/__init__.py", "ftb/data.py", "ftb/evaluate.py", "ftb/model.py",
             "ftb/prepare.py", "ftb/submit.py", "ftb/tokenizer.py", "ftb/train.py")
FULL_TOKENS = 100_007_936  # 3 052 pas de 64 × 512 tokens
KAGGLE_CELL = """\
import os; from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!test -d kit || (git clone -q https://github.com/Meuleur/Piscine.git kit && git -C kit checkout -q 3709c85)
!pip install -q "tokenizers=={tok}"
!python -c "import shutil, huggingface_hub as h; shutil.copy(h.hf_hub_download('{runs}', 'kaggle.py'), 'kaggle.py')"
!python kaggle.py run --dataset {dataset} --runs {runs} --kit kit --run {run}
"""


class Hub:
    """Dépôt Hugging Face privé, ou un dossier local (`--local-hub`, pour les tests)."""

    def __init__(self, repo: str, repo_type: str, local: str | None = None):
        self.repo, self.type = repo, repo_type
        self.local = Path(local) / repo_type / repo if local else None
        if not self.local:
            from huggingface_hub import HfApi
            self.api = HfApi()

    def create(self) -> None:
        if self.local:
            self.local.mkdir(parents=True, exist_ok=True)
        else:
            self.api.create_repo(self.repo, repo_type=self.type, private=True, exist_ok=True)

    def exists(self, name: str) -> bool:
        if self.local:
            return (self.local / name).exists()
        return self.api.file_exists(self.repo, name, repo_type=self.type)

    def get(self, name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.local:
            src = self.local / name
        else:
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download(self.repo, name, repo_type=self.type))
        shutil.copyfile(src, dest)

    def put(self, src: Path, name: str) -> None:
        if self.local:  # copie puis renommage : atomique, comme un commit Hugging Face
            dest = self.local / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest.with_name(dest.name + ".part"))
            os.replace(dest.with_name(dest.name + ".part"), dest)
        else:
            self.api.upload_file(path_or_fileobj=str(src), path_in_repo=name, repo_id=self.repo, repo_type=self.type)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def kit_fingerprint(kit: Path) -> dict:
    return {f: sha256(kit / f) for f in KIT_FILES}


def kit_cmd(kit: Path, module: str, args: list) -> list[str]:
    return [sys.executable, "-m", module, *map(str, args)]


def kit_env(kit: Path) -> dict:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(kit), os.environ.get("PYTHONPATH", "")])}


def run_kit(kit: Path, module: str, args: list) -> None:
    print("$", module, *map(str, args), flush=True)
    if subprocess.run(kit_cmd(kit, module, args), cwd=kit, env=kit_env(kit), check=False).returncode:
        sys.exit(f"{module} a échoué")


# --------------------------------------------------------------------------- Kaggle

def train_with_sync(kit: Path, args: list, out: Path, runs: Hub, prefix: str, poll: float) -> None:
    """Lance ftb.train ; à chaque nouveau state.pt (tous les 500 pas), l'envoie sur le dépôt des runs."""
    print("$ ftb.train", *map(str, args), flush=True)
    proc = subprocess.Popen(kit_cmd(kit, "ftb.train", args), cwd=kit, env=kit_env(kit))
    sent = None
    while True:
        done = proc.poll() is not None
        state = out / "state.pt"
        try:
            if state.exists() and state.stat().st_mtime != sent:
                mtime = state.stat().st_mtime
                tmp = out / "_state_upload.pt"  # copie figée : le kit peut réécrire state.pt pendant l'envoi
                shutil.copyfile(state, tmp)
                runs.put(tmp, prefix + "state.pt")
                if (out / "train_log.jsonl").exists():
                    runs.put(out / "train_log.jsonl", prefix + "train_log.jsonl")
                tmp.unlink()
                sent = mtime
                print(f"point de reprise envoyé ({time.strftime('%H:%M:%S')})", flush=True)
        except (OSError, RuntimeError, ValueError) as e:  # réseau : on réessaie au tour suivant
            print(f"envoi du point de reprise impossible ({e}), nouvel essai plus tard", flush=True)
        if done:
            break
        time.sleep(poll)
    if proc.returncode:
        sys.exit(f"ftb.train a échoué (code {proc.returncode}) : relancer la cellule pour reprendre")


def cmd_run(a) -> None:
    kit, work = Path(a.kit).resolve(), Path(a.work).resolve()
    data, prepared, out = work / "data", work / "prepared", work / "runs" / a.run
    ds, runs = Hub(a.dataset, "dataset", a.local_hub), Hub(a.runs, "model", a.local_hub)
    prefix = f"{a.run}/"

    # 0. le kit cloné est exactement celui du Mac
    ref = work / "kit_sha256.json"
    runs.get("kit_sha256.json", ref)
    bad = [f for f, h in kit_fingerprint(kit).items() if json.loads(ref.read_text())[f] != h]
    if bad:
        sys.exit(f"le kit de {kit} diffère de celui du Mac : {bad} (git checkout 3709c85 ?)")

    # 1. données, puis ftb.prepare refait ici (comme l'orga) : même empreinte exigée
    for f in DATA_FILES:
        if not (data / f).exists():
            ds.get(f, data / f)
    expected = json.loads((data / "data_meta.json").read_text())
    import tokenizers
    if tokenizers.__version__ != expected["tokenizers_version"]:
        print(f"ATTENTION : tokenizers {tokenizers.__version__} ici, {expected['tokenizers_version']} sur le Mac "
              f"(pip install tokenizers=={expected['tokenizers_version']})", flush=True)
    meta_path = prepared / "data_meta.json"
    if not meta_path.exists() or json.loads(meta_path.read_text())["dataset_sha256"] != expected["dataset_sha256"]:
        run_kit(kit, "ftb.prepare", ["--inputs", data / "train.jsonl.gz", "--tokenizer", data / "tokenizer.json",
                                     "--out", prepared])
    got = json.loads(meta_path.read_text())["dataset_sha256"]
    if got != expected["dataset_sha256"]:
        sys.exit(f"empreinte du dataset différente ({got[:12]}… ici, {expected['dataset_sha256'][:12]}… sur le Mac) : "
                 f"le ré-entraînement de l'orga échouerait. Même version de tokenizers ?")

    # 2. entraînement : run déjà fini sur le dépôt, reprise, ou départ
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "ckpt.pt").exists() and runs.exists(prefix + "ckpt.pt"):
        print("run déjà terminé sur le dépôt des runs : récupéré", flush=True)
        for f in RUN_FILES:
            if runs.exists(prefix + f):
                runs.get(prefix + f, out / f)
    if not (out / "ckpt.pt").exists():
        if not (out / "state.pt").exists() and runs.exists(prefix + "state.pt"):
            runs.get(prefix + "state.pt", out / "state.pt")
            if runs.exists(prefix + "train_log.jsonl"):
                runs.get(prefix + "train_log.jsonl", out / "train_log.jsonl")
            print("point de reprise récupéré : --resume", flush=True)
        args = ["--data", prepared, "--tokenizer", data / "tokenizer.json", "--out", out]
        args += ["--resume"] if (out / "state.pt").exists() else []
        args += ["--compile"] if a.compile else []
        args += ["--device", a.device] if a.device else []
        args += ["--micro-batch", a.micro_batch] if a.micro_batch else []
        args += ["--debug-budget", a.debug_budget] if a.debug_budget else []  # tests seulement : non soumissible
        args += ["--validation", a.validation] if a.validation else []
        train_with_sync(kit, args, out, runs, prefix, a.poll)

    # 3. évaluation officielle, puis envoi du run complet
    if not (out / "eval.json").exists():
        run_kit(kit, "ftb.evaluate", ["--ckpt", out / "ckpt.pt", "--tokenizer", out / "tokenizer.json",
                                      "--out", out / "eval.json", *(["--device", a.device] if a.device else []),
                                      *(["--data", a.validation] if a.validation else [])])
    for f in RUN_FILES:
        runs.put(out / f, prefix + f)
    ev = json.loads((out / "eval.json").read_text())
    print(f"terminé : BPB {ev['bpb']:.4f} {json.dumps(ev['bpb_par_genre'])} → dépôt {a.runs}, dossier {prefix}", flush=True)


# --------------------------------------------------------------------------- Mac

def cmd_push(a) -> None:
    from bench.common import REPO
    from bench.config import load_run_config, mix_dir
    from bench.run import stream_coverage

    cfg = load_run_config(a.config)
    mdir = mix_dir(cfg)
    src = {"train.jsonl.gz": mdir / "train.jsonl.gz", "tokenizer.json": mdir / "tokenizer.json",
           "data_meta.json": mdir / "prepared" / "data_meta.json", "manifest.json": mdir / "manifest.json"}
    missing = [str(p) for p in src.values() if not p.exists()]
    if missing:
        sys.exit(f"manquant : {missing} (bench.run --until prepare d'abord)")
    meta = json.loads(src["data_meta.json"].read_text())
    if meta["tokenizer_sha256"] != sha256(src["tokenizer.json"]):
        sys.exit("train.bin préparé avec un autre tokenizer : relancer bench.run --until prepare")
    cov = stream_coverage(cfg, FULL_TOKENS)
    for w in cov["warnings"]:
        print(f"ATTENTION flux (100 M tokens) : {w}")
    if cov["warnings"] and not a.force:
        sys.exit("flux coupé ou relu au budget complet : corriger le mix (ou --force si c'est voulu)")
    ds, runs = Hub(a.dataset, "dataset", a.local_hub), Hub(a.runs, "model", a.local_hub)
    ds.create()
    runs.create()
    for name, path in src.items():
        ds.put(path, name)
    fp = Path(a.tmp or mdir) / "kit_sha256.json"
    fp.write_text(json.dumps(kit_fingerprint(REPO), indent=2))
    runs.put(fp, "kit_sha256.json")
    runs.put(Path(__file__), "kaggle.py")
    print(f"envoyé : dataset {a.dataset} (empreinte {meta['dataset_sha256'][:12]}…, tokenizers "
          f"{meta['tokenizers_version']}), script et empreinte du kit sur {a.runs}\n\nCellule Kaggle :\n"
          + KAGGLE_CELL.format(tok=meta["tokenizers_version"], runs=a.runs, dataset=a.dataset, run=cfg["run"]))


def cmd_pull(a) -> None:
    from bench.common import REPO
    runs = Hub(a.runs, "model", a.local_hub)
    out = Path(a.out) if a.out else REPO / "runs" / a.run
    if (out / "ckpt.pt").exists() and not a.force:
        sys.exit(f"{out} contient déjà un checkpoint (--force pour l'écraser)")
    for f in RUN_FILES:
        runs.get(f"{a.run}/{f}", out / f)
    print(f"run rapatrié dans {out}\n  uv run python -m ftb.submit --run {out} --participant \"Benjamin Assouline\" "
          f"--dataset-url https://huggingface.co/datasets/<dataset>")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run complet sur Kaggle (T4) avec reprise ; voir la docstring")
    ap.add_argument("--local-hub", default=None, help="tests : un dossier local remplace Hugging Face")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="Mac : envoie le dataset et ce script")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--tmp", default=None)
    r = sub.add_parser("run", help="Kaggle : prepare, train (reprise), evaluate")
    r.add_argument("--dataset", required=True)
    r.add_argument("--runs", required=True)
    r.add_argument("--kit", default="kit")
    r.add_argument("--work", default="/kaggle/working")
    r.add_argument("--run", default="full")
    r.add_argument("--no-compile", dest="compile", action="store_false")
    r.add_argument("--device", default=None)
    r.add_argument("--micro-batch", type=int, default=None, help="accumulation du kit : même batch effectif, moins de mémoire")
    r.add_argument("--poll", type=float, default=30.0)
    r.add_argument("--debug-budget", type=int, default=None, help="tests seulement (run non soumissible)")
    r.add_argument("--validation", default=None, help="tests seulement")
    q = sub.add_parser("pull", help="Mac : rapatrie le run dans runs/<run>/")
    q.add_argument("--runs", required=True)
    q.add_argument("--run", default="full")
    q.add_argument("--out", default=None)
    q.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"push": cmd_push, "run": cmd_run, "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    main()
