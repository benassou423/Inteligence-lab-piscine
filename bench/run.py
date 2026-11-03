"""Banc d'essai : une fine couche au-dessus du kit officiel (`ftb/`, jamais modifié).

    uv run python -m bench.run --config configs/bench/<run>.yaml [--stage mix|tokenizer|prepare|train|eval|all]
    uv run python -m bench.run --config configs/bench/full.yaml --until prepare   (mix → train.bin, avant Kaggle)

Un run = mix → tokenizer → ftb.prepare → ftb.train → ftb.evaluate → une ligne dans
experiments/results.csv. Chaque étape est sautée si ses entrées n'ont pas changé ; un run
d'entraînement déjà fini sur un AUTRE dataset est renommé `runs/<run>.prev-<date>` (le kit
refuse d'écraser un run). Mode `proxy` : `--debug-budget` (non soumissible), sur le GPU du Mac ;
mode `full` : 100 M tokens officiels, pour Kaggle (T4, voir bench/kaggle.py). Après `prepare`, le banc
vérifie que le budget ne coupe pas le dernier bloc et ne relit pas le flux (`stream_coverage`).
Runs de plus de 5 minutes : les lancer détachés, par exemple
    nohup uv run python -m bench.run --config configs/bench/proxy.yaml > runs/proxy.out 2>&1 &
puis `--resume` reprend un entraînement interrompu (points de sauvegarde du kit, tous les 500 pas).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bench.common import DOMAINS, REPO, append_csv, get_logger, read_json, resolve_path, sha256_file, write_json
from bench.config import data_root, load_run_config, mix_dir, results_csv, run_dir
from bench.mix import build_mix

STAGES = ("mix", "tokenizer", "prepare", "train", "eval")
RESULT_FIELDS = (["run", "date", "mode", "dataset_sha256", "dataset_tokens", "tokens_seen", "val_bytes_per_token",
                  "bpb_global"] + [f"bpb_{d}" for d in DOMAINS] + ["train_s", "total_s", "device", "notes"])

# Variante française du pré-découpage : l'apostrophe (' et ’) reste collée aux lettres qui la
# précèdent (« l' » + « homme », « qu' » + « il ») ; contractions anglaises retirées.
FR_SPLIT_PATTERN = (
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+['’]|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def split_pattern(name: str) -> str:
    from ftb.tokenizer import GPT4_SPLIT_PATTERN
    patterns = {"gpt4": GPT4_SPLIT_PATTERN, "fr": FR_SPLIT_PATTERN}
    if name not in patterns:
        raise SystemExit(f"tokenizer.pretokenizer inconnu : {name!r} (gpt4 | fr)")
    return patterns[name]


# --------------------------------------------------------------------------- appels au kit

def kit(module: str, args: list[str], log) -> None:
    """Lance `python -m <module> <args>` (le kit tel quel) ; sa sortie va dans le journal."""
    cmd = [sys.executable, "-m", module, *map(str, args)]
    log.info("$ " + " ".join(cmd[1:]))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    last_progress = 0.0
    for line in stream:
        text = line.strip()
        if not text:
            continue
        if line.endswith("\r"):  # barre de progression (\r) : une ligne toutes les 20 s au plus
            if time.time() - last_progress < 20:
                continue
            last_progress = time.time()
        log.info(f"  {text}")
    if proc.wait() != 0:
        raise SystemExit(f"{module} a échoué (code {proc.returncode}), voir le journal")


def device_for(cfg, key: str = "train") -> str:
    dev = (cfg[key].get("device") if key in cfg else None) or cfg["train"]["device"]
    if dev and dev != "auto":
        return dev
    from ftb.evaluate import default_device
    return default_device()


def tokens_seen_expected(cfg) -> int:
    from ftb import load_config
    tr = load_config()["train"]
    if cfg["train"]["mode"] == "full":
        return tr["max_steps"] * tr["tokens_per_step"]
    return max(1, math.ceil(int(cfg["train"]["debug_budget"]) / tr["tokens_per_step"])) * tr["tokens_per_step"]


def _ckpt_meta(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=True).get("meta", {})


def stream_coverage(cfg, seen: int | None = None) -> dict:
    """Ce que l'entraînement verra du flux `train.bin` : le kit lit exactement les `tokens_seen`
    premiers tokens, et relit le flux depuis le début s'il est plus court.

    `passes` > 1 : le début du flux est revu à la FIN de l'entraînement (répétition non choisie).
    `passes` < 1 : la fin du flux n'est jamais vue ; avec l'ordre `blocks` ou `as_is`, c'est le
    dernier bloc (« le meilleur pour la fin ») qui est coupé. `segments` dit lequel, et combien."""
    import numpy as np
    from tokenizers import Tokenizer

    mdir = mix_dir(cfg)
    meta = read_json(mdir / "prepared" / "data_meta.json")
    man = read_json(mdir / "manifest.json")
    seen, n = seen or tokens_seen_expected(cfg), meta["n_tokens"]
    out = {"tokens_seen": seen, "dataset_tokens": n, "passes": round(seen / n, 4), "warnings": [], "segments": []}
    if seen > n:
        out["warnings"].append(f"flux de {n:,} tokens < {seen:,} vus : relu {seen / n:.2f} fois, le début du mix "
                               f"revient en fin d'entraînement (agrandir le mix ou répéter exprès)")
    segs = man.get("segments") or []
    if seen < n and (len(segs) > 1 or man.get("order") == "as_is"):  # un seul bloc mélangé : fin tirée au hasard
        eot = Tokenizer.from_file(str(mdir / "tokenizer.json")).token_to_id("<|endoftext|>")
        stream = np.memmap(mdir / "prepared" / "train.bin", dtype=np.uint16, mode="r")
        docs_seen = int(np.count_nonzero(stream[:seen] == eot))  # documents vus en entier
        start = 0
        for s in segs:
            frac = min(max(docs_seen - start, 0), s["documents"]) / max(s["documents"], 1)
            out["segments"].append({**s, "seen": round(frac, 4)})
            if frac < 0.99:
                out["warnings"].append(f"{s['label']} : {frac:.0%} de ses documents vus (fin du flux coupée par le budget)")
            start += s["documents"]
    return out


# --------------------------------------------------------------------------- étapes

def stage_mix(cfg, force, log) -> dict:
    return build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], data_root(cfg), force=force, log=log)


def stage_tokenizer(cfg, force, log) -> dict:
    from ftb.tokenizer import check, train_hf

    mdir = mix_dir(cfg)
    out = mdir / "tokenizer.json"
    side = mdir / "tokenizer_info.json"
    tc = cfg["tokenizer"]
    if tc["path"]:
        src = resolve_path(tc["path"])
        key = {"from": str(src), "sha256": sha256_file(src)}
    else:
        sample = mdir / "tokenizer_sample.jsonl.gz"
        key = {"sample_sha256": sha256_file(sample), "pretokenizer": tc["pretokenizer"]}
    if not force and out.exists() and side.exists() and read_json(side).get("key") == key:
        log.info("tokenizer déjà fait pour ce même échantillon : réutilisé")
        return read_json(side)
    t0 = time.time()
    if tc["path"]:
        shutil.copyfile(src, out)
    else:
        tk = train_hf([str(sample)], pattern=split_pattern(tc["pretokenizer"]))
        tk.save(str(out))
    info = {"key": key, "check": check(str(out)), "seconds": round(time.time() - t0, 1)}
    write_json(side, info)
    log.info(f"tokenizer : {info['check']['bytes_per_token_validation']} octets/token sur la val "
             f"({tc['pretokenizer'] if not tc['path'] else 'réutilisé'}, {info['seconds']} s)")
    return info


def stage_prepare(cfg, force, log) -> dict:
    mdir = mix_dir(cfg)
    prepared = mdir / "prepared"
    man = read_json(mdir / "manifest.json")
    key = {"train_sha256": man["train_sha256"], "tokenizer_sha256": sha256_file(mdir / "tokenizer.json")}
    side = prepared / "bench_prepare.json"
    if not force and side.exists() and (prepared / "train.bin").exists() and read_json(side).get("key") == key:
        log.info("ftb.prepare déjà fait pour ce mix et ce tokenizer : réutilisé")
        meta = read_json(prepared / "data_meta.json")
    else:
        kit("ftb.prepare", ["--inputs", mdir / "train.jsonl.gz", "--tokenizer", mdir / "tokenizer.json",
                            "--out", prepared], log)
        write_json(side, {"key": key})
        meta = read_json(prepared / "data_meta.json")
        log.info(f"train.bin : {meta['n_tokens']:,} tokens, {meta['bytes_per_token']} octets/token, "
                 f"dataset_sha256 {meta['dataset_sha256'][:12]}…")
    cov = stream_coverage(cfg)  # le « relu N fois » du kit compte 100 M tokens, même en proxy
    log.info(f"budget de ce run : {cov['tokens_seen']:,} tokens, flux vu {cov['passes']:.2f} fois")
    for w in cov["warnings"]:
        log.warning(f"ATTENTION flux : {w}")
    return meta


def stage_train(cfg, force, resume, log) -> dict:
    mdir, rdir = mix_dir(cfg), run_dir(cfg)
    data_meta = read_json(mdir / "prepared" / "data_meta.json")
    tok_sha = sha256_file(mdir / "tokenizer.json")
    expected = tokens_seen_expected(cfg)
    tc = cfg["train"]
    official = tc["mode"] == "full"
    ckpt = rdir / "ckpt.pt"
    if ckpt.exists() and not (rdir / "state.pt").exists():
        meta = _ckpt_meta(ckpt)
        # `official` aussi : un --debug-budget de 100 M tokens voit autant de tokens qu'un run complet,
        # mais il n'est pas soumissible
        same = (meta.get("dataset_sha256") == data_meta["dataset_sha256"] and meta.get("tokenizer_sha256") == tok_sha
                and meta.get("tokens_seen") == expected and bool(meta.get("official")) == official)
        if same and not force:
            log.info(f"entraînement déjà fait pour ce dataset ({rdir}) : réutilisé")
            return meta
        old = rdir.with_name(f"{rdir.name}.prev-{dt.datetime.now():%Y%m%d-%H%M%S}")
        rdir.rename(old)
        log.info(f"ancien run (autre dataset, tokenizer, budget ou mode) déplacé dans {old}")
    elif (rdir / "state.pt").exists() and not resume:
        raise SystemExit(f"{rdir} contient un entraînement interrompu : relancer avec --resume")
    for w in stream_coverage(cfg)["warnings"]:
        log.warning(f"ATTENTION flux : {w}")
    device = device_for(cfg)
    if official and not device.startswith("cuda"):
        log.warning(f"ATTENTION : run complet sur {device} (float32). L'orga ré-entraîne sur T4 en float16 et "
                    f"exige le même BPB à ±0,02 : faire le run soumis sur Kaggle (bench/kaggle.py)")
    rdir.mkdir(parents=True, exist_ok=True)
    args = ["--data", mdir / "prepared", "--tokenizer", mdir / "tokenizer.json", "--out", rdir,
            "--device", device]
    if not official:
        args += ["--debug-budget", int(tc["debug_budget"])]
    if tc["micro_batch"]:
        args += ["--micro-batch", int(tc["micro_batch"])]
    if tc["compile"]:
        args.append("--compile")
    if cfg["eval"]["validation"] and not official:  # run complet : le BPB du checkpoint reste celui de la val du kit
        args += ["--validation", resolve_path(cfg["eval"]["validation"])]
    if resume and (rdir / "state.pt").exists():
        args.append("--resume")
    kit("ftb.train", args, log)
    meta = _ckpt_meta(ckpt)
    log.info(f"entraînement : {meta['tokens_seen']:,} tokens en {meta['train_time_s'] / 60:.1f} min "
             f"({meta['tokens_seen'] / max(meta['train_time_s'], 1e-9):,.0f} tok/s), BPB val {meta['val_bpb']:.4f}")
    return meta


def stage_eval(cfg, force, log) -> tuple[dict, bool]:
    rdir = run_dir(cfg)
    out = rdir / "eval.json"
    ckpt = rdir / "ckpt.pt"
    val = resolve_path(cfg["eval"]["validation"]) if cfg["eval"]["validation"] else REPO / "data" / "validation.jsonl"
    if not force and out.exists() and out.stat().st_mtime >= ckpt.stat().st_mtime and read_json(out).get("data") == str(val):
        log.info("ftb.evaluate déjà fait pour ce checkpoint : réutilisé")
        return read_json(out), False
    kit("ftb.evaluate", ["--ckpt", ckpt, "--tokenizer", rdir / "tokenizer.json", "--data", val,
                         "--out", out, "--device", device_for(cfg, "eval")], log)
    res = read_json(out)
    log.info(f"BPB {res['bpb']:.4f} " + " ".join(f"{g} {v:.4f}" for g, v in res["bpb_par_genre"].items()))
    return res, True


def append_result(cfg, timings: dict, log) -> dict:
    """Une ligne par évaluation dans experiments/results.csv."""
    mdir, rdir = mix_dir(cfg), run_dir(cfg)
    data_meta = read_json(mdir / "prepared" / "data_meta.json")
    meta = _ckpt_meta(rdir / "ckpt.pt")
    ev = read_json(rdir / "eval.json")
    man = read_json(mdir / "manifest.json")
    mode = "full" if meta.get("official") else "proxy"  # ce que le checkpoint EST, pas ce que dit la config
    budget = f"{meta['tokens_seen'] / 1e6:.4g}M"
    tok = "réutilisé" if cfg["tokenizer"]["path"] else cfg["tokenizer"]["pretokenizer"]
    cov = stream_coverage(cfg, meta["tokens_seen"])
    auto = (f"[{mode} {budget}, tok {tok}, mix {man['train_sha256'][:10]}, {man['order']}, "
            f"flux vu {cov['passes']:.2f}×" + (", ATTENTION flux" if cov["warnings"] else "") + "]")
    row = {
        "run": cfg["run"],
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dataset_sha256": data_meta["dataset_sha256"],
        "dataset_tokens": data_meta["n_tokens"],
        "tokens_seen": meta.get("tokens_seen"),
        "val_bytes_per_token": f"{ev['octets_par_token']:.4f}",
        "bpb_global": f"{ev['bpb']:.4f}",
        **{f"bpb_{d}": (f"{ev['bpb_par_genre'][d]:.4f}" if d in ev["bpb_par_genre"] else "") for d in DOMAINS},
        "train_s": meta.get("train_time_s"),
        "total_s": round(sum(timings.values()), 1),
        "device": meta.get("device"),
        "notes": f"{cfg.get('notes') or ''} {auto}".strip(),
    }
    append_csv(results_csv(cfg), row, RESULT_FIELDS)
    log.info(f"ligne ajoutée à {results_csv(cfg)}")
    return row


def run(cfg: dict, stage="all", force: bool = False, resume: bool = False, log=None) -> dict:
    rdir = run_dir(cfg)
    rdir.parent.mkdir(parents=True, exist_ok=True)
    mix_dir(cfg).mkdir(parents=True, exist_ok=True)
    log = log or get_logger("bench", mix_dir(cfg) / "bench.log")
    write_json(mix_dir(cfg) / "config.json", cfg)  # config complète, valeurs par défaut comprises
    state_path = mix_dir(cfg) / "bench_timings.json"
    timings = read_json(state_path) if state_path.exists() else {}
    out = {}
    stages = STAGES if stage == "all" else (stage,) if isinstance(stage, str) else tuple(stage)
    for st in stages:
        t0 = time.time()
        log.info(f"=== {cfg['run']} : {st} ===")
        if st == "mix":
            out[st] = stage_mix(cfg, force, log)
        elif st == "tokenizer":
            out[st] = stage_tokenizer(cfg, force, log)
        elif st == "prepare":
            out[st] = stage_prepare(cfg, force, log)
        elif st == "train":
            out[st] = stage_train(cfg, force, resume, log)
        elif st == "eval":
            out[st], fresh = stage_eval(cfg, force, log)
            timings[st] = round(time.time() - t0, 1)
            if fresh:
                out["result"] = append_result(cfg, timings, log)
        dt_s = round(time.time() - t0, 1)
        if dt_s > 2 or st not in timings:  # garde la durée du dernier vrai calcul
            timings[st] = dt_s
        write_json(state_path, timings)
        log.info(f"=== {st} : {dt_s} s ===")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Banc d'essai : mix → tokenizer → kit (prepare, train, evaluate) → results.csv")
    ap.add_argument("--config", required=True, help="configs/bench/<run>.yaml")
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    ap.add_argument("--until", choices=STAGES, default=None,
                    help="enchaîne les étapes depuis mix jusqu'à celle-ci (ex. prepare avant Kaggle)")
    ap.add_argument("--force", action="store_true", help="refaire l'étape même si rien n'a changé")
    ap.add_argument("--resume", action="store_true", help="reprendre un entraînement interrompu")
    ap.add_argument("--mode", choices=["proxy", "full"], default=None, help="remplace train.mode")
    ap.add_argument("--debug-budget", type=float, default=None, help="tokens vus en mode proxy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run", default=None, help="nom du run (défaut : config, puis nom du fichier)")
    args = ap.parse_args(argv)
    over: dict = {"train": {}}
    if args.mode:
        over["train"]["mode"] = args.mode
    if args.debug_budget:
        over["train"]["debug_budget"] = int(args.debug_budget)
    if args.device:
        over["train"]["device"] = args.device
    if args.run:
        over["run"] = args.run
    cfg = load_run_config(args.config, over)
    stage = STAGES[:STAGES.index(args.until) + 1] if args.until else args.stage
    run(cfg, stage, force=args.force, resume=args.resume)


if __name__ == "__main__":
    main()
