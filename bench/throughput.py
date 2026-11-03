"""Débit de l'entraînement officiel (`ftb`) sur ce device, et budget proxy conseillé.

    uv run python -m bench.throughput [--device mps] [--micro-batches 64,32,16] [--target-min 12]

- même calcul que `ftb.train` : modèle et optimiseur du kit, lot de 64 × 512 tokens, précision
  du kit (float32 sur MPS et CPU, float16 sur CUDA), `--micro-batch` = accumulation (même résultat) ;
- mesure aussi les coûts fixes d'un run proxy : évaluations sur la val (3 quarts pendant
  l'entraînement, la val entière à la fin, puis `ftb.evaluate`) et les 5 échantillons ;
- propose le budget `--debug-budget` qui tient dans `--target-min` minutes au total.
Sortie : experiments/throughput_<device>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import tempfile
import time

import numpy as np
import torch

from bench.common import REPO, get_logger, write_json


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def chip_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name()
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return device


def train_speed(device: str, micro: int, steps: int = 10, warmup: int = 3) -> dict:
    """tokens/s de pas complets d'entraînement du kit (avant, arrière, AdamW)."""
    from ftb import load_config
    from ftb.model import build_model
    from ftb.train import make_optimizer, pick_precision

    cfg = load_config()
    tr = cfg["train"]
    torch.manual_seed(tr["seed"])
    model = build_model(cfg["model"]).to(device)
    opt = make_optimizer(model, tr, device)
    amp_dtype, use_scaler = pick_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    B, T = tr["batch_size"], tr["seq_len"]
    accum = B // micro
    rng = np.random.default_rng(0)

    def step():
        xy = torch.from_numpy(rng.integers(0, cfg["model"]["vocab_size"], size=(B, T + 1)))
        for m in range(accum):
            chunk = xy[m * micro:(m + 1) * micro].to(device)
            x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                _, loss = model(x, y)
            scaler.scale(loss / accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    sec = time.perf_counter() - t0
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()
    return {"micro_batch": micro, "tokens_per_s": B * T * steps / sec, "s_per_step": sec / steps,
            "precision": str(amp_dtype)}


def fixed_costs(device: str) -> dict:
    """Secondes d'une évaluation de la val entière (fenêtres glissantes, float32) et des 5 échantillons."""
    from ftb import VALIDATION_PATH, load_config
    from ftb.evaluate import evaluate_bpb, read_jsonl
    from ftb.model import build_model
    from ftb.tokenizer import load_tokenizer, train_hf
    from ftb.train import generate_samples

    cfg = load_config()
    model = build_model(cfg["model"]).to(device)
    docs = read_jsonl(VALIDATION_PATH)
    with tempfile.TemporaryDirectory() as d:  # tokenizer jetable, entraîné sur la val elle-même
        path = f"{d}/tok.json"
        train_hf([str(VALIDATION_PATH), str(REPO / "corpus" / "moliere_train.txt")]).save(path)
        tok = load_tokenizer(path)
        _sync(device)
        t0 = time.perf_counter()
        evaluate_bpb(model, tok, docs, device=device)
        _sync(device)
        t_eval = time.perf_counter() - t0
        t0 = time.perf_counter()
        generate_samples(model, tok, device)
        t_samples = time.perf_counter() - t0
    return {"full_eval_s": t_eval, "samples_s": t_samples}


def main(argv=None):
    from ftb.evaluate import default_device

    ap = argparse.ArgumentParser(description="Débit de ftb.train et budget proxy conseillé")
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro-batches", default="64,32,16")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--target-min", type=float, default=12.0, help="durée totale visée d'un run proxy")
    args = ap.parse_args(argv)
    log = get_logger("bench")
    device = args.device or default_device()
    log.info(f"{chip_name(device)} ({device})")
    rows = []
    for mb in [int(x) for x in args.micro_batches.split(",")]:
        r = train_speed(device, mb, args.steps)
        rows.append(r)
        log.info(f"micro-batch {mb:3d} : {r['tokens_per_s']:,.0f} tok/s ({r['s_per_step']:.2f} s/pas, {r['precision']})")
    best = max(rows, key=lambda r: r["tokens_per_s"])
    costs = fixed_costs(device)
    # pendant un run proxy : 3 quarts de val + val entière (ftb.train) + val entière (ftb.evaluate)
    overhead = 2.75 * costs["full_eval_s"] + costs["samples_s"] + 30  # + démarrages, prepare, vérifs
    tps = best["tokens_per_s"]
    budget_tokens = max(0.0, args.target_min * 60 - overhead) * tps
    step_tokens = 64 * 512
    budget = int(budget_tokens // (step_tokens * 10)) * step_tokens * 10  # multiple de 10 pas
    est = {f"{b // 1_000_000}M": round((b / tps + overhead) / 60, 1) for b in (10_000_000, 20_000_000, 30_000_000,
                                                                               100_007_936)}
    out = {
        "measured": dt.datetime.now().isoformat(timespec="seconds"),
        "chip": chip_name(device), "device": device, "torch": torch.__version__,
        "train": rows, "best_micro_batch": best["micro_batch"], "tokens_per_s": round(tps),
        "fixed_costs_s": {k: round(v, 1) for k, v in costs.items()}, "overhead_per_run_s": round(overhead),
        "total_minutes_for_budget": est,
        "recommended_debug_budget": budget, "target_minutes": args.target_min,
        "note": "budget = (cible − coûts fixes) × débit, arrondi à 10 pas de 32 768 tokens",
    }
    path = REPO / "experiments" / f"throughput_{device.split(':')[0]}.json"
    write_json(path, out)
    log.info(f"meilleur micro-batch {best['micro_batch']} : {tps:,.0f} tok/s ; val entière {costs['full_eval_s']:.0f} s, "
             f"échantillons {costs['samples_s']:.0f} s → budget proxy conseillé {budget:,} tokens "
             f"(~{args.target_min:g} min) ; {est}")
    log.info(f"→ {path}")


if __name__ == "__main__":
    main()
