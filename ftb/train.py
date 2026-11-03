"""Entraînement officiel de la compétition. Modèle, hyperparamètres et budget sont figés.

    python -m ftb.train --data prepared/ --tokenizer tokenizer.json --out runs/mon_run

`prepared/` est la sortie de `python -m ftb.prepare` (train.bin + data_meta.json).
Sortie : runs/mon_run/ckpt.pt, train_log.jsonl, samples.txt, tokenizer.json, data_meta.json.

Options autorisées pour un run officiel (elles ne changent pas le calcul) :
  --micro-batch N   accumulation de gradient si la mémoire manque (batch effectif inchangé)
  --compile         torch.compile (plus rapide sur T4 et GPU récents, pas sur P100)
  --resume          reprend un run interrompu (Colab qui coupe) depuis le dernier state.pt
Option de mise au point (le run n'est alors PAS soumissible) :
  --debug-budget N  s'arrête après N tokens au lieu de 100M
"""

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers
import torch

from ftb import KIT_VERSION, VALIDATION_PATH, load_config
from ftb.data import TokenStream, dataset_sha256
from ftb.evaluate import default_device, evaluate_bpb, read_jsonl
from ftb.model import build_model
from ftb.tokenizer import load_tokenizer


def lr_at(step, tr):
    """Warmup linéaire puis décroissance cosinus jusqu'à min_lr au dernier pas."""
    if step < tr["warmup_steps"]:
        return tr["lr"] * (step + 1) / tr["warmup_steps"]
    progress = (step - tr["warmup_steps"]) / max(1, tr["max_steps"] - tr["warmup_steps"])
    return tr["min_lr"] + 0.5 * (tr["lr"] - tr["min_lr"]) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, tr, device):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": tr["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    extra = {"fused": True} if device.startswith("cuda") else {}
    return torch.optim.AdamW(groups, lr=tr["lr"], betas=(tr["beta1"], tr["beta2"]), eps=1e-8,
                             **extra)


def pick_precision(device):
    """float16 + GradScaler sur tout GPU CUDA (T4, P100, L4, A100...) : même calcul que le
    ré-entraînement de vérification de l'orga, qui se fait sur T4. float32 sur CPU et MPS."""
    if device.startswith("cuda"):
        return torch.float16, True
    return torch.float32, False


def generate_samples(model, tok, device, n=5, max_new_tokens=200, seed=0):
    g_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model.eval()
    out = []
    for _ in range(n):
        idx = torch.tensor([[tok.eot_id]], device=device)
        ids = model.generate(idx, max_new_tokens, temperature=0.8, top_k=200,
                             eot_id=tok.eot_id)[0, 1:].tolist()
        if tok.eot_id in ids:
            ids = ids[:ids.index(tok.eot_id)]
        out.append(tok.decode(ids))
    torch.random.set_rng_state(g_state)
    model.train()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ftb.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dossier produit par ftb.prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=default_device())
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--debug-budget", type=int, default=None)
    p.add_argument("--validation", default=str(VALIDATION_PATH))
    args = p.parse_args(argv)

    cfg = load_config()
    tr = dict(cfg["train"])
    official = args.debug_budget is None
    if not official:
        tr["max_steps"] = max(1, math.ceil(args.debug_budget / tr["tokens_per_step"]))
        tr["warmup_steps"] = min(tr["warmup_steps"], max(1, tr["max_steps"] // 10))
        tr["eval_every_steps"] = max(1, tr["max_steps"] // 4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    with open(data_dir / "data_meta.json", encoding="utf-8") as f:
        data_meta = json.load(f)

    tok = load_tokenizer(args.tokenizer)
    if data_meta["tokenizer_sha256"] != tok.sha256:
        sys.exit("ERREUR : train.bin a été préparé avec un autre tokenizer (relance ftb.prepare)")
    print("vérification de l'empreinte du dataset...", file=sys.stderr)
    if dataset_sha256(data_dir / "train.bin") != data_meta["dataset_sha256"]:
        sys.exit("ERREUR : train.bin ne correspond pas à data_meta.json (relance ftb.prepare)")

    device = args.device
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    model = build_model(cfg["model"]).to(device)
    n_params = model.num_params()
    opt = make_optimizer(model, tr, device)
    amp_dtype, use_scaler = pick_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_model = torch.compile(model) if args.compile else model

    B, T = tr["batch_size"], tr["seq_len"]
    micro = args.micro_batch or B
    if B % micro:
        sys.exit(f"--micro-batch doit diviser {B}")
    accum = B // micro
    stream = TokenStream(data_dir / "train.bin", B, T, tr["seed"], tr["shuffle_window_steps"],
                         total_steps=tr["max_steps"])
    val_docs = read_jsonl(args.validation)

    start_step = 0
    prior_time = 0.0
    resumed_from = []
    state_path = out / "state.pt"
    run_id = {"dataset_sha256": data_meta["dataset_sha256"], "tokenizer_sha256": tok.sha256,
              "train_config": json.dumps(tr, sort_keys=True), "kit_version": KIT_VERSION}
    if (out / "ckpt.pt").exists() and not state_path.exists():
        sys.exit(f"ERREUR : {out} contient déjà un run terminé. Choisis un autre --out "
                 f"(ou supprime ce dossier pour recommencer).")
    if state_path.exists() and not args.resume:
        sys.exit(f"ERREUR : {out} contient un run interrompu : ajoute --resume pour le reprendre, "
                 f"ou choisis un autre --out.")
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st["run_id"] != run_id:
            sys.exit("ERREUR : le run interrompu a été lancé avec un autre dataset, tokenizer ou "
                     "configuration. Reprise impossible : choisis un autre --out.")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_step = st["step"]
        prior_time = st["train_time_s"]
        resumed_from = st["resumed_from"] + [start_step]
        torch.random.set_rng_state(st["rng_cpu"])
        print(f"reprise au pas {start_step}", file=sys.stderr)
    # Journal : on ne garde que les lignes antérieures au point de reprise
    log_path = out / "train_log.jsonl"
    kept = []
    if start_step and log_path.exists():
        kept = [l for l in open(log_path, encoding="utf-8") if json.loads(l)["step"] <= start_step]

    print(f"modèle : {n_params / 1e6:.2f}M paramètres | {tr['max_steps']} pas x "
          f"{tr['tokens_per_step']:,} tokens = {tr['max_steps'] * tr['tokens_per_step']:,} tokens | "
          f"device {device} {amp_dtype} | dataset {data_meta['n_tokens']:,} tokens "
          f"({data_meta['passes_over_data']} passes)", file=sys.stderr)

    log = open(log_path, "w", encoding="utf-8")
    log.writelines(kept)
    t0 = time.time()
    t_last = t0
    model.train()
    for step in range(start_step, tr["max_steps"]):
        lr = lr_at(step, tr)
        for g in opt.param_groups:
            g["lr"] = lr
        xy = torch.from_numpy(stream.batch(step))
        loss_sum = 0.0
        for m in range(accum):
            chunk = xy[m * micro:(m + 1) * micro].to(device, non_blocking=True)
            x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                _, loss = train_model(x, y)
            scaler.scale(loss / accum).backward()
            loss_sum += loss.item() / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        done = step + 1
        rec = None
        if done % 10 == 0 or done == tr["max_steps"] or step == start_step:
            now = time.time()
            elapsed = now - t0
            tps = (done - start_step) * tr["tokens_per_step"] / max(elapsed, 1e-9)
            eta = (tr["max_steps"] - done) * tr["tokens_per_step"] / max(tps, 1e-9)
            rec = {"step": done, "tokens": done * tr["tokens_per_step"], "loss": round(loss_sum, 4),
                   "lr": lr, "grad_norm": round(gnorm, 4), "tokens_par_s": round(tps),
                   "temps_s": round(elapsed, 1)}
            if now - t_last > 15 or done == tr["max_steps"]:
                print(f"pas {done}/{tr['max_steps']} | loss {loss_sum:.4f} | lr {lr:.2e} | "
                      f"{tps / 1e3:.1f}k tok/s | reste ~{eta / 60:.0f} min", file=sys.stderr)
                t_last = now
        if done % tr["eval_every_steps"] == 0 and done != tr["max_steps"]:
            subset = val_docs[::4]
            res = evaluate_bpb(model, tok, subset, device=device, dtype=amp_dtype)
            rec = rec or {"step": done, "tokens": done * tr["tokens_per_step"]}
            rec["val_bpb_quart"] = round(res["bpb"], 4)
            print(f"pas {done} | BPB validation (1/4 du jeu) : {res['bpb']:.4f}", file=sys.stderr)
            model.train()
            if rec:
                log.write(json.dumps(rec) + "\n")
                log.flush()
                rec = None
            tmp = state_path.with_suffix(".tmp")  # écriture atomique : une coupure ne corrompt rien
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": done, "run_id": run_id,
                        "train_time_s": prior_time + time.time() - t0,
                        "resumed_from": resumed_from,
                        "rng_cpu": torch.random.get_rng_state()}, tmp)
            os.replace(tmp, state_path)
        if rec:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    train_time = prior_time + time.time() - t0
    # Évaluation finale en float32, exactement comme ftb.evaluate
    res = evaluate_bpb(model, tok, val_docs, device=device, progress=True)
    log.write(json.dumps({"step": tr["max_steps"], "val_bpb": res["bpb"],
                          "val_bpb_par_genre": res["bpb_par_genre"]}) + "\n")
    log.close()
    print(f"BPB validation : {res['bpb']:.4f}  {json.dumps(res['bpb_par_genre'])}", file=sys.stderr)

    meta = {
        "kit_version": KIT_VERSION,
        "official": official,
        "steps": tr["max_steps"],
        "tokens_seen": tr["max_steps"] * tr["tokens_per_step"],
        "train_config": {k: v for k, v in tr.items()},
        "dataset_sha256": data_meta["dataset_sha256"],
        "dataset_n_tokens": data_meta["n_tokens"],
        "tokenizer_sha256": tok.sha256,
        "val_bpb": res["bpb"],
        "val_bpb_par_genre": res["bpb_par_genre"],
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "device": torch.cuda.get_device_name() if device.startswith("cuda") else device,
        "precision": str(amp_dtype),
        "torch": str(torch.__version__),
        "tokenizers": tokenizers.__version__,
        "python": platform.python_version(),
        "resumed_from_steps": resumed_from,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    torch.save({"model": model.state_dict(), "model_config": cfg["model"], "meta": meta},
               out / "ckpt.pt")
    if Path(args.tokenizer).resolve() != (out / "tokenizer.json").resolve():
        shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    shutil.copyfile(data_dir / "data_meta.json", out / "data_meta.json")
    samples = generate_samples(model, tok, device)
    with open(out / "samples.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            f.write(f"=== échantillon {i} ===\n{s}\n\n")
    if state_path.exists():
        state_path.unlink()
    print(f"terminé : {out / 'ckpt.pt'} ({train_time / 60:.1f} min)", file=sys.stderr)


if __name__ == "__main__":
    main()
