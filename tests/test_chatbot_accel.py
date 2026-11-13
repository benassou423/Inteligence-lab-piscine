"""Calcul accéléré (chatbot/accel.py) et reprise de l'affinage — tout sur CPU, modèles minuscules.

Ce que ces tests prouvent sans carte NVIDIA :
1. sur CPU en float32, pretrain.py et finetune.py donnent les MÊMES poids, au bit près, que les
   versions d'avant ce chantier (copies dans data/stages/chatbot_20261113/backup_rtx/) ;
2. le chemin précision mixte (autocast bfloat16) tourne : la loss baisse, pas de NaN, checkpoint
   relisible en float32 ; le GradScaler (fp16) aussi, PyTorch l'acceptant sur CPU ;
3. reprise de l'affinage (10 + 10 pas = 20 pas d'un trait), micro-lots, repli de --compile,
   console et remplacement de fichier façon Windows.
Ce qui reste invérifiable ici : la vitesse et la mémoire réelles sur CUDA, TF32, torch.compile
réussi sur la RTX 5090, l'épinglage mémoire (pin_memory).
"""
import importlib.util
import io
import json
import math
import os
import signal
import sys
from pathlib import Path

import pytest
import torch

from chatbot import accel as A
from chatbot import finetune as FT
from chatbot import pretrain as P
from ftb.model import build_model, load_checkpoint
from pipeline.io import write_docs

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "submissions/2026-11-11_153839/tokenizer.json"
BACKUP = ROOT / "data/stages/chatbot_20261113/backup_rtx"
needs_tok = pytest.mark.skipif(not TOKENIZER.exists(), reason="tokenizer absent")

TINY = {"vocab_size": 8192, "context_length": 64, "d_model": 32, "n_layer": 2, "n_head": 2,
        "mlp_ratio": 4, "dropout": 0.0, "tie_embeddings": True}
TINY_JSON = '{"d_model":32,"n_layer":1,"n_head":2,"context_length":64}'

LINES = ["Au voleur ! au voleur ! à l'assassin !", "Mon argent, mon pauvre argent.",
         "Je suis malade, appelez le médecin.", "Il faut dire la vérité, toujours.",
         "Que voulez-vous dire, mon père ?", "Vous vous moquez de moi."]
PROSE = ("Le château se dressait au bord de la rivière, entouré de vieux chênes que le vent "
         "faisait gémir chaque soir. Les paysans du village racontaient des histoires ; personne "
         "ne les croyait vraiment.\n\nUn matin d'automne, le jeune comte revint de la ville.")


def scenes(n, offset=0):
    names = [("HARPAGON", "CLÉANTE"), ("ARGAN", "TOINETTE")]
    return [{"id": f"s{i}", "doc_id": "f", "is_moliere": True,
             "turns": [{"speaker": names[i % 2][k % 2], "text": LINES[(i + k) % 6]}
                       for k in range(6)]}
            for i in range(offset, offset + n)]


def dialogues(n, offset=0):
    """Contrat « tirets » : A commence, puis A et B alternent."""
    return [{"id": f"d{i}", "doc_id": "f",
             "turns": [{"speaker": "AB"[k % 2], "text": LINES[(i + k) % 6]} for k in range(4 + i % 3)]}
            for i in range(offset, offset + n)]


def load_old(name):
    """Module d'avant ce chantier, chargé depuis la sauvegarde (absent sur une autre machine)."""
    path = BACKUP / f"{name}.py"
    if not path.exists():
        pytest.skip(f"sauvegarde {path} absente")
    spec = importlib.util.spec_from_file_location(f"ancien_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_weights(a, b):
    sa = torch.load(a, weights_only=True)["model"]
    sb = torch.load(b, weights_only=True)["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def log_of(out):
    return [json.loads(l) for l in (Path(out) / "log.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------------------------
# Fichiers d'essai


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("accel")
    write_docs(d / "a.jsonl.zst", scenes(30))
    write_docs(d / "b.jsonl.zst", scenes(10, offset=30))
    write_docs(d / "val.jsonl.zst", scenes(6, offset=100))
    write_docs(d / "dial.jsonl.zst", dialogues(30))
    write_docs(d / "dial_val.jsonl.zst", dialogues(6, offset=100))
    write_docs(d / "prose.jsonl.zst", [{"id": f"p{i}", "text": f"Chapitre {i}.\n\n" + PROSE}
                                       for i in range(200)])
    return d


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER.exists():
        pytest.skip("tokenizer absent")
    return FT.open_tokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def init_ckpt(files, tok):
    torch.manual_seed(0)
    path = files / "init.pt"
    torch.save({"model": build_model(TINY).state_dict(), "model_config": TINY,
                "meta": {"tokenizer_sha256": tok.sha256}}, path)
    return path


@pytest.fixture(scope="module")
def pre_data(files, tok):
    d = files / "pre"
    ref = P.build_reference()
    P.prepare_source("prose", [str(files / "prose.jsonl.zst")], None, d, tok, ref)
    P.prepare_source("theatre", [str(files / "a.jsonl.zst")], None, d, tok, ref)
    return d


def ft_args(files, init, out, *extra, fmt="repliques", steps=20, batch=8):
    data = ([files / "a.jsonl.zst", files / "b.jsonl.zst"] if fmt == "repliques"
            else [files / "dial.jsonl.zst"])
    val = files / ("val.jsonl.zst" if fmt == "repliques" else "dial_val.jsonl.zst")
    return ["--init", str(init), "--tokenizer", str(TOKENIZER), "--format", fmt,
            "--data", *map(str, data), "--val", str(val), "--out", str(out),
            "--tokens", str(batch * 32 * steps), "--batch-size", str(batch), "--seq-len", "32",
            "--lr", "3e-3", "--min-lr", "3e-4", "--warmup-steps", "2", "--eval-every", "5",
            "--device", "cpu", *extra]


def pre_args(data, files, out, *extra, steps=5):
    return ["train", "--data", str(data), "--weights", "prose=0.7", "theatre=0.3",
            "--model-config", TINY_JSON, "--tokenizer", str(TOKENIZER),
            "--val", str(files / "val.jsonl.zst"),
            "--out", str(out), "--tokens", str(8 * 32 * steps), "--batch-size", "8",
            "--micro-batch", "4", "--seq-len", "32", "--lr", "1e-2", "--min-lr", "1e-3",
            "--warmup-steps", "1", "--eval-every", "5", "--device", "cpu", *extra]


# ---------------------------------------------------------------------------------------------
# 1. Mac / CPU : rien ne change, au bit près


@needs_tok
def test_pretrain_cpu_fp32_bit_identical_to_old_module(tmp_path, files, pre_data):
    old = load_old("pretrain")
    old.main(pre_args(pre_data, files, tmp_path / "ancien"))
    P.main(pre_args(pre_data, files, tmp_path / "nouveau"))  # précision auto -> fp32 sur CPU
    for name in ("best.pt", "last.pt"):
        assert same_weights(tmp_path / "ancien" / name, tmp_path / "nouveau" / name), name
    la, lb = log_of(tmp_path / "ancien"), log_of(tmp_path / "nouveau")
    assert [(r["step"], r["train_loss"], r["val_loss"]) for r in la] == \
           [(r["step"], r["train_loss"], r["val_loss"]) for r in lb]
    # Témoin : la comparaison est sensible (en bfloat16, les poids ne sont plus identiques).
    P.main(pre_args(pre_data, files, tmp_path / "bf16", "--precision", "bf16"))
    assert not same_weights(tmp_path / "ancien" / "last.pt", tmp_path / "bf16" / "last.pt")
    rep = json.loads((tmp_path / "nouveau" / "report.json").read_text(encoding="utf-8"))
    assert rep["precision"] == "fp32" and rep["environment"]["grad_scaler"] is False
    assert rep["environment"]["compile"]["status"] == "désactivé"


@needs_tok
@pytest.mark.parametrize("fmt,extra", [("repliques", ["--dropout", "0.1"]), ("tirets", [])])
def test_finetune_cpu_fp32_bit_identical_to_old_module(tmp_path, files, init_ckpt, fmt, extra):
    """5 pas, dans les deux formats ; dropout 0,1 : les tirages aléatoires doivent aussi suivre."""
    old = load_old("finetune")
    old.main(ft_args(files, init_ckpt, tmp_path / "ancien", *extra, fmt=fmt, steps=5))
    FT.main(ft_args(files, init_ckpt, tmp_path / "nouveau", *extra, fmt=fmt, steps=5))
    for name in ("best.pt", "last.pt"):
        assert same_weights(tmp_path / "ancien" / name, tmp_path / "nouveau" / name), name
    assert [(r["step"], r["train_loss"], r["val_loss"]) for r in log_of(tmp_path / "ancien")] == \
           [(r["step"], r["train_loss"], r["val_loss"]) for r in log_of(tmp_path / "nouveau")]


def test_auto_precision_per_device(monkeypatch):
    assert A.resolve_precision("auto", "cpu") == "fp32"
    assert A.resolve_precision("auto", "mps") == "fp32"
    monkeypatch.setattr(A, "bf16_native", lambda: True)
    assert A.resolve_precision("auto", "cuda") == "bf16"
    assert A.resolve_precision("auto", "cuda:1") == "bf16"
    monkeypatch.setattr(A, "bf16_native", lambda: False)  # carte ancienne (T4) : fp16 + GradScaler
    assert A.resolve_precision("auto", "cuda") == "fp16"
    with pytest.raises(SystemExit, match="bfloat16"):
        A.resolve_precision("bf16", "cuda")
    with pytest.raises(SystemExit):
        A.resolve_precision("fp8", "cpu")
    acc = A.Accel("cpu")
    assert acc.precision == "fp32" and acc.scaler is None and not acc.pin and not acc.tf32
    assert type(acc.autocast()).__name__ == "nullcontext"  # pas même un autocast désactivé
    x = torch.zeros(2, 3, dtype=torch.long)
    assert acc.host(x) is x and acc.to_device(x) is x


# ---------------------------------------------------------------------------------------------
# 2. Précision mixte sur CPU


@needs_tok
def test_bf16_autocast_pretrain_learns_and_checkpoint_is_float32(tmp_path, files, pre_data):
    out = tmp_path / "bf16"
    rep = P.main(pre_args(pre_data, files, out, "--precision", "bf16", steps=20))
    log = log_of(out)
    assert all(math.isfinite(r["val_loss"]) for r in log)
    assert log[-1]["val_loss"] < log[0]["val_loss"] - 1.0  # la loss baisse nettement
    assert rep["precision"] == "bf16" and rep["environment"]["autocast_dtype"] == "torch.bfloat16"
    model, ckpt = load_checkpoint(out / "best.pt")  # relu par le kit, en float32
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(v.dtype == torch.float32 for v in ckpt["model"].values())
    assert ckpt["meta"]["precision"] == "bf16" and ckpt["meta"]["environment"]["device"] == "cpu"


@needs_tok
def test_bf16_finetune_tirets_learns(tmp_path, files, init_ckpt):
    out = tmp_path / "ft_bf16"
    rep = FT.main(ft_args(files, init_ckpt, out, "--precision", "bf16", fmt="tirets"))
    log = log_of(out)
    assert all(math.isfinite(r["val_loss"]) for r in log) and log[-1]["val_loss"] < log[0]["val_loss"]
    assert rep["precision"] == "bf16" and rep["format"] == "tirets-v1"
    _, ckpt = load_checkpoint(out / "best.pt")
    assert ckpt["meta"]["precision"] == "bf16" and ckpt["meta"]["finetune"]


def test_grad_scaler_skips_overflow_then_recovers():
    """fp16 : un gradient infini fait SAUTER le pas (poids intacts) et baisse le facteur."""
    torch.manual_seed(0)
    acc = A.Accel("cpu", "fp16")
    assert acc.scaler is not None and acc.describe()["grad_scaler"]
    lin = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    before = [p.detach().clone() for p in lin.parameters()]
    scale0 = acc.scaler.get_scale()
    with acc.autocast():
        out = lin(torch.full((2, 4), 1e4))
    acc.backward(out.float().pow(2).sum())  # ×65 536 : déborde en float16
    acc.unscale(opt)
    acc.step(opt)
    assert all(torch.equal(a, b) for a, b in zip(before, lin.parameters()))  # pas sauté
    assert acc.scaler.get_scale() < scale0
    opt.zero_grad()
    for _ in range(3):  # petite loss : les pas reprennent
        with acc.autocast():
            out = lin(torch.ones(2, 4))
        acc.backward(out.float().mean())
        acc.unscale(opt)
        acc.step(opt)
        opt.zero_grad()
    assert not all(torch.equal(a, b) for a, b in zip(before, lin.parameters()))


def test_diverged_tolerates_scaler_skips():
    plain, fp16 = A.Accel("cpu", "fp32"), A.Accel("cpu", "fp16")
    assert FT.diverged(float("nan"), 1.0, plain, 0) == (True, 0)
    assert FT.diverged(2.0, float("inf"), plain, 0)[0]  # fp32 : gradient infini = divergence
    assert FT.diverged(2.0, float("inf"), fp16, 0) == (False, 1)  # fp16 : pas sauté, normal
    assert FT.diverged(2.0, 1.0, fp16, 7) == (False, 0)
    assert FT.diverged(2.0, float("inf"), fp16, FT.MAX_SCALER_SKIPS)[0]  # sauté sans fin : arrêt
    assert FT.diverged(float("nan"), 1.0, fp16, 0)[0]


class FixedSampler:
    """Lots aléatoires mais reproductibles, pour appeler train_step sans fichiers."""

    def __init__(self, seq=16):
        self.g, self.seq = torch.Generator().manual_seed(0), seq

    def batch(self, n):
        return torch.randint(0, 8192, (n, self.seq + 1), generator=self.g)


def test_fp16_nan_loss_skipped_then_training_continues():
    """fp16 : une loss NaN fait sauter le pas (poids intacts) ; le pas SUIVANT marche.

    Avant la correction, train_step faisait unscale_ puis sautait scaler.step/update : le pas
    suivant levait « unscale_() has already been called on this optimizer since the last update() ».
    """
    torch.manual_seed(0)
    model = build_model(TINY)
    acc = A.Accel("cpu", "fp16")
    sched = {"lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 1, "max_steps": 10, "weight_decay": 0.1,
             "beta1": 0.9, "beta2": 0.95}
    opt = FT.make_optimizer(model, sched, "cpu")
    calls = []

    def nan_first(x, y):
        logits, loss = model(x, y)
        calls.append(1)
        return logits, (loss * float("nan") if len(calls) == 1 else loss)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    scale0 = acc.scaler.get_scale()
    sampler = FixedSampler()
    loss, gnorm, _ = FT.train_step(model, opt, sampler, sched, 0, 4, 4, "cpu", 1.0, acc, nan_first)
    assert math.isnan(loss) and all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert acc.scaler.get_scale() < scale0  # pas sauté ET facteur baissé (update() appelé)
    assert FT.diverged(loss, gnorm, acc, 0)[0]  # l'entraînement, lui, s'arrêterait ici
    loss2, gnorm2, _ = FT.train_step(model, opt, sampler, sched, 1, 4, 4, "cpu", 1.0, acc, nan_first)
    assert math.isfinite(loss2) and math.isfinite(gnorm2)
    assert not all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@needs_tok
def test_pretrain_benchmark_stops_clearly_on_nan(tmp_path, files, pre_data, monkeypatch):
    monkeypatch.setattr(P, "train_step", lambda *a, **k: (float("nan"), float("nan"), 1e-3))
    with pytest.raises(SystemExit, match="ERREUR benchmark : loss nan"):
        P.main(pre_args(pre_data, files, tmp_path / "b", "--benchmark-steps", "2",
                        "--precision", "fp16"))


@needs_tok
def test_fp16_finetune_scaler_state_saved_and_resumed(tmp_path, files, init_ckpt):
    out = tmp_path / "fp16"
    res = FT.main(ft_args(files, init_ckpt, out, "--precision", "fp16", "--stop-at-step", "7"))
    assert res == {"status": "interrompu (--stop-at-step)", "step": 7}
    st = torch.load(out / "state.pt", weights_only=True)
    assert st["precision"] == "fp16" and st["scaler"]["scale"] > 0
    with pytest.raises(SystemExit, match="fp16"):  # autre précision : reprise refusée
        FT.main(ft_args(files, init_ckpt, out, "--resume", "--precision", "fp32"))
    rep = FT.main(ft_args(files, init_ckpt, out, "--precision", "fp16", "--resume"))
    assert rep["resumed_from_steps"] == [7] and rep["precision"] == "fp16"
    assert all(math.isfinite(r["val_loss"]) for r in log_of(out))


# ---------------------------------------------------------------------------------------------
# 3. Reprise, micro-lots, compilation


@needs_tok
def test_finetune_resume_equals_one_shot(tmp_path, files, init_ckpt):
    """10 pas + coupure + reprise + 10 pas = 20 pas d'un trait, au bit près (dropout compris)."""
    extra = ["--dropout", "0.1"]
    FT.main(ft_args(files, init_ckpt, tmp_path / "trait", *extra))
    out = tmp_path / "coupe"
    res = FT.main(ft_args(files, init_ckpt, out, *extra, "--stop-at-step", "10"))
    assert res == {"status": "interrompu (--stop-at-step)", "step": 10}
    assert FT.exit_code(res) == 3 and (out / "state.pt").exists() and not (out / "report.json").exists()
    with pytest.raises(SystemExit, match="--resume"):  # sans --resume : refus, rien d'écrasé
        FT.main(ft_args(files, init_ckpt, out, *extra))
    with pytest.raises(SystemExit, match="lr"):  # autres réglages : reprise refusée
        FT.main(ft_args(files, init_ckpt, out, *extra, "--resume", "--lr", "1e-3"))
    with pytest.raises(SystemExit, match="data"):  # autres données : reprise refusée
        args = ft_args(files, init_ckpt, out, *extra, "--resume")
        args[args.index(str(files / "b.jsonl.zst"))] = str(files / "val.jsonl.zst")
        FT.main(args)
    rep = FT.main(ft_args(files, init_ckpt, out, *extra, "--resume"))
    assert rep["resumed_from_steps"] == [10] and not (out / "state.pt").exists()
    for name in ("best.pt", "last.pt"):
        assert same_weights(tmp_path / "trait" / name, out / name), name
    a, b = log_of(tmp_path / "trait"), log_of(out)
    assert [(r["step"], r["train_loss"], r["val_loss"]) for r in a] == \
           [(r["step"], r["train_loss"], r["val_loss"]) for r in b]
    one = json.loads((tmp_path / "trait" / "report.json").read_text(encoding="utf-8"))
    assert [d["windows_sampled"] for d in one["data"]] == [d["windows_sampled"] for d in rep["data"]]
    with pytest.raises(SystemExit, match="terminé"):  # run fini : --resume n'a rien à reprendre
        FT.main(ft_args(files, init_ckpt, out, *extra, "--resume"))


@needs_tok
def test_finetune_sigint_saves_state(tmp_path, files, init_ckpt, monkeypatch):
    real = FT.train_step

    def step_then_ctrl_c(*args, **kw):
        res = real(*args, **kw)
        if args[4] == 3:
            # Ctrl-C simulé. Pas os.kill(getpid(), SIGINT) : sous Windows, os.kill avec un autre
            # signal que CTRL_C_EVENT / CTRL_BREAK_EVENT TUE le processus (TerminateProcess), donc
            # pytest. raise_signal appelle le gestionnaire Python sur toutes les plateformes.
            signal.raise_signal(signal.SIGINT)
        return res

    monkeypatch.setattr(FT, "train_step", step_then_ctrl_c)
    before = signal.getsignal(signal.SIGINT)
    res = FT.main(ft_args(files, init_ckpt, tmp_path / "sig"))
    assert res == {"status": "interrompu (SIGINT)", "step": 4}
    assert torch.load(tmp_path / "sig" / "state.pt", weights_only=True)["step"] == 4
    assert signal.getsignal(signal.SIGINT) == before  # gestionnaire d'origine remis
    assert {signal.SIGINT, signal.SIGTERM} <= set(A.StopRequest.SIGNALS)
    if hasattr(signal, "SIGBREAK"):  # Windows : Ctrl-Break
        assert signal.SIGBREAK in A.StopRequest.SIGNALS


@needs_tok
def test_finetune_micro_batch_equals_full_batch(tmp_path, files, init_ckpt):
    FT.main(ft_args(files, init_ckpt, tmp_path / "plein", steps=10))
    rep = FT.main(ft_args(files, init_ckpt, tmp_path / "micro", "--micro-batch", "2", steps=10))
    assert rep["micro_batch"] == 2
    a = torch.load(tmp_path / "plein" / "last.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "micro" / "last.pt", weights_only=True)["model"]
    for k in a:  # même pas, seul l'ordre des additions change : écart d'arrondi seulement
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    la, lb = log_of(tmp_path / "plein"), log_of(tmp_path / "micro")
    assert [r["val_loss"] for r in la] == pytest.approx([r["val_loss"] for r in lb], abs=1e-4)
    with pytest.raises(SystemExit, match="micro-batch"):
        FT.main(ft_args(files, init_ckpt, tmp_path / "x", "--micro-batch", "3"))


def test_compile_failure_falls_back(monkeypatch, capsys):
    torch.manual_seed(0)
    model = build_model(TINY)

    def broken(m):
        raise RuntimeError("pas de compilateur C++ ici")

    monkeypatch.setattr(torch, "compile", broken)
    acc = A.Accel("cpu", compile=True)
    assert acc.compile(model, (2, 8)) is model
    assert acc.compile_status.startswith("échec") and "pas de compilateur" in acc.compile_error
    assert "SANS compilation" in capsys.readouterr().err

    # Échec tardif (au premier appel) : SafeForward bascule sur le modèle normal.
    def lazy_failure(m):
        def fn(*a, **k):
            raise RuntimeError("Triton absent")
        return fn

    monkeypatch.setattr(torch, "compile", lazy_failure)
    acc = A.Accel("cpu", compile=True)
    fwd = acc.compile(model, None)  # sans essai : l'erreur arrivera au premier vrai appel
    assert acc.compile_status == "actif"
    x = torch.zeros(2, 8, dtype=torch.long)
    logits, loss = fwd(x, x)
    assert torch.equal(logits, model(x)[0]) and acc.compile_status.startswith("échec")
    assert all(p.grad is None for p in model.parameters())


@needs_tok
def test_compile_failure_does_not_stop_training(tmp_path, files, init_ckpt, monkeypatch):
    """--compile qui échoue : même calcul qu'sans --compile (essai jeté, aléa remis en l'état)."""
    def broken(m):
        def fn(*a, **k):
            raise RuntimeError("inductor indisponible")
        return fn

    monkeypatch.setattr(torch, "compile", broken)
    FT.main(ft_args(files, init_ckpt, tmp_path / "sans", "--dropout", "0.1", steps=5))
    rep = FT.main(ft_args(files, init_ckpt, tmp_path / "avec", "--dropout", "0.1", "--compile",
                          steps=5))
    assert rep["environment"]["compile"]["requested"] is True
    assert rep["environment"]["compile"]["status"].startswith("échec")
    assert same_weights(tmp_path / "sans" / "last.pt", tmp_path / "avec" / "last.pt")


def flaky_compile(m):
    """torch.compile simulé qui réussit l'essai de démarrage puis échoue au pas suivant."""
    calls = []

    def fn(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("recompilation impossible")
        return m(*a, **k)
    return fn


@needs_tok
def test_late_compile_failure_is_in_checkpoint_meta(tmp_path, files, init_ckpt, pre_data,
                                                    monkeypatch):
    """Échec de --compile EN COURS de run : last.pt dit « échec » comme report.json."""
    monkeypatch.setattr(torch, "compile", flaky_compile)
    rep = FT.main(ft_args(files, init_ckpt, tmp_path / "ft", "--compile", steps=5))
    meta = torch.load(tmp_path / "ft" / "last.pt", weights_only=True)["meta"]
    assert rep["environment"]["compile"]["status"].startswith("échec")
    assert meta["environment"]["compile"]["status"].startswith("échec")
    rep = P.main(pre_args(pre_data, files, tmp_path / "pre", "--compile"))
    meta = torch.load(tmp_path / "pre" / "last.pt", weights_only=True)["meta"]
    assert meta["environment"]["compile"]["status"].startswith("échec")


# ---------------------------------------------------------------------------------------------
# 4. Pré-entraînement : reprise et précision


@needs_tok
def test_pretrain_resume_checks_precision_and_accepts_old_state(tmp_path, files, pre_data):
    out = tmp_path / "pre"
    res = P.main(pre_args(pre_data, files, out, "--precision", "bf16", "--stop-at-step", "7",
                          steps=20))
    assert res["status"].startswith("interrompu")
    with pytest.raises(SystemExit, match="bf16"):
        P.main(pre_args(pre_data, files, out, "--resume", steps=20))  # auto = fp32 sur CPU
    rep = P.main(pre_args(pre_data, files, out, "--resume", "--precision", "bf16", steps=20))
    assert rep["resumed_from_steps"] == [7] and rep["precision"] == "bf16"

    # Un state.pt écrit AVANT ce chantier (sans précision ni GradScaler) = fp32 : accepté.
    out2 = tmp_path / "ancien_etat"
    P.main(pre_args(pre_data, files, out2, "--stop-at-step", "7", steps=20))
    st = torch.load(out2 / "state.pt", weights_only=True)
    for key in ("precision", "scaler", "cuda_rng"):
        del st[key]
    torch.save(st, out2 / "state.pt")
    rep2 = P.main(pre_args(pre_data, files, out2, "--resume", steps=20))
    assert rep2["resumed_from_steps"] == [7] and rep2["precision"] == "fp32"


@needs_tok
def test_pretrain_benchmark_reports_precision(tmp_path, files, pre_data, capsys):
    res = P.main(pre_args(pre_data, files, tmp_path / "b", "--benchmark-steps", "2",
                          "--precision", "bf16"))
    assert res["precision"] == "bf16" and res["peak_memory_gb"] is None  # pas de CUDA ici
    assert not (tmp_path / "b").exists() and "précision bf16" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# 5. Windows : console et remplacement de fichier


def test_setup_console_reconfigures_cp1252(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(UnicodeEncodeError):
        stream.write("≈")
        stream.flush()
    A.setup_console()
    assert stream.encoding == "utf-8"
    print("reste ≈ 2 h, vocab ≠ 8192, poids ∝ taille, Molière")
    stream.flush()
    assert "reste ≈ 2 h" in raw.getvalue().decode("utf-8")


def test_stop_request_survives_broken_pipe(monkeypatch):
    """Lanceur mort : écrire dans son tube lève BrokenPipeError. L'arrêt doit quand même aboutir."""
    class Broken(io.StringIO):
        def write(self, s):
            raise BrokenPipeError(32, "tube cassé")

    monkeypatch.setattr(sys, "stdout", Broken())
    monkeypatch.setattr(sys, "stderr", Broken())
    with A.StopRequest() as stop:
        stop.handler(signal.SIGTERM, None)  # ne lève pas
        assert stop.requested == "SIGTERM"
        print("interrompu au pas 4 : état sauvé", file=sys.stderr, flush=True)  # jeté, sans erreur
        sink = sys.stderr
    assert not isinstance(sink, Broken)
    sink.close()


def test_replace_file_retries_then_explains(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("nouveau", encoding="utf-8")
    dst.write_text("ancien", encoding="utf-8")
    real, calls = os.replace, []

    def locked_twice(a, b):  # antivirus qui garde le fichier deux instants
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError("fichier utilisé par un autre processus")
        real(a, b)

    monkeypatch.setattr(A.os, "replace", locked_twice)
    A.replace_file(src, dst, wait=0)
    assert dst.read_text(encoding="utf-8") == "nouveau" and len(calls) == 3

    def always_locked(a, b):
        raise PermissionError("verrou")

    src.write_text("encore", encoding="utf-8")
    monkeypatch.setattr(A.os, "replace", always_locked)
    with pytest.raises(PermissionError, match="ouvert par un autre programme"):
        A.replace_file(src, dst, tries=3, wait=0)


def test_describe_is_safe_for_checkpoint_meta(tmp_path):
    """La description de la machine se relit avec weights_only=True (pas d'objet torch)."""
    info = A.Accel("cpu").describe()
    assert info["device"] == "cpu" and info["torch"] and "gpu" not in info
    torch.save({"meta": info}, tmp_path / "m.pt")
    assert torch.load(tmp_path / "m.pt", weights_only=True)["meta"] == info
    json.dumps(info)
