"""Tests de minigpt/train_moliere.py : uv run pytest -q tests/test_minigpt_train.py"""
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from minigpt import train_moliere as tm
from minigpt.bpe import BPETokenizer
from minigpt.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent  # racine du dépôt


def test_lr_schedule():
    """Warmup de 0 à LR, puis cosinus qui descend jusqu'à MIN_LR à la fin du budget."""
    assert tm.lr_at(0) == 0
    assert math.isclose(tm.lr_at(tm.WARMUP), tm.LR)
    assert math.isclose(tm.lr_at(1), tm.MIN_LR)
    after_warmup = [tm.lr_at(tm.WARMUP + k * (1 - tm.WARMUP) / 10) for k in range(11)]
    assert after_warmup == sorted(after_warmup, reverse=True)  # ne fait que descendre


def test_total_nats_counts_every_token_once():
    """Un modèle aux logits tous nuls donne p = 1/V à chaque token : chaque token coûte ln V nats.
    total_nats doit donc valoir (N − 1) × ln V : tous les tokens sauf le 1er, chacun une seule fois,
    que la longueur tombe juste sur les fenêtres ou non."""
    torch.manual_seed(0)
    vocab = 50
    model = GPT(GPTConfig(vocab_size=vocab, block_size=tm.BLOCK_SIZE, n_layer=1, n_head=2, n_embd=16))
    with torch.no_grad():
        model.lm_head.weight.zero_()  # matrice partagée avec tok_emb : logits = 0 partout
    for n in [2, tm.BLOCK_SIZE + 1, 2 * tm.BLOCK_SIZE + 1, 300]:
        ids = torch.randint(vocab, (n,))
        assert math.isclose(tm.total_nats(model, ids), (n - 1) * math.log(vocab), rel_tol=1e-5)


def test_script_tiny_run(tmp_path):
    """Le script complet en mode minuscule : corpus de quelques Ko, BPE de 300 tokens, 3 pas."""
    corpus = (ROOT / "corpus/moliere_train.txt").read_text(encoding="utf-8")  # lecture seule
    # Deux mini-corpus au format du kit : des pièces séparées par une ligne « ===== ».
    train = "\n=====\n".join(["PIÈCE UN\n\n" + corpus[:3000], "PIÈCE DEUX\n\n" + corpus[50_000:53_000]])
    val = "\n=====\n".join(["PIÈCE TROIS\n\n" + corpus[100_000:101_500], "PIÈCE QUATRE\n\n" + corpus[150_000:151_500]])
    (tmp_path / "train.txt").write_text(train, encoding="utf-8")
    (tmp_path / "val.txt").write_text(val, encoding="utf-8")
    out = tmp_path / "results"

    run = subprocess.run(
        [sys.executable, "-m", "minigpt.train_moliere", "--train", str(tmp_path / "train.txt"),
         "--val", str(tmp_path / "val.txt"), "--out", str(out), "--vocab", "300", "--steps", "3"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stderr
    assert "PIÈCE UN, PIÈCE DEUX" in run.stdout  # les pièces de chaque fichier sont affichées
    assert "PIÈCE TROIS, PIÈCE QUATRE" in run.stdout

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["steps"] == 3 and m["seed"] == tm.SEED and m["vocab_size"] == 300 and m["fixed_steps"]
    assert [h[0] for h in m["history"]] == [0, 3]  # une mesure au départ, une à la fin
    assert m["n_val_bytes"] == len(val.encode("utf-8"))  # le BPB divise par les octets de TOUT le texte
    for key in ["bpb_val_transformer", "bpb_val_bigram_bytes", "bpb_val_bigram_tokens"]:
        assert 0 < m[key] < 20
    assert m["n_params"] == GPT(GPTConfig(vocab_size=300, block_size=tm.BLOCK_SIZE, **tm.MODEL)).num_params()

    assert (out / "loss_curve.png").read_bytes()[:4] == b"\x89PNG"  # en-tête d'un fichier PNG
    samples = (out / "samples.md").read_text(encoding="utf-8")
    assert all(f"« {p} »" in samples for p in tm.PROMPTS)  # 5 échantillons, avec leur amorce
    report = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "Bigramme d'octets" in report and "Interprétation" in report and "Pont" not in report
    assert "`uv run python -m minigpt.train_moliere --steps 3`" in report  # la commande pour rejouer
    assert "légèrement optimiste" in report  # n_embd et lr ont été réglés sur la validation

    # Le tokenizer et les poids sauvegardés se rechargent.
    tok = BPETokenizer().load(out / "tokenizer.json")
    assert tok.decode(tok.encode(val)) == val
    checkpoint = torch.load(out / "model.pt")
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])


def test_fixed_steps_same_weights():
    """Même graine + même nombre de pas = mêmes poids au bit près : le run livré se rejoue."""
    def run():
        torch.manual_seed(tm.SEED)  # comme main() : init des poids, batchs et dropout tirés de cette graine
        model = GPT(GPTConfig(vocab_size=50, block_size=tm.BLOCK_SIZE, n_layer=1, n_head=2, n_embd=16))
        ids = torch.randint(50, (2000,), generator=torch.Generator().manual_seed(0))
        history, steps = tm.train_model(model, ids, ids, None, 5)  # 5 pas, pas de chronomètre
        return model.state_dict(), history, steps

    (w1, h1, s1), (w2, h2, s2) = run(), run()
    assert s1 == s2 == 5 and h1 == h2
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


RESULTS = ROOT / "minigpt/results"
FICHES = ROOT / "docs/comprendre"


def test_delivered_run_is_replayable():
    """Le run livré dans minigpt/results/ a un nombre de pas fixé, celui que rejoue la commande par défaut."""
    m = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    assert m["fixed_steps"] and m["steps"] == tm.STEPS and m["seed"] == tm.SEED
    assert f"`uv run python -m minigpt.train_moliere --steps {tm.STEPS}`" in (RESULTS / "RESULTS.md").read_text(encoding="utf-8")


def test_fiche_matches_delivered_run():
    """Les chiffres de la fiche d'entraînement sont ceux de metrics.json (le jury peut les vérifier)."""
    m = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    fiche = (FICHES / "minigpt_entrainement.md").read_text(encoding="utf-8")
    best_step, _, best_val = min(m["history"], key=lambda h: h[2])
    for number in [f"mini-GPT {tm.fr(m['bpb_val_transformer'])}", tm.fr(m["steps"], 0) + " pas",
                   tm.fr(m["final_val_loss"] - m["final_train_loss"]) + " nat", tm.fr(best_val, 3),
                   f"au pas {tm.fr(best_step, 0)}", tm.fr(m["final_val_loss"], 3)]:
        assert number in fiche, number
    if m["final_val_loss"] > best_val:  # la loss de validation finit au-dessus de son minimum
        assert "ne remonte pas" not in fiche
    assert "optimiste" in fiche  # n_embd et lr réglés sur la validation


def test_fiches_explain_real_lines():
    """Chaque ligne citée dans « Explique cette ligne » existe telle quelle dans le code."""
    modules = {"bigram": "bigram", "micrograd": "micrograd", "gpt": "gpt", "bpe": "bpe", "entrainement": "train_moliere"}
    for fiche, module in modules.items():
        text = (FICHES / f"minigpt_{fiche}.md").read_text(encoding="utf-8")
        section = text.split("Explique cette ligne")[1].split("## 5")[0]
        code = (ROOT / f"minigpt/{module}.py").read_text(encoding="utf-8")
        cited = [line.split("`")[1] for line in section.splitlines() if line.startswith(("1.", "2.", "3.", "4.", "5.", "|"))
                 and line.count("`") >= 2]
        assert len(cited) >= 5, fiche
        for snippet in cited:
            assert snippet.split("…")[0] in code, f"{fiche} : {snippet}"
