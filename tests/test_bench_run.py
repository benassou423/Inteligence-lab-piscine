"""bench.run de bout en bout avec le kit (CPU, 1 pas) : mix → tokenizer → prepare → train → evaluate → results.csv."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from bench.common import REPO, append_csv
from bench.compare import doc_bits, paired_bootstrap
from bench.config import mix_dir, results_csv, run_dir
from bench.run import FR_SPLIT_PATTERN, RESULT_FIELDS, run, stage_train, stream_coverage

_spec = importlib.util.spec_from_file_location("bench_fx", Path(__file__).parent / "fixtures/bench/make_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

GENRES = ("encyclopedique", "litteraire", "presse", "dialogue")


@pytest.fixture(scope="module")
def moliere_tokenizer(tmp_path_factory):
    """Tokenizer réglementaire (8 192 tokens) entraîné avec le kit sur le corpus Molière du kit."""
    from ftb.tokenizer import train_hf
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_hf([str(REPO / "corpus" / "moliere_train.txt")]).save(str(path))
    return path


def _tiny_validation(path: Path) -> Path:
    docs = [{"id": f"v{i}", "genre": g, "text": f"Texte court du genre {g}. L'homme qu'il a vu hier."}
            for i, g in enumerate(GENRES)]
    path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    return path


def test_end_to_end_with_kit(tmp_path, moliere_tokenizer):
    fx.make_corpus(tmp_path / "data", layout="stage")
    val = _tiny_validation(tmp_path / "val.jsonl")
    cfg = fx.run_config(tmp_path, run="e2e", tokenizer={"path": str(moliere_tokenizer)},
                        train={"mode": "proxy", "debug_budget": 32768, "device": "cpu", "micro_batch": 16},
                        eval={"validation": str(val), "device": "cpu"})
    out = run(cfg, "all")

    data_meta = json.loads((mix_dir(cfg) / "prepared" / "data_meta.json").read_text())
    assert data_meta["n_documents"] == out["mix"]["documents"]
    ev = out["eval"]
    assert set(ev["bpb_par_genre"]) == set(GENRES) and ev["bpb"] > 0
    rows = list(csv.DictReader(open(results_csv(cfg), encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["run"] == "e2e" and row["mode"] == "proxy"
    assert row["dataset_sha256"] == data_meta["dataset_sha256"]
    assert int(row["tokens_seen"]) == 32768 and int(row["dataset_tokens"]) == data_meta["n_tokens"]
    # recopie exacte de la sortie de ftb.evaluate (et du checkpoint), pas seulement « > 0 »
    assert row["bpb_global"] == f"{ev['bpb']:.4f}"
    for g in GENRES:
        assert row[f"bpb_{g}"] == f"{ev['bpb_par_genre'][g]:.4f}"
    assert row["val_bytes_per_token"] == f"{ev['octets_par_token']:.4f}"
    assert list(rows[0]) == list(RESULT_FIELDS)
    assert "flux vu" in row["notes"]
    assert (run_dir(cfg) / "samples.txt").exists()
    # bench.compare : un « genre » par document redonne exactement le BPB global du kit
    d = doc_bits(run_dir(cfg), val, "cpu")
    assert abs(sum(d["bits"]) / sum(d["bytes"]) - ev["bpb"]) < 1e-9 and d["genre"] == list(GENRES)

    # relancer sans rien changer : tout est réutilisé, aucune nouvelle ligne
    again = run(cfg, "all")
    assert "result" not in again
    assert len(list(csv.DictReader(open(results_csv(cfg), encoding="utf-8")))) == 1


def test_fr_pretokenizer_keeps_apostrophe_and_passes_kit_rules(tmp_path):
    from tokenizers import Regex, pre_tokenizers

    from ftb.tokenizer import check, train_hf
    split = pre_tokenizers.Split(Regex(FR_SPLIT_PATTERN), behavior="isolated")
    assert [p for p, _ in split.pre_tokenize_str("l'homme qu’il")] == ["l'", "homme", " qu’", "il"]
    path = tmp_path / "fr.json"
    train_hf([str(REPO / "corpus" / "moliere_train.txt")], pattern=FR_SPLIT_PATTERN).save(str(path))
    res = check(str(path))  # règles du kit : 8 192 tokens, sans perte sur la val
    assert res["vocab_size"] == 8192 and res["lossless"]


@pytest.fixture(scope="module")
def blocks_prepared(tmp_path_factory, moliere_tokenizer):
    """Mix ordonné en 2 blocs (web puis « le meilleur »), tokenizer et train.bin, sans entraînement."""
    root = tmp_path_factory.mktemp("blocks")
    fx.make_corpus(root / "data", layout="stage")
    cfg = fx.run_config(root, run="blk", tokenizer={"path": str(moliere_tokenizer)},
                        train={"mode": "proxy", "debug_budget": 32768, "device": "cpu"},
                        mix={"order": "blocks", "sources": [
                            {"name": "web", "pipeline": "nettoye", "sources": ["commoncrawl"], "block": 0},
                            {"name": "best", "pipeline": "nettoye", "sources": ["wikipedia_fr", "gutenberg_fr"],
                             "block": 1}]})
    run(cfg, ("mix", "tokenizer", "prepare"))
    return cfg


def test_stream_coverage_flags_cut_last_block_and_wraparound(blocks_prepared):
    cfg = blocks_prepared
    n = json.loads((mix_dir(cfg) / "prepared" / "data_meta.json").read_text())["n_tokens"]
    man = json.loads((mix_dir(cfg) / "manifest.json").read_text())
    assert [s["label"] for s in man["segments"]] == ["bloc 0", "bloc 1"]
    assert sum(s["documents"] for s in man["segments"]) == man["documents"]
    ok = stream_coverage(cfg, seen=n)
    assert ok["warnings"] == [] and ok["passes"] == 1
    cut = stream_coverage(cfg, seen=int(n * 0.8))  # le budget s'arrête avant la fin du « meilleur » bloc
    seen = {s["label"]: s["seen"] for s in cut["segments"]}
    assert seen["bloc 0"] == 1.0 and seen["bloc 1"] < 0.9
    assert any("bloc 1" in w for w in cut["warnings"])
    wrap = stream_coverage(cfg, seen=2 * n)  # flux trop court : le kit le relit depuis le début
    assert wrap["passes"] == 2 and any("relu" in w for w in wrap["warnings"])


def test_full_mode_never_reuses_a_debug_run_and_passes_only_official_options(blocks_prepared, monkeypatch, tmp_path):
    import shutil

    import torch

    import bench.run as br
    cfg = json.loads(json.dumps(blocks_prepared))
    cfg["train"].update({"mode": "full", "device": "cpu"})
    cfg["eval"]["validation"] = str(_tiny_validation(tmp_path / "val.jsonl"))
    rdir = run_dir(cfg)
    shutil.rmtree(rdir, ignore_errors=True)
    rdir.mkdir(parents=True)
    data_meta = json.loads((mix_dir(cfg) / "prepared" / "data_meta.json").read_text())
    # checkpoint d'un run de mise au point (--debug-budget) au budget complet : mêmes données, même
    # tokenizer, autant de tokens vus qu'un run officiel, mais `official` False
    torch.save({"meta": {"dataset_sha256": data_meta["dataset_sha256"], "tokenizer_sha256": data_meta["tokenizer_sha256"],
                         "tokens_seen": br.tokens_seen_expected(cfg), "official": False}}, rdir / "ckpt.pt")
    calls = []

    def fake_kit(module, args, log):
        calls.append((module, [str(a) for a in args]))
        raise RuntimeError("stop")

    monkeypatch.setattr(br, "kit", fake_kit)
    with pytest.raises(RuntimeError, match="stop"):
        stage_train(cfg, force=False, resume=False, log=br.get_logger("test"))
    assert calls and calls[0][0] == "ftb.train"  # le run de mise au point n'est PAS réutilisé
    assert list(rdir.parent.glob(f"{rdir.name}.prev-*"))
    args = calls[0][1]
    assert "--debug-budget" not in args and "--validation" not in args


def test_results_csv_refuses_a_different_header(tmp_path):
    path = tmp_path / "results.csv"
    append_csv(path, {"a": 1, "b": 2}, ["a", "b"])
    append_csv(path, {"a": 3, "b": 4}, ["a", "b"])
    with pytest.raises(SystemExit):
        append_csv(path, {"a": 5, "c": 6, "b": 7}, ["a", "c", "b"])
    assert path.read_text().splitlines() == ["a,b", "1,2", "3,4"]


def test_paired_bootstrap():
    import numpy as np
    r = np.random.default_rng(0)
    genres = ["encyclopedique"] * 50 + ["presse"] * 100
    nbytes = r.integers(1000, 6000, size=150)
    bits_a = nbytes * r.uniform(1.8, 2.2, size=150)
    same = paired_bootstrap(bits_a, bits_a, nbytes, genres, n_boot=500)
    assert same["delta_b_minus_a"] == 0 and same["ci95"] == [0.0, 0.0] and same["p_a_better"] == 0
    worse = paired_bootstrap(bits_a, bits_a * 1.05, nbytes, genres, n_boot=500)  # B 5 % moins bon partout
    assert worse["ci95"][0] > 0 and worse["p_a_better"] == 1.0
    noisy = paired_bootstrap(bits_a, bits_a * r.normal(1.0, 0.2, size=150), nbytes, genres, n_boot=2000)
    assert noisy["ci95"][0] < noisy["delta_b_minus_a"] < noisy["ci95"][1]
