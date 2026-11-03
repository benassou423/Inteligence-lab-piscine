"""bench.mix : sélection, répétition, plafonds, poids, ordre, déterminisme, décontamination."""
import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from bench.config import mix_dir
from bench.mix import build_mix, build_val_ngrams, shared_ngrams
from pipeline.io import write_docs

_spec = importlib.util.spec_from_file_location("bench_fx", Path(__file__).parent / "fixtures/bench/make_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("mix")
    docs = fx.make_corpus(root / "data", layout="stage")
    return {"root": root, "docs": docs, "all": [d for ds in docs.values() for d in ds]}


def _mix(corpus, run, **over):
    cfg = fx.run_config(corpus["root"], run=run, **over)
    man = build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], corpus["root"] / "data")
    return man, mix_dir(cfg)


def _texts(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def test_natural_mix_keeps_every_document_once(corpus):
    man, d = _mix(corpus, "nat")
    texts = _texts(d / "train.jsonl.gz")
    assert Counter(texts) == Counter(x["text"] for x in corpus["all"])
    assert man["documents"] == len(corpus["all"])
    assert man["bytes"] == sum(len(x["text"].encode()) for x in corpus["all"])
    assert set(man["by_source"]) == set(corpus["docs"])
    assert man["train_sha256"]


def test_deterministic_and_cached(corpus):
    a, da = _mix(corpus, "det1")
    b, db = _mix(corpus, "det2")
    assert a["train_sha256"] == b["train_sha256"]  # gzip sans date : même mix, même fichier
    assert (da / "train.jsonl.gz").read_bytes() == (db / "train.jsonl.gz").read_bytes()
    c, _ = _mix(corpus, "det1")  # rien n'a changé : relu, pas refait
    assert c["created"] == a["created"]


def test_repeat_caps_filter_and_weights(corpus):
    one, _ = _mix(corpus, "r1")
    two, _ = _mix(corpus, "r2", mix={"sources": [{"pipeline": "nettoye", "repeat": 2}]})
    assert two["documents"] == 2 * one["documents"] and two["bytes"] == 2 * one["bytes"]
    capped, _ = _mix(corpus, "cap", mix={"max_bytes": one["bytes"] // 3})
    assert 0 < capped["bytes"] <= one["bytes"] // 3
    wiki, _ = _mix(corpus, "wiki", mix={"sources": [{"pipeline": "nettoye", "sources": ["wikipedia_fr"]}]})
    assert set(wiki["by_source"]) == {"wikipedia_fr"}
    mixw, _ = _mix(corpus, "w", mix={"total_bytes": 200_000, "sources": [
        {"name": "wiki", "pipeline": "nettoye", "sources": ["wikipedia_fr"], "weight": 1},
        {"name": "cc", "pipeline": "nettoye", "sources": ["commoncrawl"], "weight": 1}]})
    b = {s: v["bytes"] for s, v in mixw["by_source"].items()}
    assert 0.6 < b["wikipedia_fr"] / b["commoncrawl"] < 1.6  # ~50/50 malgré des tailles différentes


def test_weights_and_fractional_repeat_are_exact_in_bytes(tmp_path):
    # Une source de GROS documents (12 livres de ~50 Ko) : un tirage document par document
    # s'écartait de la part visée de ±60 % ; la sélection en octets tient au demi-document près.
    import random
    r = random.Random(1)
    para = lambda: " ".join(r.choice(["le", "chat", "dort", "sur", "la", "table", "et", "rêve"]) for _ in range(60))  # noqa: E731
    big = [{"id": f"livre:{i}", "source": "livres", "text": "\n\n".join(para() for _ in range(r.randint(140, 200)))}
           for i in range(12)]
    small = [{"id": f"page:{i}", "source": "pages", "text": para()} for i in range(400)]
    write_docs(tmp_path / "raw" / "livres" / "part-00000.jsonl.zst", big)
    write_docs(tmp_path / "raw" / "pages" / "part-00000.jsonl.zst", small)
    half_doc = max(len(d["text"].encode()) for d in big) / 2
    total_big = sum(len(d["text"].encode()) for d in big)
    for seed in range(5):
        cfg = fx.run_config(tmp_path, run=f"w{seed}", seed=seed, mix={"total_bytes": 200_000, "sources": [
            {"name": "livres", "path": str(tmp_path / "raw" / "livres"), "weight": 1},
            {"name": "pages", "path": str(tmp_path / "raw" / "pages"), "weight": 1}]})
        man = build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], tmp_path)
        assert abs(man["by_entry"]["livres"]["bytes"] - 100_000) <= half_doc
        cfg = fx.run_config(tmp_path, run=f"r{seed}", seed=seed, mix={"sources": [
            {"name": "livres", "path": str(tmp_path / "raw" / "livres"), "repeat": 1.5}]})
        man = build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], tmp_path)
        assert abs(man["bytes"] - 1.5 * total_big) <= half_doc


def test_orders(corpus):
    sh, dsh = _mix(corpus, "o_shuf")
    asis, das = _mix(corpus, "o_asis", mix={"order": "as_is"})
    assert sh["train_sha256"] != asis["train_sha256"]
    assert sorted(_texts(dsh / "train.jsonl.gz")) == sorted(_texts(das / "train.jsonl.gz"))
    # blocs : le web d'abord, le « meilleur » (Wikipédia + Gutenberg) à la fin, mélangés dans chaque bloc
    bl, dbl = _mix(corpus, "o_blocks", mix={"order": "blocks", "sources": [
        {"name": "best", "pipeline": "nettoye", "sources": ["wikipedia_fr", "gutenberg_fr"], "block": 1},
        {"name": "web", "pipeline": "nettoye", "sources": ["commoncrawl"], "block": 0}]})
    web = {x["text"] for x in corpus["docs"]["commoncrawl"]}
    flags = [t in web for t in _texts(dbl / "train.jsonl.gz")]
    n_web = sum(flags)
    assert flags == [True] * n_web + [False] * (len(flags) - n_web)
    assert bl["by_entry"]["best"]["block"] == 1


def test_prefilter_gives_same_mix(corpus):
    on, _ = _mix(corpus, "pf_on", mix={"max_bytes": 60_000})
    off, _ = _mix(corpus, "pf_off", mix={"max_bytes": 60_000, "prefilter": False})
    assert on["prefilter"]["skipped_docs"] > 0 and off["prefilter"]["skipped_docs"] == 0
    assert on["train_sha256"] == off["train_sha256"]


def test_txt_input_and_tokenizer_sample(corpus, tmp_path):
    txt = tmp_path / "piece.txt"
    txt.write_text("ACTE I\n\nSCÈNE PREMIÈRE\n\nALCESTE. — Laissez-moi, je vous prie.\n", encoding="utf-8")
    man, d = _mix(corpus, "txt", mix={"sources": [{"name": "piece", "path": str(txt), "repeat": 3},
                                                  {"name": "wiki", "pipeline": "nettoye", "sources": ["wikipedia_fr"]}],
                                      "tokenizer_sample_bytes": 20_000})
    assert man["by_entry"]["piece"]["documents"] == 3
    warn = {i["name"]: i["warning"] for i in man["inputs"]}  # source brute sans décontamination : signalée
    assert "decontaminate" in warn["piece"] and warn["wiki"] is None
    texts = _texts(d / "train.jsonl.gz")
    sample = _texts(d / "tokenizer_sample.jsonl.gz")
    assert 0 < man["tokenizer_sample"]["bytes"] <= 40_000
    assert set(sample) <= set(texts) and len(sample) == len(set(sample))


def test_decontamination_ignores_templates_but_removes_copies(tmp_path):
    # Paragraphe-gabarit (rubrique « Climat » des communes) dans 12 documents ET dans la val :
    # ce n'est pas une copie, on garde. Une vraie copie du document de la val est retirée.
    import random
    r = random.Random(0)
    vocab = ["mot%d" % i for i in range(400)]
    filler = lambda n: " ".join(r.choice(vocab) for _ in range(n))  # noqa: E731
    template = ("Le climat qui caractérise la commune est qualifié de climat océanique altéré "
                "selon la typologie des climats de la France qui compte alors huit grands types "
                "de climats en métropole et la commune sort du type des climats de montagne.")
    val_text = template + " " + filler(80)
    docs = [{"id": f"w:{i}", "source": "wiki", "text": template + " " + filler(120)} for i in range(12)]
    docs.append({"id": "copie", "source": "wiki", "text": "Préambule. " + val_text})
    docs += [{"id": f"autre:{i}", "source": "wiki", "text": filler(150)} for i in range(20)]
    write_docs(tmp_path / "data" / "raw" / "wiki" / "part-00000.jsonl.zst", docs)
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"id": "v", "genre": "encyclopedique", "text": val_text}) + "\n", encoding="utf-8")
    cfg = fx.run_config(tmp_path, run="dec", mix={
        "sources": [{"name": "wiki", "path": str(tmp_path / "data" / "raw")}],
        "decontaminate": {"enabled": True, "against": str(val), "min_matches": 20, "max_df": 5}})
    man = build_mix(cfg["mix"], mix_dir(cfg), cfg["seed"], tmp_path / "data")
    assert man["contaminated_by_source"] == {"wiki": 1}
    assert man["inputs"][0]["warning"] is None  # décontaminée : pas d'alerte
    texts = _texts(mix_dir(cfg) / "train.jsonl.gz")
    assert sum(t.startswith(template) for t in texts) == 12
    assert not any(t.startswith("Préambule") for t in texts)


def test_shared_ngrams_normalizes_typography():
    vocab, hashes = build_val_ngrams(["L’homme qu’il a vu « hier » au marché de la ville voisine sans rien dire"], 5)
    other_edition = "l'homme qu'il a vu \"hier\" au marché de la ville voisine sans rien dire"
    assert len(shared_ngrams(other_edition, vocab, hashes, 5)) == len(hashes) > 0
