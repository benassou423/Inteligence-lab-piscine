"""Tests des outils communs de la collecte (écriture en parts, compteurs, lanceur avec reprise)."""
import json

import pytest

from pipeline.collect import common
from pipeline.io import read_docs

NAME = "fake_source_test"
DOMAIN = "web"


def _doc(i, date="2026-01-01"):
    return common.make_doc(NAME, DOMAIN, str(i), "x" * 10, url=None, date=date, license="test")


def test_make_doc_format():
    d = common.make_doc("src", "web", "42", "texte", url="http://a.fr", date="2026-01-01", license="L",
                        meta={"k": 1}, html="<p>texte</p>")
    assert d == {"id": "src:42", "source": "src", "domain": "web", "text": "texte", "url": "http://a.fr",
                 "date": "2026-01-01", "license": "L", "meta": {"k": 1}, "html": "<p>texte</p>"}
    assert "html" not in common.make_doc("s", "web", "1", "t", url=None, date=None, license="L")


def test_date_ok_freeze():
    assert common.date_ok("2026-10-16")
    assert common.date_ok("1997-01-01")
    assert not common.date_ok("2026-10-17")
    assert not common.date_ok(None)


def test_write_parts_splits_by_text_bytes(tmp_path):
    paths = common.write_parts((_doc(i) for i in range(5)), tmp_path, max_bytes=25)
    assert [p.name for p in paths] == ["part-00000.jsonl.zst", "part-00001.jsonl.zst", "part-00002.jsonl.zst"]
    counts = [len(list(read_docs(p))) for p in paths]
    assert counts == [2, 2, 1]
    assert not list(tmp_path.glob("*.tmp"))


def test_write_parts_empty(tmp_path):
    assert common.write_parts(iter([]), tmp_path) == []


def test_tally_merge():
    a, b = common.Tally(), common.Tally()
    a.add(_doc(1, "2020-01-01"))
    b.add(_doc(2, "2021-05-05"))
    b.skip("pas_francais", 3)
    m = common.Tally.merge([a.to_dict(), b.to_dict()])
    assert (m["docs"], m["bytes"], m["date_min"], m["date_max"]) == (2, 20, "2020-01-01", "2021-05-05")
    assert m["removed_by_reason"] == {"pas_francais": 3}


# --- faux module source pour tester le lanceur (sans réseau)
CALLS = []


def prepare(cfg, log, sample):
    return {"n_units": 3}


def units(ctx):
    return [f"u{i}" for i in range(ctx["n_units"])]


def process(unit, ctx, tally, log):
    CALLS.append(unit)
    k = int(unit[1:])
    for j in range(4):
        if j == 3:
            tally.skip("rejet_test")
            continue
        yield _doc(f"{k}-{j}")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW_DIR", tmp_path)
    return tmp_path


def test_run_source_sample_then_full(raw_dir):
    import sys
    mod = sys.modules[__name__]
    cfg = {"sources": {NAME: {"workers": 1}}}
    CALLS.clear()
    st = common.run_source(mod, cfg, limit=2)
    src = raw_dir / NAME
    assert st["docs"] == 2 and CALLS == ["u0"]
    assert [d["id"] for d in read_docs(src / "part-00000.jsonl.zst")] == [f"{NAME}:0-0", f"{NAME}:0-1"]
    assert json.loads((src / "_state.json").read_text())["mode"] == "sample"

    # collecte complète, interrompue après u0 puis reprise : u0 n'est pas refait
    work = src / "_work"
    common.append_done(work, "u0", {"docs": 3, "bytes": 30, "date_min": "2026-01-01", "date_max": "2026-01-01",
                                     "removed_by_reason": {"rejet_test": 1}, "extra": {}})
    common.write_parts((_doc(f"0-{j}") for j in range(3)), work / "u0")
    CALLS.clear()
    st = common.run_source(mod, cfg)
    assert CALLS == ["u1", "u2"]
    assert st["docs"] == 9 and st["removed_by_reason"] == {"rejet_test": 3}
    ids = [d["id"] for p in sorted(src.glob("part-*.jsonl.zst")) for d in read_docs(p)]
    assert ids == [f"{NAME}:{k}-{j}" for k in range(3) for j in range(3)]  # ordre déterministe
    assert not work.exists()
    # un échantillon ne remplace pas une collecte complète
    assert common.run_source(mod, cfg, limit=1)["mode"] == "full"
