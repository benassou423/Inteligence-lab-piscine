"""Étape exact_dedup : normalisation, première occurrence gardée, doublons entre sources."""
import numpy as np

from pipeline.dedup.exact import ExactDedupStage, normalize, text_key
from pipeline.run import Context
from pathlib import Path


def test_normalisation():
    assert normalize("  Le  CHAT\n\tdort ici ") == "le chat dort ici"
    assert text_key("Le chat dort.") == text_key("le   chat\ndort.")
    assert text_key("Le chat dort.") != text_key("Le chien dort.")
    assert len(text_key("x")) == 16  # 128 bits


def _keys(texts):
    buf = b"".join(text_key(t) for t in texts)
    return np.frombuffer(buf, dtype=np.uint64).reshape(-1, 2)


def test_premiere_occurrence_gardee_entre_sources():
    stage = ExactDedupStage({}, Context(root=Path("."), config_name="t", sources=["a", "b"]))
    a = ["Texte un.", "Texte deux.", "texte UN."]  # doublon interne (casse différente)
    b = ["Texte deux.", "Texte trois."]  # doublon de a
    stage.plan([("a/part-00000.jsonl.zst", "a", _keys(a)), ("b/part-00000.jsonl.zst", "b", _keys(b))])
    assert stage.keep.tolist() == [True, True, False, False, True]
    assert stage.copies.tolist() == [2, 2, 2, 2, 1]
    out = stage.apply_at({"id": "b:1", "text": b[0]}, "b/part-00000.jsonl.zst", 0)
    assert not out.keep and out.reason == "exact_duplicate"
    assert out.doc["meta"]["exact_dedup_first"] == "a/part-00000.jsonl.zst#1"
    kept = stage.apply_at({"id": "a:1", "text": a[1]}, "a/part-00000.jsonl.zst", 1)
    assert kept.keep and kept.doc["meta"]["exact_dedup_copies"] == 2
    assert stage.top[0]["copies"] == 2


def test_aucun_doublon():
    stage = ExactDedupStage({}, Context(root=Path("."), config_name="t", sources=["a"]))
    stage.plan([("a/p", "a", _keys(["x", "y", "z"]))])
    assert stage.keep.all() and stage.top == []
