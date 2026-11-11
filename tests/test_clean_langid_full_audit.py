"""L'audit couvre les textes entiers, compte les effets et laisse les entrées intactes."""
import copy

import pytest

from pipeline.io import read_docs, write_docs
from tools import audit_langid_full as A


def doc(key, text, fr=.8, family=.9, source="wikipedia_fr"):
    return {"id": key, "source": source, "text": text, "meta": {
        "langid_fr": fr, "langid_family": family, "langid_model": "fake"}, "audit_roles": ["test"]}


@pytest.mark.parametrize("text", ["", "é漢字" * 2000, "\n" * 6100, "Un texte français.\n" * 400,
                                  "x" * 7001, "Avant.\n" + " " * 5000 + "Après."])
def test_windows_reconstruct_full_text(text):
    ranges = list(A.windows(text, 1000))
    assert "".join(chunk for _, _, chunk in ranges) == text
    assert sum(end-start for start, end, _ in ranges) == len(text)
    assert all(0 < end-start <= 1000 and text[start:end] == chunk for start, end, chunk in ranges)


def test_full_audit_detects_foreign_tail_without_changing_input():
    class Scorer:
        def score(self, text):
            fr = 0.01 if "FOREIGN" in text else .99
            return {"langid_label": "fr" if fr > .5 else "en", "langid_fr": fr, "langid_family": fr}
    d = doc("w:1", "Texte français. " * 130 + "FOREIGN ordinary English text. " * 200)
    before = copy.deepcopy(d)
    result = A.analyse_doc(d, Scorer(), Scorer(), window_chars=1000)
    assert d == before
    assert result["prefix_lid176"]["langid_fr"] > .65
    assert result["full_lid176"]["langid_fr"] < .65
    assert result["segments"][-1]["lid176"]["langid_fr"] < .65
    assert result["coverage_chars"] == len(d["text"])


def test_scan_exact_counts_cache_and_full_text(tmp_path):
    source = tmp_path / "input" / "wikipedia_fr" / "part-00000.jsonl.zst"
    data = [doc("w:1", "a" * 5000, .2, .9), doc("w:2", "Bonjour.", .8, .9), doc("w:3", "Other.", .01, .02)]
    write_docs(source, data)
    before = source.read_bytes()
    args = (str(source), str(tmp_path / "audit"), {"w:1"}, .65, "code-v1")
    result = A.scan_unit(args)
    assert result["counts"]["docs"] == 3 and result["counts"]["current_docs"] == 1
    assert result["counts"]["candidate_docs"] == 2 and result["counts"]["added_docs"] == 1
    assert result["counts"]["added_bytes"] == 5000
    assert list(read_docs(result["original_path"]))[0]["text"] == "a" * 5000
    assert len(list(read_docs(result["candidate_path"]))) == 1
    assert A.scan_unit(args) == result and source.read_bytes() == before
    write_docs(source, [doc("w:1", "Changed.", .8, .9)])
    assert A.scan_unit(args)["counts"].get("added_docs", 0) == 0


@pytest.mark.parametrize("fr,family", [(float("nan"), .9), (.9, .2), (-.1, .9), (.1, 1.1)])
def test_bad_probabilities_stop_the_audit(fr, family):
    with pytest.raises(ValueError, match="Scores invalides"):
        A.decisions(doc("w:1", "texte", fr, family))


def test_quality_signals_do_not_claim_language():
    result = A.quality_features("Titre\n| align=center| 1 | 2 |\nBretagne " + "Bretagne " * 50)
    assert result["markup_lines"] == 1 and result["max_repeated_word_run"] == 51
    assert "label" not in result
