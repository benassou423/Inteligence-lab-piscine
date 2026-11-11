"""Outils d'annotation du filtre de langue : tirage stratifié, session au clavier, estimations pondérées."""
import io
import json

import pytest

from pipeline.io import write_docs
from tools import annotate_langid as A
from tools import langid_report as R

POP = {"kept_0.99-1.00": 900, "kept_0.65-0.80": 100, "rejected_fam_mid": 50, "rejected_other_high": 50}
# strate -> (étiquette, score 1re étiquette, p(français), p(français + parlers proches))
SCORES = {"kept_0.99-1.00": ("fra_Latn", 0.995, 0.995, 0.996), "kept_0.65-0.80": ("fra_Latn", 0.7, 0.7, 0.75),
          "rejected_fam_mid": ("fra_Latn", 0.5, 0.5, 0.6), "rejected_other_high": ("eng_Latn", 0.97, 0.01, 0.02)}


def _fake_stage(tmp_path, old_format=False):
    """Faux dossier d'étape langid : stats.json + _strata_sample (10 documents par strate)."""
    d = tmp_path / "stages" / "cfg" / "05_langid"
    recs = []
    for stratum in POP:
        label, score, fr, fam = SCORES[stratum]
        for i in range(10):
            rec = {"id": f"a:{stratum}:{i}", "source": "a", "text": f"texte {stratum} {i}", "text_bytes": 20,
                   "langid_label": label, "langid_score": score, "keep": stratum.startswith("kept"),
                   "stratum": stratum}
            if not old_format:
                rec.update(langid_fr=fr, langid_family=fam, langid_group="fr" if fr > 0.4 else "other",
                           langid_top={label: score})
            recs.append(rec)
    write_docs(d / "_strata_sample" / "a.jsonl.zst", recs)
    stats = {"params": {"threshold": 0.65, "rule": "fr"}, "sources": {"a": {"details": {"strata": POP}}}}
    (d / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return d


def test_tirage_stratifie_et_poids(tmp_path):
    d = _fake_stage(tmp_path)
    items, meta = A.draw_sample(d, n=20)
    assert items == A.draw_sample(d, n=20)[0]  # déterministe
    by = {}
    for it in items:
        by.setdefault(it["stratum"], []).append(it)
    assert {h: len(v) for h, v in by.items()} == {
        "kept_0.99-1.00": 6, "kept_0.65-0.80": 6, "rejected_fam_mid": 4, "rejected_other_high": 4}
    assert by["kept_0.99-1.00"][0]["weight"] == pytest.approx(150)
    assert by["rejected_fam_mid"][0]["weight"] == pytest.approx(12.5)
    assert meta["population"]["a|kept_0.99-1.00"] == 900 and meta["rule"] == "fr"


def test_tirage_refuse_un_ancien_echantillon(tmp_path):
    """Sans langid_fr / langid_family, la règle « famille » ne serait pas évaluable : on refuse."""
    with pytest.raises(SystemExit):
        A.draw_sample(_fake_stage(tmp_path, old_format=True), n=20)


def test_repartition_plus_forts_restes():
    assert A._largest_remainder(10, {"x": 1, "y": 1, "z": 1}, {"x": 9, "y": 9, "z": 9}) == {"x": 4, "y": 3, "z": 3}
    assert A._largest_remainder(10, {"x": 1, "y": 1}, {"x": 2, "y": 20}) == {"x": 2, "y": 8}
    assert A._largest_remainder(10, {"x": 1}, {"x": 3}) == {"x": 3}


def _annotate_all(items, path):
    """f partout, sauf : moitié n dans kept_0.65-0.80, tout n dans rejected_other_high, moitié n dans fam_mid."""
    seen = {}
    for it in items:
        k = seen.get(it["stratum"], 0)
        seen[it["stratum"]] = k + 1
        h = it["stratum"]
        label = "f"
        if h == "rejected_other_high" or (h in ("kept_0.65-0.80", "rejected_fam_mid") and k % 2 == 1):
            label = "n"
        A.append_annotation(path, {"id": it["id"], "label": label})


def test_estimations_ponderees(tmp_path):
    d = _fake_stage(tmp_path)
    items, meta = A.draw_sample(d, n=20)
    ann_path = tmp_path / "langid.jsonl"
    _annotate_all(items, ann_path)
    ann = {r["id"]: r["label"] for r in map(json.loads, ann_path.read_text().splitlines())}
    rows, coverage = R.annotated_rows({it["id"]: it for it in items}, meta, ann)
    assert coverage == pytest.approx(1.0)
    p, r, lost, kept = R.estimate(rows)
    # Précision = (900 + 100 × 1/2) / 1000 ; rappel = 950 / (950 + 50 × 1/2)
    assert p == pytest.approx(0.95)
    assert r == pytest.approx(950 / 975)
    assert lost == pytest.approx(25) and kept == pytest.approx(1000)
    # Seuil 0.5 : les « fam_mid » (p(fr) = 0,5) deviennent gardés -> rappel 100 %
    assert R.estimate(rows, threshold=0.5)[1] == pytest.approx(1.0)
    # Mêmes annotations, règle « famille » : à 0,55 elle repêche fam_mid (0,6), la règle fr non (0,5)
    assert R.estimate(rows, threshold=0.55, rule="family")[1] == pytest.approx(1.0)
    assert R.estimate(rows, threshold=0.55, rule="fr")[1] == pytest.approx(950 / 975)
    assert R.estimate(rows, threshold=0.55, rule="family")[0] == pytest.approx(975 / 1050)
    report, summary = R.build_report({it["id"]: it for it in items}, meta, ann, n_boot=50)
    assert "Précision" in report and "95,0 %" in report and summary["n_annotated"] == 20
    assert "Précision famille" in report and "Repêchés par la règle famille" in report
    assert summary["sweep"]["family@0.55"]["recall"] == pytest.approx(1.0)


def test_session_clavier_annuler_et_reprendre(tmp_path):
    d = _fake_stage(tmp_path)
    items, meta = A.draw_sample(d, n=6)
    sample = tmp_path / "langid_sample.jsonl"
    sample.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    ann = tmp_path / "langid.jsonl"
    keys = iter(["f", "x", "n", "u", "m", "q"])  # x ignoré ; u annule le « n »
    A.run_session(sample, ann, out=io.StringIO(), key_reader=lambda: next(keys))
    got = [json.loads(line) for line in ann.read_text().splitlines()]
    assert [(g["id"], g["label"]) for g in got] == [(items[0]["id"], "f"), (items[1]["id"], "m")]
    # L'annotation garde toutes les notes : les deux règles restent évaluables sans le tirage
    for g, it in zip(got, items):
        assert (g["langid_fr"], g["langid_family"], g["stratum"]) == (it["langid_fr"], it["langid_family"], it["stratum"])
    # Reprise : repart au 3e document
    keys = iter(["s", "f", "f", "f"])
    out = io.StringIO()
    A.run_session(sample, ann, out=out, key_reader=lambda: next(keys))
    assert len(A.load_annotations(ann)) == 6 and "annotés" in out.getvalue()
    # À l'aveugle : ni l'étiquette ni la note du modèle à l'écran
    assert "fra_Latn" not in out.getvalue() and "eng_Latn" not in out.getvalue()


def test_report_discloses_ai_origin_and_unrepresented_sources(tmp_path):
    items, meta = A.draw_sample(_fake_stage(tmp_path), n=20)
    meta.update(annotation_kind="ai", uncertain_ids=[items[0]["id"]], n_truncated=2, sample_max_chars=1500)
    meta["population"]["absente|kept_0.99-1.00"] = 20
    report, summary = R.build_report({it["id"]: it for it in items}, meta,
                                    {it["id"]: "f" for it in items}, n_boot=10)
    assert "annotations IA" in report and "pas une validation humaine" in report
    assert "2 extraits tronqués" in report and "Cas à examiner : 1" in report
    assert "10 tirages avec remise" in report
    assert "Sources sans annotation" in report and "`absente`" in report
    assert summary["annotation_kind"] == "ai" and summary["coverage"] < 1
    assert summary["missing_sources"] == ["absente"]


def test_load_detects_ai_labels_even_without_origin_in_metadata(tmp_path):
    items, meta = A.draw_sample(_fake_stage(tmp_path), n=2)
    (tmp_path / "langid_sample.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
    (tmp_path / "langid_sample.meta.json").write_text(json.dumps(meta))
    (tmp_path / "langid.jsonl").write_text("".join(json.dumps({"id": it["id"], "label": "f", "annotation_kind": "ai"}) + "\n" for it in items))
    _, loaded_meta, _ = R.load(tmp_path)
    assert loaded_meta["annotation_kind"] == "ai"
