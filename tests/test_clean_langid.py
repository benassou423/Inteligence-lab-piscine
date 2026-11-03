"""Filtre de langue : notes (langid_score, GlotLID) puis seuil (langid). Tests du modèle ignorés s'il est absent."""
import json
import random

import pytest

from pipeline.clean.langid import LangIdStage, kept_strata, stratum_of
from pipeline.clean.langid_score import MODELS, ROOT, model_path, summarize_probs
from pipeline.run import Context, load_config

HAS_MODEL = model_path("glotlid").exists()
needs_model = pytest.mark.skipif(
    not HAS_MODEL, reason="modèle GlotLID absent (uv run python -m pipeline.clean.langid_score --download glotlid)")
CTX = Context(root=ROOT, config_name="t", sources=["s"])
FR, FAM = MODELS["glotlid"]["fr_labels"], MODELS["glotlid"]["family_labels"]


def _doc(probs, i=0):
    return {"id": f"s:{i}", "source": "s", "text": "texte", "meta": summarize_probs(probs, FR, FAM)}


def test_strates():
    names = [n for n, _, _ in kept_strata(0.65)]
    assert names == ["kept_0.65-0.80", "kept_0.80-0.90", "kept_0.90-0.95", "kept_0.95-0.99", "kept_0.99-1.00"]
    # stratum_of(gardé, score de la règle, score famille, score de la 1re étiquette, seuil)
    assert stratum_of(True, 1.0, 1.0, 1.0, 0.65) == "kept_0.99-1.00"
    assert stratum_of(True, 0.70, 0.9, 0.7, 0.65) == "kept_0.65-0.80"
    assert stratum_of(False, 0.58, 0.93, 0.58, 0.65) == "rejected_fam_high"  # fra 0,58 + nrm 0,35
    assert stratum_of(False, 0.30, 0.50, 0.30, 0.65) == "rejected_fam_mid"
    assert stratum_of(False, 0.01, 0.02, 0.5, 0.65) == "rejected_other_low"
    assert stratum_of(False, 0.01, 0.02, 0.97, 0.65) == "rejected_other_high"


def test_resume_des_probabilites():
    m = summarize_probs({"fra_Latn": 0.58, "nrm_Latn": 0.35, "atj_Latn": 0.023, "hat_Latn": 0.005}, FR, FAM)
    assert (m["langid_label"], m["langid_score"], m["langid_group"]) == ("fra_Latn", 0.58, "fr")
    assert m["langid_fr"] == 0.58 and m["langid_family"] == pytest.approx(0.93)
    assert m["langid_top"] == {"fra_Latn": 0.58, "nrm_Latn": 0.35, "atj_Latn": 0.023}  # >= 0,01 seulement
    assert summarize_probs({"nrm_Latn": 0.6, "fra_Latn": 0.4}, FR, FAM)["langid_group"] == "family"


def test_regle_fr_identique_a_l_ancienne():
    """Seuil >= 0,5 : « p(français) >= seuil » = l'ancienne règle « 1re étiquette fra_Latn et score >= seuil »."""
    rng = random.Random(0)
    labels = ["fra_Latn", "nrm_Latn", "frp_Latn", "eng_Latn", "dag_Latn", "run_Latn"]
    for t in (0.5, 0.65, 0.8):
        stage = LangIdStage({"threshold": t}, CTX)
        for i in range(2000):
            w = [rng.random() ** 3 for _ in labels]
            probs = {lab: x / sum(w) for lab, x in zip(labels, w)}
            top = max(probs, key=probs.get)
            ancienne = top == "fra_Latn" and probs[top] >= t
            out = stage.apply(_doc(probs, i))
            assert out.keep == ancienne
            if not out.keep:
                assert out.reason == ("low_score" if top == "fra_Latn" else "not_french")


def test_regle_famille_repeche_les_parlers_proches():
    foot = {"fra_Latn": 0.58, "nrm_Latn": 0.35, "atj_Latn": 0.07}  # compte rendu de foot de Wikinews
    out = LangIdStage({"threshold": 0.65}, CTX).apply(_doc(foot))
    assert not out.keep and out.reason == "low_score"
    out = LangIdStage({"threshold": 0.65, "rule": "family"}, CTX).apply(_doc(foot))
    assert out.keep
    rec = out.samples[0][2]  # l'échantillon d'annotation garde toutes les notes
    assert rec["langid_fr"] == 0.58 and rec["langid_family"] == pytest.approx(0.93) and rec["keep"]
    nrm = {"nrm_Latn": 0.55, "fra_Latn": 0.30, "eng_Latn": 0.15}
    out = LangIdStage({"threshold": 0.65, "rule": "family"}, CTX).apply(_doc(nrm))
    assert out.keep  # 0,85 >= 0,65 : 1re étiquette normande, mais famille française
    out = LangIdStage({"threshold": 0.9, "rule": "family"}, CTX).apply(_doc(nrm))
    assert not out.keep and out.reason == "low_score"


def test_sans_notes_ou_regle_inconnue():
    with pytest.raises(RuntimeError):
        LangIdStage({}, CTX).apply({"id": "s:1", "source": "s", "text": "texte", "meta": {}})
    with pytest.raises(ValueError):
        LangIdStage({"rule": "toto"}, CTX)


def test_config_langid_sans_langid_score_refusee(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nstages:\n  - name: exclude_heldout\n  - name: langid\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("name: ok\nstages:\n  - name: exclude_heldout\n  - name: langid_score\n  - name: langid\n",
                 encoding="utf-8")
    assert [s["name"] for s in load_config(p)["stages"]][1:] == ["langid_score", "langid"]


@pytest.fixture(scope="module")
def scorer():
    from pipeline.clean.langid_score import LangIdScoreStage

    st = LangIdScoreStage({"download": False}, CTX)
    st.setup()
    return st


@needs_model
def test_francais_garde_anglais_retire(scorer):
    fr = {"id": "s:1", "source": "s", "text": "La Révolution française est une période de bouleversements\n"
                                              "politiques et sociaux en France à la fin du XVIIIe siècle."}
    en = {"id": "s:2", "source": "s", "text": "The French Revolution was a period of political and societal change."}
    filt = LangIdStage({}, CTX)
    out = filt.apply(scorer.apply(fr).doc)
    meta = out.doc["meta"]
    assert out.keep and meta["langid_label"] == "fra_Latn" and meta["langid_fr"] >= 0.65
    assert meta["langid_family"] >= meta["langid_fr"] and meta["langid_model"] == "glotlid"
    out = filt.apply(scorer.apply(en).doc)
    assert not out.keep and out.reason == "not_french" and out.doc["meta"]["langid_label"] == "eng_Latn"
    out = filt.apply(scorer.apply({"id": "s:3", "source": "s", "text": "  \n "}).doc)
    assert not out.keep and out.reason == "empty"


@needs_model
def test_changer_le_seuil_ne_relance_pas_glotlid(tmp_path):
    from pipeline.run import Runner
    from tests.fixtures.clean import minicorpus

    raw = minicorpus.build(tmp_path)
    stages = ("encoding", "boilerplate", "langid_score", "langid", "exact_dedup")
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"), stages=stages)
    assert Runner(cfg, input_dir=raw, out_root=tmp_path / "out", workers=2).run() == 0
    out = tmp_path / "out" / "mini"
    st = json.loads((out / "05_langid" / "stats.json").read_text(encoding="utf-8"))
    assert st["sources"]["wikipedia_fr"]["docs_out"] == 4  # aucun ID exclu ici : 4 articles français
    assert (out / "05_langid" / "_strata_sample" / "wikipedia_fr.jsonl.zst").exists()
    assert sum(st["sources"]["wikipedia_fr"]["details"]["score_hist"]["kept"]) == 4
    # Nouveau seuil et nouvelle règle : les notes viennent du cache, seul le filtre tourne
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"), stages=stages,
                                  params={"langid": {"threshold": 0.999, "rule": "family"}})
    assert Runner(cfg, input_dir=raw, out_root=tmp_path / "out", workers=2).run() == 0
    timing = {d: json.loads((out / d / "stats.json").read_text(encoding="utf-8"))["timing"]
              for d in ("04_langid_score", "05_langid")}
    assert timing["04_langid_score"]["units_run"] == 0 and timing["05_langid"]["units_run"] == 3
    st = json.loads((out / "05_langid" / "stats.json").read_text(encoding="utf-8"))
    assert st["params"] == {"threshold": 0.999, "rule": "family"}
    assert st["sources"]["wikipedia_fr"]["docs_out"] < 4  # seuil 0,999 : des articles tombent
