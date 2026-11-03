"""Mini-run du runner sur fixtures : sorties, stats.json, funnel.md, déterminisme et reprise."""
import json

import pytest

from pipeline.io import read_docs
from pipeline.run import Runner, load_config
from tests.fixtures.clean import minicorpus


def _setup(tmp_path, stages=("encoding", "boilerplate", "exact_dedup")):
    raw = minicorpus.build(tmp_path)
    (tmp_path / "val" / "x").mkdir(parents=True)
    (tmp_path / "val" / "x" / "heldout_ids.txt").write_text("# IDs de la val\nwikipedia_fr:4\n", encoding="utf-8")
    cfg = minicorpus.write_config(tmp_path / "mini.yaml", str(tmp_path / "val" / "*" / "heldout_ids.txt"), stages)
    return raw, cfg


def _run(cfg, raw, out, **kw):
    r = Runner(cfg, input_dir=raw, out_root=out, **kw)
    assert r.run() == 0
    return out / "mini"


def _docs(stage_dir):
    return {p.parent.name + "/" + p.name: list(read_docs(p)) for p in sorted(stage_dir.glob("*/part-*.jsonl.zst"))}


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mini")
    raw, cfg = _setup(tmp)
    return tmp, raw, cfg, _run(cfg, raw, tmp / "stages", workers=2)


def test_arborescence_et_contrat_stats(mini):
    _, _, _, out = mini
    dirs = ["01_exclude_heldout", "02_encoding", "03_boilerplate", "04_exact_dedup"]
    for d in dirs:
        st = json.loads((out / d / "stats.json").read_text(encoding="utf-8"))
        for s in ("wikipedia_fr", "gutenberg_fr", "commoncrawl"):
            entry = st["sources"][s]
            for k in ("docs_in", "docs_out", "bytes_in", "bytes_out", "removed_by_reason"):
                assert k in entry, (d, s, k)
            assert entry["docs_in"] - entry["docs_out"] == sum(entry["removed_by_reason"].values())
    assert (out / "funnel.md").exists() and (out / "manifest.json").exists()
    funnel = (out / "funnel.md").read_text(encoding="utf-8")
    assert "| 00 entrée |" in funnel and "04_exact_dedup" in funnel
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_dir"].endswith("04_exact_dedup")


def test_heldout_exclu(mini):
    _, _, _, out = mini
    ids = {d["id"] for docs in _docs(out / "01_exclude_heldout").values() for d in docs}
    assert "wikipedia_fr:4" not in ids and "wikipedia_fr:1" in ids
    rej = list(read_docs(out / "01_exclude_heldout" / "_rejected_sample" / "wikipedia_fr.jsonl.zst"))
    assert [r["id"] for r in rej] == ["wikipedia_fr:4"] and rej[0]["reason"] == "heldout"


def test_html_garde_puis_retire(mini):
    _, _, _, out = mini
    cc_enc = _docs(out / "02_encoding")["commoncrawl/part-00000.jsonl.zst"]
    assert all("html" in d for d in cc_enc)  # encore utile à l'étape boilerplate
    for d in ("03_boilerplate", "04_exact_dedup"):
        assert all("html" not in doc for docs in _docs(out / d).values() for doc in docs)


def test_nettoyage_applique(mini):
    _, _, _, out = mini
    final = {d["id"]: d for docs in _docs(out / "04_exact_dedup").values() for d in docs}
    assert final["wikipedia_fr:2"]["text"] == "L’été est chaud : c’est la saison des fêtes."
    assert final["wikipedia_fr:1"]["text"] == minicorpus.ARTICLE_BODY
    assert final["gutenberg_fr:100"]["text"].startswith("LE VOYAGE DE MONSIEUR FERRAND")
    assert "Accueil" not in final["commoncrawl:a"]["text"]
    assert "commoncrawl:b" not in final  # rien d'extractible
    rej = list(read_docs(out / "03_boilerplate" / "_rejected_sample" / "commoncrawl.jsonl.zst"))
    assert [(r["id"], r["reason"]) for r in rej] == [("commoncrawl:b", "no_extraction")]


def test_doublon_exact_entre_sources(mini):
    _, _, _, out = mini
    final = {d["id"]: d for docs in _docs(out / "04_exact_dedup").values() for d in docs}
    # Même texte dans Wikipédia et sur une page web : la source listée en premier (wikipedia_fr) garde.
    assert "wikipedia_fr:3" in final and "commoncrawl:c" not in final
    assert final["wikipedia_fr:3"]["meta"]["exact_dedup_copies"] == 2
    rej = list(read_docs(out / "04_exact_dedup" / "_rejected_sample" / "commoncrawl.jsonl.zst"))
    assert rej[0]["id"] == "commoncrawl:c" and rej[0]["meta"]["exact_dedup_first_id"] == "wikipedia_fr:3"
    st = json.loads((out / "04_exact_dedup" / "stats.json").read_text(encoding="utf-8"))
    assert st["report"]["top_duplicates"][0]["copies"] == 2


def test_deterministe_et_reprise(mini, tmp_path):
    _, raw, cfg, out = mini
    ref = {d: _docs(out / d) for d in ("02_encoding", "04_exact_dedup")}
    # 1 worker au lieu de 2 : sorties identiques
    out1 = _run(cfg, raw, tmp_path / "seq", workers=1)
    assert {d: _docs(out1 / d) for d in ref} == ref
    # Relance : tout vient du cache, rien n'est recalculé, sorties inchangées
    _run(cfg, raw, tmp_path / "seq", workers=1)
    for d in ("01_exclude_heldout", "02_encoding", "03_boilerplate", "04_exact_dedup"):
        st = json.loads((out1 / d / "stats.json").read_text(encoding="utf-8"))
        assert st["timing"]["units_run"] == 0, d
    assert {d: _docs(out1 / d) for d in ref} == ref


def test_limit(tmp_path):
    raw, cfg = _setup(tmp_path, stages=("encoding",))
    out = _run(cfg, raw, tmp_path / "lim", workers=1, limit=2)
    st = json.loads((out / "01_exclude_heldout" / "stats.json").read_text(encoding="utf-8"))
    assert st["sources"]["wikipedia_fr"]["docs_in"] == 2 and st["limit"] == 2
    assert st["sources"]["gutenberg_fr"]["docs_in"] == 2  # la source n'a que 2 documents
    assert "2 premiers documents par source" in (out / "funnel.md").read_text(encoding="utf-8")


def test_decontamination_dans_le_runner(tmp_path):
    pytest.importorskip("ftb.contamination")
    raw = minicorpus.build(tmp_path)
    val = tmp_path / "val.jsonl"
    extrait = ("Monsieur Ferrand avait résolu de partir un mardi, parce que le mardi,\ndisait-il, les diligences "
               "sont moins pleines et les aubergistes moins\npressés. Il prépara sa malle la veille")
    val.write_text(json.dumps({"id": "v", "genre": "litteraire", "text": extrait}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"),
                                  stages=("encoding", "boilerplate", "decontaminate", "exact_dedup"), val_path=val)
    out = _run(cfg, raw, tmp_path / "out", workers=2)
    st = json.loads((out / "04_decontaminate" / "stats.json").read_text(encoding="utf-8"))
    assert st["sources"]["gutenberg_fr"]["removed_by_reason"] == {"val_copy": 1}
    assert st["report"]["actions"] == {"copy": 1}
    ids = {d["id"] for docs in _docs(out / "05_exact_dedup").values() for d in docs}
    assert "gutenberg_fr:100" not in ids and "gutenberg_fr:200" in ids
    _run(cfg, raw, tmp_path / "out", workers=2)  # reprise : rien à refaire
    st = json.loads((out / "04_decontaminate" / "stats.json").read_text(encoding="utf-8"))
    assert st["timing"]["units_run"] == 0 and st["sources"]["gutenberg_fr"]["removed_by_reason"] == {"val_copy": 1}


def test_config_doit_commencer_par_exclude_heldout(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nstages:\n  - name: encoding\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_configs_du_projet_valides():
    for name in ("brut", "nettoye"):
        cfg = load_config(f"configs/pipeline/{name}.yaml")
        assert cfg["name"] == name and cfg["stages"][0]["name"] == "exclude_heldout"
    nettoye = load_config("configs/pipeline/nettoye.yaml")["stages"]
    assert [s["name"] for s in nettoye] == ["exclude_heldout", "encoding", "boilerplate", "langid_score", "langid",
                                            "reflow", "decontaminate", "exact_dedup"]
    assert nettoye[1]["params"]["uncurl_quotes"] is False
    # Règle de langue par défaut inchangée tant que l'annotation n'a pas tranché
    assert nettoye[4]["params"]["threshold"] == 0.65 and nettoye[4]["params"].get("rule", "fr") == "fr"


# ------------------------------------------------------------ cache : ce qui doit l'invalider

def _timing(out, stage):
    return json.loads((out / stage / "stats.json").read_text(encoding="utf-8"))["timing"]


def test_cache_invalide_si_les_ids_exclus_changent(tmp_path):
    raw, cfg = _setup(tmp_path, stages=("encoding",))
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    ids = {d["id"] for docs in _docs(out / "02_encoding").values() for d in docs}
    assert "wikipedia_fr:1" in ids and "wikipedia_fr:4" not in ids
    # Même nom de fichier, contenu changé : wikipedia_fr:1 rejoint la val
    (tmp_path / "val" / "x" / "heldout_ids.txt").write_text("wikipedia_fr:4\nwikipedia_fr:1\n", encoding="utf-8")
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    ids = {d["id"] for docs in _docs(out / "02_encoding").values() for d in docs}
    assert "wikipedia_fr:1" not in ids
    # Seule la source concernée est recalculée
    assert _timing(out, "01_exclude_heldout")["units_run"] == 1 and _timing(out, "02_encoding")["units_run"] == 1


def test_cache_encodage_suit_la_methode_de_boilerplate(tmp_path):
    """Commoncrawl : encoding laisse le texte à boilerplate (ré-extrait du HTML). Si boilerplate ne le fait
    plus, encoding doit traiter commoncrawl lui-même : son cache doit être invalidé."""
    raw = minicorpus.build(tmp_path)
    stages = ("encoding", "boilerplate")
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"), stages=stages)
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    enc = json.loads((out / "02_encoding" / "stats.json").read_text(encoding="utf-8"))
    assert enc["sources"]["commoncrawl"]["details"].get("deferred_to_boilerplate") == 3
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"), stages=stages,
                                  params={"boilerplate": {"methods": {"commoncrawl": "aucune"}}})
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    enc = json.loads((out / "02_encoding" / "stats.json").read_text(encoding="utf-8"))
    assert enc["timing"]["units_run"] == 1  # commoncrawl seulement
    assert "deferred_to_boilerplate" not in enc["sources"]["commoncrawl"]["details"]


_FAKE_STAGE = """
from pipeline.clean.base import Outcome, Stage


class {cls}(Stage):
    name = "{name}"

    def setup(self):
        {setup}

    def apply(self, doc):
        doc["text"] = doc["text"] + "{suffix}"
        return Outcome(doc)
"""


def test_code_modifie_pendant_le_run_invalide_le_cache(tmp_path, monkeypatch):
    """L'empreinte du code est prise au lancement : un fichier modifié pendant le run (ici par l'étape A,
    avant que l'étape B ne démarre) ne doit pas faire passer les sorties de l'ancien code pour celles du nouveau."""
    import sys

    from pipeline import run as R

    mods = tmp_path / "mods"
    mods.mkdir()
    b_path = mods / "fake_b_mod.py"
    a_setup = f"open({str(b_path)!r}, 'a').write('# modifié pendant le run\\n')"
    (mods / "fake_a_mod.py").write_text(_FAKE_STAGE.format(cls="A", name="fake_a", setup=a_setup, suffix=" a"),
                                        encoding="utf-8")
    b_path.write_text(_FAKE_STAGE.format(cls="B", name="fake_b", setup="pass", suffix=" b"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(mods))
    monkeypatch.setitem(R.STAGE_CLASSES, "fake_a", "fake_a_mod:A")
    monkeypatch.setitem(R.STAGE_CLASSES, "fake_b", "fake_b_mod:B")
    raw = minicorpus.build(tmp_path)
    cfg = minicorpus.write_config(tmp_path / "m.yaml", str(tmp_path / "rien" / "*.txt"), stages=("fake_a", "fake_b"))
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    assert "# modifié pendant le run" in b_path.read_text(encoding="utf-8")
    for m in ("fake_a_mod", "fake_b_mod"):
        sys.modules.pop(m, None)
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    assert _timing(out, "02_fake_a")["units_run"] == 0  # A n'a pas changé (son setup ne tourne plus)
    assert _timing(out, "03_fake_b")["units_run"] == 3  # B a changé après son empreinte : recalculé


def test_sortie_reecrite_depuis_le_recu_invalide_le_cache(tmp_path):
    """Un run --limit et un run complet écrivent le même fichier (part-00000) : si l'un l'a réécrit
    après le reçu de l'autre (run interrompu avant le ménage), le reçu ne doit plus être cru."""
    from pipeline.io import write_docs

    raw, cfg = _setup(tmp_path, stages=("encoding",))
    out = _run(cfg, raw, tmp_path / "o", workers=1, limit=2)
    part = out / "01_exclude_heldout" / "wikipedia_fr" / "part-00000.jsonl.zst"
    write_docs(part, [{"id": "intrus", "source": "wikipedia_fr", "text": "écrit par un autre run", "meta": {}}])
    out = _run(cfg, raw, tmp_path / "o", workers=1, limit=2)
    assert _timing(out, "01_exclude_heldout")["units_run"] == 1  # wikipedia_fr refait, les autres en cache
    ids = [d["id"] for d in _docs(out / "02_encoding")["wikipedia_fr/part-00000.jsonl.zst"]]
    assert ids == ["wikipedia_fr:1", "wikipedia_fr:2"]


def test_run_partiel_retire_les_sorties_orphelines(tmp_path):
    raw, cfg = _setup(tmp_path)
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    assert (out / "04_exact_dedup" / "gutenberg_fr").exists()
    out = _run(cfg, raw, tmp_path / "o", workers=1, sources=["wikipedia_fr"])
    final = {p.name for p in (out / "04_exact_dedup").iterdir() if p.is_dir() and not p.name.startswith("_")}
    assert final == {"wikipedia_fr"}  # l'étape globale ne garde que ce que ce run a décidé
    assert not (out / "04_exact_dedup" / "_units" / "gutenberg_fr").exists()
    assert (out / "02_encoding" / "gutenberg_fr").exists()  # étape par document : cache gardé, toujours valide


def test_etape_interrompue_sans_stats_ni_manifest(tmp_path, monkeypatch):
    from pipeline.clean.encoding import EncodingStage

    raw, cfg = _setup(tmp_path)
    out = _run(cfg, raw, tmp_path / "o", workers=1)
    assert (out / "manifest.json").exists() and (out / "02_encoding" / "stats.json").exists()

    def boom(self, doc):
        raise RuntimeError("panne")

    monkeypatch.setattr(EncodingStage, "apply", boom)
    monkeypatch.setattr(EncodingStage, "fingerprint_extra", lambda self, source=None: {"panne": 1})
    with pytest.raises(RuntimeError):
        Runner(cfg, input_dir=raw, out_root=tmp_path / "o", workers=1).run()
    # Ni l'étape interrompue ni le run ne passent pour terminés
    assert not (out / "manifest.json").exists() and not (out / "02_encoding" / "stats.json").exists()
