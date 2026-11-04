"""Régressions Wikipédia : modèles, filtre XML, intégrité et échantillon rejouable."""
import bz2
import hashlib
import json
import logging
from pathlib import Path

import pytest

from pipeline.collect import common, wikipedia as wiki, wikitext as wt
from pipeline.collect.wikipedia_render import WikipediaRenderer
from pipeline.io import read_docs
from tools.archive_wikipedia_hf2023 import relabel

FIX = Path(__file__).parent / "fixtures" / "collect"
CASES = json.loads((FIX / "wikipedia_render.json").read_text())
UNIT = "frwiki-20261001-pages-articles-multistream1.xml-p1p42.bz2"


@pytest.mark.parametrize("case", CASES, ids=[c["wikitext"][:65] for c in CASES])
def test_render(case):
    assert wt.to_text(case["wikitext"], WikipediaRenderer()) == case["expected"]


def test_unknowns_are_counted_once_including_empty_and_numeric():
    renderer = WikipediaRenderer()
    wt.to_text("{{small|{{Mystère|42}}}}{{Sans argument}}{{Mystère|Texte}}", renderer)
    assert renderer.unknown == {"mystère": 2, "sans argument": 1}
    wt.to_text("<ref>{{Caché inconnu}}</ref>{{Infobox|x={{Caché inconnu}}}}", renderer)
    assert "caché inconnu" not in renderer.unknown


def test_archive_changes_only_namespace():
    original = {"source": "wikipedia_fr", "id": "wikipedia_fr:42", "text": "Café 123.", "meta": {"x": 1}}
    migrated = relabel(original)
    assert migrated == {**original, "source": "wikipedia_fr_hf2023", "id": "wikipedia_fr_hf2023:42"}
    assert relabel(migrated) == migrated
    with pytest.raises(ValueError):
        relabel({**original, "id": "other:42"})


def test_population_does_not_leak_between_articles_or_entities():
    ctx = {"dump": "20261001", "min_chars": 1}
    renderer = WikipediaRenderer()
    first, _ = wiki.article_doc("1", "Ville", "2026-09-30T00:00:00Z",
                                "{{Infobox|population=4200}}Elle compte {{Population}} habitants.", ctx, renderer)
    second, _ = wiki.article_doc("2", "Autre ville", "2026-09-30T00:00:00Z",
                                 "Elle compte {{Population}} habitants.", ctx, renderer)
    assert "4 200" in first["text"] and "4 200" not in second["text"]
    assert renderer.unknown["population"] == 1
    assert "4 200" not in wt.to_text("{{Infobox|population=4200}}{{Population|qid=Q99}}", WikipediaRenderer())


@pytest.fixture
def local_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW_DIR", tmp_path / "raw")
    blob = bz2.compress((FIX / "frwiki_mini.xml").read_bytes())
    dest = common.source_dir(wiki.NAME) / "_dl" / UNIT
    dest.parent.mkdir(parents=True)
    dest.write_bytes(blob)
    return {"dump": "20261001", "cutoff": "2026-10-01", "sample": False, "min_chars": 1,
            "parts": {UNIT: {"url": "https://unused.invalid/", "size": len(blob),
                             "sha1": hashlib.sha1(blob).hexdigest()}}}


def test_full_xml_stream_filters_and_metadata(local_dump):
    tally = common.Tally()
    docs = list(wiki.process(UNIT, local_dump, tally, logging.getLogger("test")))
    assert [d["id"] for d in docs] == ["wikipedia_fr:1", "wikipedia_fr:7"]
    assert docs[0]["text"] == "L'algorithme compte 4 200 éléments le 1er février 2000."
    assert docs[0]["date"] == "2026-09-30" and docs[0]["meta"]["last_revision"] == "2026-09-30T12:30:00Z"
    assert docs[0]["url"] == "https://fr.wikipedia.org/wiki/Algorithme"
    assert tally.removed == {"redirection": 1, "homonymie": 1, "revision_apres_dump": 1, "date_invalide": 1}
    assert not (common.source_dir(wiki.NAME) / "_dl" / UNIT).exists()


def test_reject_bad_checksum_before_render(local_dump):
    local_dump["parts"][UNIT]["sha1"] = "0" * 40
    with pytest.raises(RuntimeError, match="empreinte"):
        list(wiki.process(UNIT, local_dump, common.Tally(), logging.getLogger("test")))
    assert (common.source_dir(wiki.NAME) / "_dl" / UNIT).exists()


def test_common_full_run_and_resume(local_dump, monkeypatch):
    monkeypatch.setattr(wiki, "prepare", lambda *a, **kw: local_dump)
    cfg = {"sources": {wiki.NAME: {"workers": 1}}}
    stats = common.run_source(wiki, cfg, workers=1)
    assert stats["docs"] == 2
    src = common.source_dir(wiki.NAME)
    docs = [d for p in sorted(src.glob("part-*.jsonl.zst")) for d in read_docs(p)]
    assert len(docs) == 2 and json.loads((src / "_state.json").read_text())["complete"]
    assert common.run_source(wiki, cfg)["complete"]


def test_sample_stops_exactly_and_saves_raw_input(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW_DIR", tmp_path / "raw")
    ctx = {"dump": "20261001", "sample": True, "sample_per_part": 10, "min_chars": 1, "parts": {UNIT: {}}}
    records = [{"pid": str(i), "title": f"Titre {i}", "ts": "2026-09-30T00:00:00Z",
                "wikitext": f"Texte {{{{Inconnu|{i}}}}}."} for i in range(10)]
    monkeypatch.setattr(wiki, "prepare", lambda *a, **kw: ctx)
    monkeypatch.setattr(wiki, "_sample_pages", lambda *a: iter(records))
    stats = common.run_source(wiki, {"sources": {wiki.NAME: {"workers": 1}}}, limit=2)
    assert stats["docs"] == 2 and stats["extra"]["unknown_templates"] == {"inconnu": 2}
    cache = common.source_dir(wiki.NAME) / "_sample_input" / f"{UNIT}.jsonl.zst"
    assert list(read_docs(cache)) == records[:2]


def test_sample_unknowns_accumulate_across_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RAW_DIR", tmp_path / "raw")
    ctx = {"dump": "20261001", "sample": True, "sample_per_part": 1, "min_chars": 1}
    record = {"pid": "1", "title": "Un titre", "ts": "2026-09-30T00:00:00Z", "wikitext": "{{Mystère|42}}"}
    monkeypatch.setattr(wiki, "_sample_pages", lambda *a: iter([record]))
    tally = common.Tally()
    for unit in ("part1", "part2"):
        list(wiki.process(unit, ctx, tally, logging.getLogger("test")))
    assert tally.extra["unknown_templates"] == {"mystère": 2}


def test_select_parts_excludes_index_and_recombined(monkeypatch):
    names = [UNIT, UNIT.replace("1.xml-p1p42", "2.xml-p43p100"),
             "frwiki-20261001-pages-articles-multistream-index.txt.bz2",
             "frwiki-20261001-pages-articles-multistream.xml.bz2"]
    status = {"jobs": {"articlesmultistreamdump": {"status": "done", "files": {
        n: {"size": 42, "sha1": "a" * 40} for n in reversed(names)}}}}
    class Reply:
        def json(self):
            return status
    monkeypatch.setattr(wiki, "http_get", lambda *a, **kw: Reply())
    assert [p["name"] for p in wiki.part_files("https://example.org", "20261001", None, None)] == names[:2]
    status["jobs"]["articlesmultistreamdump"]["status"] = "in-progress"
    with pytest.raises(RuntimeError, match="incomplet"):
        wiki.part_files("https://example.org", "20261001", None, None)


def test_empty_source_fields_only_are_removed():
    source = "* Musique :\n* Genre :\n* Durée : {{nombre|90}} minutes\n* Dates :\n** France : 1983\n"
    result = wiki.trim_empty_fields(source)
    assert result == "* Durée : {{nombre|90}} minutes\n* Dates :\n** France : 1983\n"
    assert wiki.trim_empty_fields("Il est né le .") == "Il est né le ."  # ne pas cacher les trous à la mesure


def test_apostrophe_measure_uses_source_evidence():
    from tools.check_wikipedia_quality import lost_apostrophe
    assert lost_apostrophe("Lalgorithmique est utile.", "L{{'}}'''algorithmique''' est utile.")
    assert lost_apostrophe("Lalgèbre est utile.", "L''''algèbre''' est utile.")
    assert not lost_apostrophe("L'algorithmique est utile.", "L{{'}}'''algorithmique''' est utile.")
    assert not lost_apostrophe("Dans la ville.", "Dans la ville.")


def test_quality_gate_requires_manual_review_of_current_text(tmp_path):
    from pipeline.io import write_docs
    from tools.check_wikipedia_quality import measure, sha
    src, out = tmp_path / "source", tmp_path / "review"
    docs = [{"id": f"wikipedia_fr:{i}", "text": "Né le 1er janvier 2000.", "date": "2026-09-30",
             "meta": {"title": str(i), "last_revision": "2026-09-30T00:00:00Z", "dump_part": str(i % 2)}}
            for i in range(20)]
    write_docs(src / "part-00000.jsonl.zst", docs)
    write_docs(src / "_sample_input" / "part.jsonl.zst", [{"pid": str(i), "wikitext": d["text"]}
                                                           for i, d in enumerate(docs)])
    common.write_json(src / "_state.json", {"complete": True, "mode": "sample"})
    first = measure(src, out, expected=20)
    assert first["numeric_pass"] and not first["gate_pass"] and first["manual_validated"] == 0
    review = {d["id"]: {"sha256": sha(d["text"]), "verdict": "sans_trou_visible"} for d in docs}
    common.write_json(out / "manual_review.json", review)
    assert measure(src, out, expected=20)["gate_pass"]
    docs[0]["text"] = "Une nouvelle version."
    write_docs(src / "part-00000.jsonl.zst", docs)
    assert not measure(src, out, expected=20)["gate_pass"]


def test_volumes_include_archive_without_counting_it_as_active(tmp_path, monkeypatch):
    from pipeline.collect import volumes
    monkeypatch.setattr(volumes, "OUT", tmp_path / "volumes.md")
    monkeypatch.setattr(volumes, "ROOT", tmp_path)
    monkeypatch.setattr(volumes, "source_dir", lambda name: tmp_path / "raw" / name)
    def volume(name):
        return {"docs": 2 if name == "wikipedia_fr" else 3, "bytes": 100, "date_min": "2023-11-01",
                "date_max": "2026-10-01", "files": 1, "disk": 50, "mode": "complet"}
    monkeypatch.setattr(volumes, "source_volume", volume)
    cfg = {"order": ["wikipedia_fr"], "sources": {"wikipedia_fr": {}, "wikipedia_fr_hf2023": {"enabled": False}}}
    report = volumes.write_volumes(cfg)
    assert "wikipedia_fr_hf2023" in report and "complet · archive" in report
    assert "Corpus actif : **2 documents**" in report
    assert "**Total stocké (archives comprises)** | | | **5**" in report
