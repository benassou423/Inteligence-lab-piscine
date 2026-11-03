"""Moteur du pipeline : enchaîne les étapes d'une config YAML, en parallèle.

    uv run python -m pipeline.run --config configs/pipeline/nettoye.yaml \
        [--sources a,b] [--limit N] [--workers 8] [--force]

Sorties (data/stages/<nom>/) :
    <NN>_<etape>/<source>/part-*.jsonl.zst   documents après l'étape
    <NN>_<etape>/stats.json                  docs et octets entrés/sortis, raisons, débits
    <NN>_<etape>/_rejected_sample/           documents retirés (2 000 max par source)
    funnel.md, manifest.json, run.log

Reprise : chaque fichier traité laisse un reçu (`_units/`) avec une empreinte
(paramètres, code de l'étape et versions de ses bibliothèques, entrées). Relancer la
même commande saute ce qui est déjà fait et refait seulement ce qui a changé.
L'empreinte du code est prise au lancement, quand le code est chargé : modifier un
fichier pendant un run invalide son cache au run suivant (et non l'inverse).
Une étape sans `stats.json` est en cours ou interrompue ; `manifest.json` n'existe
que pour un run terminé.
"""
import argparse
import importlib
import importlib.metadata
import inspect
import json
import logging
import multiprocessing as mp
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xxhash
import yaml
import zstandard

from pipeline import stats as S
from pipeline.clean.base import utf8_len
from pipeline.io import read_docs, write_docs

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1  # à incrémenter si le format des sorties du runner change

STAGE_CLASSES = {
    "exclude_heldout": "pipeline.clean.heldout:ExcludeHeldoutStage",
    "encoding": "pipeline.clean.encoding:EncodingStage",
    "boilerplate": "pipeline.clean.boilerplate:BoilerplateStage",
    "langid_score": "pipeline.clean.langid_score:LangIdScoreStage",
    "langid": "pipeline.clean.langid:LangIdStage",
    "reflow": "pipeline.clean.reflow:ReflowStage",
    "decontaminate": "pipeline.dedup.decontaminate:DecontaminateStage",
    "exact_dedup": "pipeline.dedup.exact:ExactDedupStage",
}
CODE_DEPS = {"boilerplate": ["pipeline.clean.encoding"], "decontaminate": ["ftb.contamination"]}
# Bibliothèques dont la version change la sortie d'une étape : dans l'empreinte du code.
LIB_DEPS = {"encoding": ["ftfy"], "boilerplate": ["ftfy", "trafilatura", "lxml"],
            "langid_score": ["fasttext-numpy2"]}

log = logging.getLogger("pipeline")


# ----------------------------------------------------------------------- config

@dataclass
class Context:
    """Ce que les étapes savent du run."""

    root: Path
    config_name: str
    sources: list


@dataclass
class Unit:
    """Une unité de travail : un fichier d'entrée (ou une source entière avec --limit)."""

    key: str  # « source/part-00000.jsonl.zst » ou « source/__limit200 »
    source: str
    inputs: list
    limit: int | None
    fingerprint: str = ""
    size: int = 0

    @property
    def record_name(self):
        return self.key.split("/", 1)[1]


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "name" not in cfg:
        cfg["name"] = Path(path).stem
    stages = cfg.get("stages") or []
    norm = []
    for s in stages:
        if isinstance(s, str):
            s = {"name": s}
        if s.get("name") not in STAGE_CLASSES:
            raise ValueError(f"Étape inconnue : {s.get('name')!r} (connues : {', '.join(STAGE_CLASSES)})")
        norm.append({"name": s["name"], "params": s.get("params") or {}})
    if not norm or norm[0]["name"] != "exclude_heldout":
        raise ValueError("La première étape doit être exclude_heldout (la val ne sert jamais à l'entraînement).")
    if any(s["name"] == "exclude_heldout" for s in norm[1:]):
        raise ValueError("exclude_heldout ne doit apparaître qu'une fois, en première position.")
    for i, s in enumerate(norm):
        # langid applique un seuil aux notes de langid_score : elles doivent porter sur le même texte.
        if s["name"] == "langid" and norm[i - 1]["name"] != "langid_score":
            raise ValueError("langid doit suivre directement langid_score (qui calcule les notes de langue).")
    cfg["stages"] = norm
    cfg.setdefault("input", "data/raw")
    cfg.setdefault("sources", "all")
    return cfg


def resolve(path, root=ROOT):
    p = Path(path)
    return p if p.is_absolute() else root / p


def discover_sources(input_dir):
    if not input_dir.exists():
        return []
    return sorted(p.name for p in input_dir.iterdir() if p.is_dir() and any(p.glob("part-*.jsonl.zst")))


def order_sources(found, wanted, order):
    """Sources à traiter, dans l'ordre qui décide qui garde un doublon."""
    if wanted in (None, "all", ["all"]):
        chosen = list(found)
    else:
        chosen = [s for s in wanted if s in found]
        for s in wanted:
            if s not in found:
                log.warning("source %s absente de l'entrée : ignorée", s)
    rank = {s: i for i, s in enumerate(order or [])}
    return sorted(chosen, key=lambda s: (rank.get(s, len(rank)), s))


def build_stage(spec, ctx):
    mod_name, cls_name = STAGE_CLASSES[spec["name"]].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(spec["params"], ctx)


def lib_version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_digest(stage):
    """Empreinte du code de l'étape (et des versions de ses bibliothèques) : la changer invalide son cache."""
    files = [inspect.getfile(type(stage)), str(ROOT / "pipeline/clean/base.py")]
    files += [importlib.import_module(m).__file__ for m in CODE_DEPS.get(stage.name, [])]
    h = xxhash.xxh64()
    for f in sorted(set(files)):
        h.update(Path(f).read_bytes())
    for dist in LIB_DEPS.get(stage.name, []):
        h.update(f"{dist}=={lib_version(dist)}".encode())
    return h.hexdigest()


def digest(obj):
    return xxhash.xxh64(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- lecture sûre

def iter_input(path):
    """Lit un fichier ; s'il est tronqué (collecte en cours), s'arrête proprement."""
    try:
        yield from read_docs(path)
    except (zstandard.ZstdError, json.JSONDecodeError, EOFError, UnicodeDecodeError) as e:
        log.warning("%s : lecture interrompue (%s) — fichier incomplet ?", path, type(e).__name__)
        _W.setdefault("truncated", []).append(str(path))


# ------------------------------------------------------------------- workers

_W = {}  # état hérité par les workers (fork) : étape, options, dossier de sortie


def _sampler(samplers, group, stratum):
    key = (group, stratum)
    if key not in samplers:
        k = _W["rejected_k"] if group == "rejected" else _W["stage"].sample_sizes.get(group, 200)
        samplers[key] = S.BottomK(k)
    return samplers[key]


def _rejected_record(doc, reason):
    rec = {k: v for k, v in doc.items() if k not in ("html", "text")}
    text = doc.get("text") or ""
    rec["text"] = text[: _W["sample_chars"]]
    rec["text_bytes"] = utf8_len(text)
    if doc.get("html"):
        rec["html_bytes"] = utf8_len(doc["html"])
    rec["reason"] = reason
    return rec


def _process_unit(unit):
    """Applique l'étape à une unité, écrit la sortie et le reçu. Tourne dans un worker."""
    stage, stage_dir = _W["stage"], _W["stage_dir"]
    keep_html, is_global = _W["keep_html"], stage.kind == "global"
    cpu0, wall0 = time.process_time(), time.time()
    _W["truncated"] = []
    st = S.new_source_stats()
    removed, counters = Counter(), Counter()
    samplers, captures, outputs = {}, [], []
    out_dir = stage_dir / unit.source
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = [unit.limit]

    def gen(path):
        for line, doc in enumerate(iter_input(path)):
            if remaining[0] is not None:
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
            text = doc.get("text") or ""
            b_in = utf8_len(text)
            st["docs_in"] += 1
            st["bytes_in"] += b_in
            out = stage.apply_at(doc, unit.key, line) if is_global else stage.apply(doc)
            if out.counters:
                counters.update(out.counters)
            if out.capture:
                captures.append(out.capture)
            d = out.doc
            if out.samples:
                for group, stratum, rec in out.samples:
                    h = S.sample_hash(group, d["id"])
                    sm = _sampler(samplers, group, stratum)
                    if sm.wants(h):
                        sm.add(h, d["id"], rec() if callable(rec) else rec)
            if out.keep:
                if not keep_html:
                    d.pop("html", None)
                new_text = d.get("text") or ""
                if new_text is not text and new_text != text:
                    st["docs_modified"] += 1
                st["docs_out"] += 1
                st["bytes_out"] += utf8_len(new_text)
                yield d
            else:
                removed[out.reason] += 1
                st["bytes_removed_docs"] += b_in
                h = S.sample_hash("rejected", d["id"])
                sm = _sampler(samplers, "rejected", "")
                if sm.wants(h):
                    sm.add(h, d["id"], _rejected_record(d, out.reason))

    for path in unit.inputs:
        if remaining[0] is not None and remaining[0] <= 0:
            break
        out_path = out_dir / path.name
        tmp = out_path.with_name(out_path.name + ".tmp")
        write_docs(tmp, gen(path))
        tmp.replace(out_path)
        outputs.append(path.name)

    st["removed_by_reason"] = dict(removed)
    if _W["truncated"]:
        counters["input_truncated"] += len(_W["truncated"])
    st["counters"] = dict(counters)
    st["cpu_seconds"] = time.process_time() - cpu0
    st["units"] = 1
    samples = {}
    for (group, stratum), sm in samplers.items():
        samples.setdefault(group, {})[stratum] = sm.items()
    result = {"key": unit.key, "source": unit.source, "fingerprint": unit.fingerprint,
              "outputs": outputs, "output_sigs": [_file_sig(out_dir / o) for o in outputs],
              "stats": st, "wall_seconds": time.time() - wall0,
              "captures": captures, "samples": samples}
    _save_record(stage_dir, unit, result)
    return result


def _scan_path(stage_dir, unit, stage):
    """Fichier de résultats de la passe 1 : clés fixes (.keys.npy) ou enregistrements (.records.json)."""
    suffix = ".records.json" if getattr(stage, "scan_kind", "keys") == "records" else ".keys.npy"
    return stage_dir / "_units" / unit.source / (unit.record_name + suffix)


def _load_scan(stage_dir, unit, stage):
    p = _scan_path(stage_dir, unit, stage)
    return S.read_json(p) if p.name.endswith(".json") else np.load(p)


def _scan_unit(unit):
    """Passe 1 d'une étape globale : une empreinte (ou un enregistrement) par document."""
    stage, stage_dir = _W["stage"], _W["stage_dir"]
    cpu0 = time.process_time()
    out = _scan_path(stage_dir, unit, stage)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    if getattr(stage, "scan_kind", "keys") == "records":
        records = []
        for line, doc in enumerate(iter_input(unit.inputs[0])):
            r = stage.scan_doc(doc)
            if r is not None:
                records.append([line, r])
            n += 1
        tmp.write_text(json.dumps({"n": n, "records": records}), encoding="utf-8")
    else:
        buf = bytearray()
        for doc in iter_input(unit.inputs[0]):
            buf += stage.key(doc)
            n += 1
        arr = np.frombuffer(bytes(buf), dtype=np.uint64).reshape(-1, 2) if n else np.zeros((0, 2), np.uint64)
        with open(tmp, "wb") as f:
            np.save(f, arr)
    tmp.replace(out)
    rec_dir = out.parent
    rec = {"key": unit.key, "fingerprint": unit.fingerprint, "n": n, "cpu_seconds": time.process_time() - cpu0}
    S.write_json(rec_dir / (unit.record_name + ".scan.json"), rec)
    return rec


# ------------------------------------------------------------------- reçus

RECORD_SUFFIXES = (".samples.jsonl.zst", ".keys.npy", ".records.json", ".scan.json", ".json")


def _record_base(filename):
    """« part-00000.jsonl.zst.scan.json » -> « part-00000.jsonl.zst »."""
    for suf in RECORD_SUFFIXES:
        if filename.endswith(suf):
            return filename[: -len(suf)]
    return filename


def _file_sig(path):
    """[nom, taille, date de modification] d'un fichier de sortie, ou None s'il n'existe pas."""
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return None
    return [Path(path).name, st.st_size, st.st_mtime_ns]


def _record_paths(stage_dir, unit):
    d = stage_dir / "_units" / unit.source
    return d / (unit.record_name + ".json"), d / (unit.record_name + ".samples.jsonl.zst")


def _save_record(stage_dir, unit, result):
    rec_path, smp_path = _record_paths(stage_dir, unit)
    rows = [{"group": g, "stratum": s, "hash": h, "record": r}
            for g, by_s in result["samples"].items() for s, items in by_s.items() for h, r in items]
    tmp = smp_path.with_name(smp_path.name + ".tmp")
    write_docs(tmp, rows)
    tmp.replace(smp_path)
    S.write_json(rec_path, {k: v for k, v in result.items() if k != "samples"})


def _load_record(stage_dir, unit):
    """Reçu valide (même empreinte, sorties présentes) ou None."""
    rec_path, smp_path = _record_paths(stage_dir, unit)
    if not rec_path.exists() or not smp_path.exists():
        return None
    try:
        rec = S.read_json(rec_path)
    except (OSError, json.JSONDecodeError):
        return None
    if rec.get("fingerprint") != unit.fingerprint:
        return None
    # Sorties présentes ET identiques à celles écrites par ce reçu (taille, date) : un fichier
    # réécrit depuis (par un run --limit puis complet interrompu, à la main…) invalide le reçu.
    sigs = [_file_sig(stage_dir / unit.source / o) for o in rec["outputs"]]
    if None in sigs or sigs != rec.get("output_sigs"):
        return None
    samples = {}
    for row in read_docs(smp_path):
        samples.setdefault(row["group"], {}).setdefault(row["stratum"], []).append((row["hash"], row["record"]))
    rec["samples"] = samples
    return rec


# -------------------------------------------------------------- exécution

def _pool_map(fn, units, workers):
    """Exécute fn sur les unités ; parallèle par fork (les workers héritent de _W)."""
    if not units:
        return
    units = sorted(units, key=lambda u: -u.size)  # les gros fichiers d'abord : meilleur équilibrage
    if workers <= 1 or len(units) == 1:
        for u in units:
            yield fn(u)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(units))) as pool:
        yield from pool.imap_unordered(fn, units)


class Runner:
    def __init__(self, config_path, sources=None, limit=None, workers=8, force=False,
                 input_dir=None, out_root=None, root=ROOT, rejected_k=2000, sample_chars=5000):
        self.cfg = load_config(config_path)
        self.config_path = Path(config_path)
        self.root = Path(root)
        self.name = self.cfg["name"]
        self.input_dir = resolve(input_dir or self.cfg["input"], self.root)
        self.out_root = resolve(out_root or self.cfg.get("output", "data/stages"), self.root) / self.name
        self.limit = limit
        self.workers = max(1, int(workers))
        self.force = force
        self.rejected_k = rejected_k
        self.sample_chars = sample_chars
        found = discover_sources(self.input_dir)
        wanted = sources if sources else self.cfg.get("sources", "all")
        self.sources = order_sources(found, wanted, self.cfg.get("source_order"))
        self.partial = bool(sources) and set(self.sources) < set(found)  # --sources : sous-ensemble
        self.ctx = Context(root=self.root, config_name=self.name, sources=self.sources)

    # ---- empreintes
    def _input_fingerprints(self):
        fps = {}
        for s in self.sources:
            files = sorted((self.input_dir / s).glob("part-*.jsonl.zst"))
            sig = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files]
            fps[s] = digest({"input": str(self.input_dir / s), "files": sig, "limit": self.limit})
        return fps

    # ---- programme principal
    def run(self):
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        t0 = time.time()
        if not self.sources:
            log.error("Aucune source trouvée dans %s", self.input_dir)
            return 1
        log.info("config %s · sources %s · limit %s · %d workers", self.name, ", ".join(self.sources),
                 self.limit, self.workers)
        (self.out_root / "manifest.json").unlink(missing_ok=True)  # réécrit seulement si le run va au bout
        stages = [build_stage(spec, self.ctx) for spec in self.cfg["stages"]]
        for i, stage in enumerate(stages):
            stage.bind(stages[i + 1:])
        # Empreinte du code prise maintenant, quand il est chargé (et non au début de chaque étape :
        # un fichier modifié pendant le run serait sinon associé à des sorties de l'ancien code).
        codes = [code_digest(stage) for stage in stages]
        if self.partial and any(s.kind == "global" for s in stages):
            log.warning("--sources %s : les étapes globales (doublons, décontamination) ne voient que ces "
                        "sources ; les sorties des autres sources sont retirées des étapes globales.",
                        ",".join(self.sources))
        upstream_fp = self._input_fingerprints()
        prev_dir = self.input_dir
        all_stats = []
        for i, (spec, stage) in enumerate(zip(self.cfg["stages"], stages)):
            stage_dir = self.out_root / f"{i + 1:02d}_{spec['name']}"
            if self.force and stage_dir.exists():
                shutil.rmtree(stage_dir)
            (stage_dir / "stats.json").unlink(missing_ok=True)  # l'étape n'est complète qu'à sa réécriture
            keep_html = any(s.uses_html for s in stages[i + 1:])
            base = {"stage": spec["name"], "index": i, "params": spec["params"], "keep_html": keep_html,
                    "code": codes[i], "format": FORMAT_VERSION}
            t_stage = time.time()
            if stage.kind == "global":
                st, upstream_fp = self._run_global(stage, stage_dir, prev_dir, base, upstream_fp, keep_html, i)
            else:
                st, upstream_fp = self._run_map(stage, stage_dir, prev_dir, base, upstream_fp, keep_html, i)
            st["timing"]["wall_seconds"] = round(time.time() - t_stage, 2)
            S.write_json(stage_dir / "stats.json", st)
            all_stats.append(st)
            stage.teardown()
            log.info("[%s] %s docs -> %s docs · %s Mo -> %s Mo · %s", stage_dir.name,
                     S.fmt_int(st["total"]["docs_in"]), S.fmt_int(st["total"]["docs_out"]),
                     S.fmt_mb(st["total"]["bytes_in"]), S.fmt_mb(st["total"]["bytes_out"]),
                     S.fmt_seconds(st["timing"]["wall_seconds"]))
            prev_dir = stage_dir
        self._remove_stale_stage_dirs(stages)
        self._remove_orphan_sources(stages)
        wall = time.time() - t0
        run_info = {"config": self.name, "input": self._rel(self.input_dir), "limit": self.limit,
                    "workers": self.workers, "sources": self.sources, "wall_seconds": wall,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        funnel = S.render_funnel(run_info, all_stats)
        (self.out_root / "funnel.md").write_text(funnel, encoding="utf-8")
        S.write_json(self.out_root / "manifest.json", {
            **run_info, "config_path": self._rel(self.config_path.resolve()),
            "stages": [{"index": s["index"] + 1, "name": s["stage"], "dir": s["dir"], "params": s["params"]}
                       for s in all_stats],
            "final_dir": self._rel(prev_dir),
        })
        log.info("terminé en %s · entonnoir : %s", S.fmt_seconds(wall), self._rel(self.out_root / "funnel.md"))
        return 0

    def _remove_stale_stage_dirs(self, stages):
        """Supprime les dossiers « NN_etape » d'une ancienne version de la config (sorties périmées)."""
        current = {f"{i + 1:02d}_{s.name}" for i, s in enumerate(stages)}
        for d in self.out_root.iterdir():
            if d.is_dir() and re.fullmatch(r"\d\d_[a-z_]+", d.name) and d.name not in current:
                log.info("dossier d'étape périmé supprimé : %s", d.name)
                shutil.rmtree(d)

    def _remove_orphan_sources(self, stages):
        """Après un run partiel (--sources), retire des étapes globales (et de la dernière) les sorties
        des autres sources : elles ne correspondent plus à la décision globale de ce run, et
        bench prendrait tout le dossier de la dernière étape."""
        first_global = next((i for i, s in enumerate(stages) if s.kind == "global"), len(stages) - 1)
        for i in range(first_global, len(stages)):
            stage_dir = self.out_root / f"{i + 1:02d}_{stages[i].name}"
            for d in (stage_dir, stage_dir / "_units"):
                if not d.exists():
                    continue
                for sub in d.iterdir():
                    if sub.is_dir() and not sub.name.startswith("_") and sub.name not in self.sources:
                        log.info("%s/%s : sorties d'une source hors de ce run, retirées", stage_dir.name, sub.name)
                        shutil.rmtree(sub)

    def _rel(self, p):
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    def _setup_logging(self):
        log.setLevel(logging.INFO)
        for h in list(log.handlers):
            log.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(self.out_root / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(sh)
        log.addHandler(fh)
        log.propagate = False

    # ---- unités
    def _units(self, prev_dir, source, stage_index, fp):
        files = sorted((prev_dir / source).glob("part-*.jsonl.zst"))
        if stage_index == 0 and self.limit is not None:
            u = Unit(key=f"{source}/__limit{self.limit}", source=source, inputs=files, limit=self.limit)
            u.size = sum(f.stat().st_size for f in files)
            units = [u]
        else:
            units = [Unit(key=f"{source}/{f.name}", source=source, inputs=[f], limit=None, size=f.stat().st_size)
                     for f in files]
        for u in units:
            u.fingerprint = digest({"stage": fp, "unit": u.key})
        return units

    def _execute(self, stage, stage_dir, units, keep_html, label):
        """Fait tourner les unités non encore faites ; renvoie tous les résultats."""
        results, todo = [], []
        for u in units:
            rec = _load_record(stage_dir, u)
            if rec is None:
                todo.append(u)
            else:
                results.append(rec)
        if todo:
            stage.setup()
        _W.clear()
        _W.update(stage=stage, stage_dir=stage_dir, keep_html=keep_html,
                  rejected_k=self.rejected_k, sample_chars=self.sample_chars)
        n_done, t0 = 0, time.time()
        docs = 0
        total = len(todo)
        if total:
            log.info("[%s] %d unités à traiter (%d déjà faites)", label, total, len(results))
        for res in _pool_map(_process_unit, todo, self.workers):
            results.append(res)
            n_done += 1
            docs += res["stats"]["docs_in"]
            el = time.time() - t0
            eta = el / n_done * (total - n_done)
            log.info("[%s] %d/%d · %s docs · %s docs/s · reste ~%s", label, n_done, total,
                     S.fmt_int(docs), S.fmt_rate(docs / el if el else None), S.fmt_seconds(eta))
        return results, len(todo), len(units) - len(todo)

    def _cleanup(self, stage_dir, results):
        """Retire les sorties et reçus périmés (fichiers d'un ancien run)."""
        keep_out = {(r["source"], o) for r in results for o in r["outputs"]}
        keep_rec = {r["key"] for r in results}
        for s in self.sources:
            d = stage_dir / s
            if d.exists():
                for f in d.iterdir():
                    if f.name.endswith(".tmp") or (f.name.startswith("part-") and (s, f.name) not in keep_out):
                        f.unlink()
            rd = stage_dir / "_units" / s
            if rd.exists():
                for f in rd.iterdir():
                    if f.name.endswith(".tmp") or f"{s}/{_record_base(f.name)}" not in keep_rec:
                        f.unlink()

    def _run_map(self, stage, stage_dir, prev_dir, base, upstream_fp, keep_html, i):
        fps = {s: digest({**base, "extra": stage.fingerprint_extra(s), "upstream": upstream_fp[s]})
               for s in self.sources}
        units = [u for s in self.sources for u in self._units(prev_dir, s, i, fps[s])]
        results, n_run, n_cached = self._execute(stage, stage_dir, units, keep_html, stage_dir.name)
        self._cleanup(stage_dir, results)
        st = self._stage_stats(stage, stage_dir, base, fps, results, n_run, n_cached, captures=[])
        return st, fps

    def _run_global(self, stage, stage_dir, prev_dir, base, upstream_fp, keep_html, i):
        # Passe 1 : empreintes par fichier (en cache, indépendantes des autres sources)
        base = {**base, "extra": stage.fingerprint_extra()}
        scan_fp = {s: digest({**base, "pass": "scan", "upstream": upstream_fp[s]}) for s in self.sources}
        units = [u for s in self.sources for u in self._units(prev_dir, s, i, scan_fp[s])]
        todo = []
        for u in units:
            rec_p = stage_dir / "_units" / u.source / (u.record_name + ".scan.json")
            ok = (rec_p.exists() and _scan_path(stage_dir, u, stage).exists()
                  and S.read_json(rec_p).get("fingerprint") == u.fingerprint)
            if not ok:
                todo.append(u)
        if todo:
            stage.setup()  # chargé avant le fork : les workers en héritent
        _W.clear()
        _W.update(stage=stage, stage_dir=stage_dir)
        if todo:
            log.info("[%s] passe 1 : %d fichiers à analyser", stage_dir.name, len(todo))
        scan_cpu = sum(r["cpu_seconds"] for r in _pool_map(_scan_unit, todo, self.workers))
        scanned = []
        for u in units:  # ordre global : sources dans l'ordre de la config, fichiers triés
            scanned.append((u.key, u.source, _load_scan(stage_dir, u, stage)))
        plan_digest = stage.plan(scanned)
        del scanned
        # Passe 2 : application de la décision, fichier par fichier
        gfp = digest({**base, "upstream": [upstream_fp[s] for s in self.sources], "plan": plan_digest})
        for u in units:
            u.fingerprint = digest({"stage": gfp, "unit": u.key})
        results, n_run, n_cached = self._execute(stage, stage_dir, units, keep_html, stage_dir.name)
        self._cleanup(stage_dir, results)
        captures = [c for r in results for c in r.get("captures", [])]
        st = self._stage_stats(stage, stage_dir, base, {s: gfp for s in self.sources}, results,
                               n_run, n_cached, captures=captures, extra_cpu=scan_cpu)
        return st, {s: gfp for s in self.sources}

    # ---- statistiques et échantillons
    def _stage_stats(self, stage, stage_dir, base, fps, results, n_run, n_cached, captures, extra_cpu=0.0):
        per_source = {s: S.new_source_stats() for s in self.sources}
        merged = {}  # group -> source -> stratum -> [items]
        for r in sorted(results, key=lambda r: r["key"]):
            S.merge_source_stats(per_source[r["source"]], r["stats"])
            for group, by_s in r.get("samples", {}).items():
                for stratum, items in by_s.items():
                    merged.setdefault(group, {}).setdefault(r["source"], {}).setdefault(stratum, []).append(items)
        # Fusion bottom-k, puis écriture des échantillons
        final = {}
        for group, by_source in merged.items():
            k = self.rejected_k if group == "rejected" else stage.sample_sizes.get(group, 200)
            for source, by_stratum in by_source.items():
                for stratum, lists in by_stratum.items():
                    final.setdefault(group, {}).setdefault(source, []).extend(
                        rec for _, rec in S.merge_bottom_k(lists, k))
        flat = {g: [rec for recs in by_src.values() for rec in recs] for g, by_src in final.items()}
        report, report_md = stage.finalize(captures, flat)
        self._write_samples(stage_dir, final, per_source)
        out_sources = {}
        for s, st in per_source.items():
            entry = {k: st[k] for k in S.CORE_KEYS}
            entry["removed_by_reason"] = dict(sorted(st["removed_by_reason"].items(), key=lambda kv: -kv[1]))
            entry["docs_modified"] = st["docs_modified"]
            entry["bytes_removed_docs"] = st["bytes_removed_docs"]
            entry["summary"] = stage.describe(st)
            entry["details"] = stage.summarize(Counter(st["counters"]))
            entry["cpu_seconds"] = round(st["cpu_seconds"], 3)
            entry.update(S.rates(st))
            out_sources[s] = entry
        tot = S.total_stats(per_source)
        total = {k: tot[k] for k in S.CORE_KEYS}
        total["removed_by_reason"] = dict(sorted(tot["removed_by_reason"].items(), key=lambda kv: -kv[1]))
        total["docs_modified"] = tot["docs_modified"]
        total["bytes_removed_docs"] = tot["bytes_removed_docs"]
        cpu = tot["cpu_seconds"] + extra_cpu
        timing = {"cpu_seconds": round(cpu, 2), "workers": self.workers, "units_run": n_run, "units_cached": n_cached,
                  **S.rates({"docs_in": tot["docs_in"], "bytes_in": tot["bytes_in"], "cpu_seconds": cpu})}
        return {
            "config": self.name, "stage": stage.name, "index": base["index"],
            "dir": stage_dir.name, "kind": stage.kind, "params": base["params"],
            "keep_html": base["keep_html"], "limit": self.limit, "fingerprints": fps,
            "sources": out_sources, "total": total, "timing": timing,
            "samples": sorted(f"_{g}_sample" for g in final),
            "report": report, "report_md": report_md,
        }

    def _write_samples(self, stage_dir, final, per_source):
        for old in stage_dir.glob("_*_sample"):
            if old.is_dir():
                shutil.rmtree(old)
        for group, by_source in final.items():
            d = stage_dir / f"_{group}_sample"
            d.mkdir(parents=True, exist_ok=True)
            for source, recs in by_source.items():
                if group == "rejected":
                    recs = sorted(recs, key=lambda r: (r.get("reason") or "", r.get("id")))
                elif group == "strata":
                    recs = sorted(recs, key=lambda r: (r.get("stratum") or "", r.get("id")))
                write_docs(d / f"{source}.jsonl.zst", recs)
            (d / "apercu.md").write_text(_preview(group, by_source, per_source), encoding="utf-8")


def _short(v, n=300):
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[:n] + "…"


def _preview(group, by_source, per_source, per_reason=5):
    """Aperçu lisible d'un échantillon (quelques exemples par source et par raison)."""
    lines = [f"# Échantillon « {group} »", "",
             "Quelques exemples par source. Fichiers complets : `<source>.jsonl.zst` dans ce dossier.", ""]
    for source, recs in by_source.items():
        lines.append(f"## {source} ({len(recs)} dans l'échantillon)")
        lines.append("")
        if group == "rejected":
            by_reason = {}
            for r in recs:
                by_reason.setdefault(r.get("reason"), []).append(r)
            total = per_source.get(source, {}).get("removed_by_reason", {})
            for reason, rs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"### {reason} — {S.fmt_int(total.get(reason, len(rs)))} retirés au total")
                lines.append("")
                for r in rs[:per_reason]:
                    meta = r.get("meta") or {}
                    extra = " · ".join(f"{k}={v}" for k, v in meta.items() if k.startswith(("langid", "exact")))
                    lines.append(f"- `{r.get('id')}` {extra}  \n  {_short(r.get('text', ''))}")
                lines.append("")
        else:
            for r in recs[:10]:
                fields = [f"**{k}** : {_short(v, 240)}" for k, v in r.items()
                          if k not in ("id", "source") and isinstance(v, str) and v]
                nums = [f"{k}={v}" for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                lines.append(f"- `{r.get('id')}` {' · '.join(nums)}")
                for f in fields:
                    lines.append(f"  - {f}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lance une config du pipeline (étapes de nettoyage).")
    ap.add_argument("--config", required=True, help="configs/pipeline/<nom>.yaml")
    ap.add_argument("--sources", help="sous-ensemble de sources, séparées par des virgules")
    ap.add_argument("--limit", type=int, help="N premiers documents par source (essai rapide)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="ignore le cache et recalcule tout")
    ap.add_argument("--input", help="remplace le dossier d'entrée de la config")
    ap.add_argument("--out", help="dossier racine des sorties (défaut data/stages)")
    args = ap.parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    runner = Runner(args.config, sources=sources, limit=args.limit, workers=args.workers,
                    force=args.force, input_dir=args.input, out_root=args.out)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
