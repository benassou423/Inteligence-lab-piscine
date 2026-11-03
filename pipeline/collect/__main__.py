"""Point d'entrée : uv run python -m pipeline.collect <source|all|stats> [--limit N]."""
import argparse
import importlib
import sys

from pipeline.collect.common import load_config, run_source

MODULES = {
    "wikipedia_fr": "pipeline.collect.wikipedia",
    "wikisource_fr": "pipeline.collect.wikisource",
    "gutenberg_fr": "pipeline.collect.gutenberg",
    "wikinews_fr": "pipeline.collect.wikinews",
    "commoncrawl": "pipeline.collect.commoncrawl",
    "assemblee_nationale": "pipeline.collect.assemblee",
    "moliere": "pipeline.collect.moliere",
    "fineweb2_fra": "pipeline.collect.fineweb2",
}


def run_one(name, cfg, args):
    mod = importlib.import_module(MODULES[name])
    if hasattr(mod, "run"):  # sources à déroulé particulier (wikisource en 2 phases, molière)
        return mod.run(cfg, limit=args.limit, workers=args.workers, force=args.force)
    return run_source(mod, cfg, limit=args.limit, workers=args.workers, force=args.force)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pipeline.collect", description="Collecte des sources brutes → data/raw/")
    p.add_argument("source", choices=[*MODULES, "all", "stats"])
    p.add_argument("--limit", type=int, default=None, help="échantillon de N documents (écrit part-00000.jsonl.zst)")
    p.add_argument("--workers", type=int, default=None, help="nombre de processus (défaut : configs/collect.yaml)")
    p.add_argument("--force", action="store_true", help="refaire même si déjà collecté")
    p.add_argument("--config", default=None, help="fichier de config (défaut : configs/collect.yaml)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.source == "stats":
        from pipeline.collect.volumes import write_volumes
        print(write_volumes(cfg))
        return 0
    names = [args.source] if args.source != "all" else [
        n for n in cfg.get("order", MODULES) if n in MODULES and cfg["sources"].get(n, {}).get("enabled", True)]
    for name in names:
        run_one(name, cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
