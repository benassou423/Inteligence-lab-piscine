"""Mini-corpus de test : 3 sources, avec les cas que chaque étape doit traiter."""
from pathlib import Path

from pipeline.io import write_docs

FIXTURES = Path(__file__).parent

ARTICLE = (FIXTURES / "wikipedia_article.txt").read_text(encoding="utf-8")
ARTICLE_BODY = ARTICLE.split("\nNotes et références")[0].rstrip()

# Texte présent à la fois dans Wikipédia et sur une page web (doublon exact entre sources).
SHARED = ("La Saône est une rivière de l’est de la France, principal affluent du Rhône. "
          "Elle prend sa source dans les Vosges et rejoint le fleuve à Lyon, au confluent.")

EMPTY_PAGE = ("<html><head><script>var x = 1;</script></head><body><img src='banniere.png'>"
              "<script>document.write('');</script></body></html>")

SHARED_PAGE = (f"<html><body><nav><a href='/'>Accueil</a> <a href='/c'>Contact</a></nav><main><p>{SHARED}</p>"
               "</main><footer>Mentions légales © 2026</footer></body></html>")


def doc(source, ident, text, domain, html=None, url=None):
    d = {"id": f"{source}:{ident}", "source": source, "domain": domain, "text": text, "url": url,
         "date": "2026-01-01", "license": "test", "meta": {}}
    if html is not None:
        d["html"] = html
    return d


def build(root):
    """Écrit data/raw/<source>/part-00000.jsonl.zst sous root ; renvoie le dossier d'entrée."""
    raw = Path(root) / "raw"
    wiki = [
        doc("wikipedia_fr", "1", ARTICLE, "encyclopedique"),
        doc("wikipedia_fr", "2", "Lâ€™Ã©tÃ© est chaud : câ€™est la saison des fÃªtes.", "encyclopedique"),
        doc("wikipedia_fr", "3", SHARED, "encyclopedique"),
        doc("wikipedia_fr", "4", "Article réservé à la validation : il ne doit jamais être appris.", "encyclopedique"),
    ]
    guten = [
        doc("gutenberg_fr", "100", (FIXTURES / "gutenberg_moderne.txt").read_text(encoding="utf-8"), "litteraire"),
        doc("gutenberg_fr", "200", (FIXTURES / "gutenberg_ancien.txt").read_text(encoding="utf-8"), "litteraire"),
    ]
    page = (FIXTURES / "page_actu.html").read_text(encoding="utf-8")
    raw_page_text = "Accueil Politique Économie Sport Culture S’abonner Se connecter\n" \
                    "Les jardins partagés gagnent du terrain à Lyon\nMentions légales Nous contacter"
    web = [
        doc("commoncrawl", "a", raw_page_text, "web", html=page, url="https://example.fr/jardins"),
        doc("commoncrawl", "b", "", "web", html=EMPTY_PAGE, url="https://example.fr/vide"),
        doc("commoncrawl", "c", "Accueil La Saône " + SHARED, "web", html=SHARED_PAGE, url="https://example.fr/saone"),
    ]
    for source, docs in [("wikipedia_fr", wiki), ("gutenberg_fr", guten), ("commoncrawl", web)]:
        write_docs(raw / source / "part-00000.jsonl.zst", docs)
    return raw


def write_config(path, heldout_glob, stages=("encoding", "boilerplate", "exact_dedup"), name="mini", val_path=None,
                 params=None):
    """Config de test : exclude_heldout + les étapes demandées (`params` : {étape: {paramètre: valeur}})."""
    import yaml

    params = params or {}
    defaults = {"langid_score": {"download": False}, "decontaminate": {"against": str(val_path)}}
    spec = [{"name": "exclude_heldout", "params": {"glob": str(heldout_glob)}}]
    for s in stages:
        p = {**defaults.get(s, {}), **params.get(s, {})}
        spec.append({"name": s, "params": p} if p else {"name": s})
    cfg = {"name": name, "sources": "all", "source_order": ["wikipedia_fr", "gutenberg_fr", "commoncrawl"],
           "stages": spec}
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
