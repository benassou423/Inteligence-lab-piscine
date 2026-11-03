"""Mini-corpus synthétique et déterministe pour les tests du banc d'essai.

    make_corpus(data_root, layout="raw")    → data_root/raw/<source>/part-00000.jsonl.zst
    make_corpus(data_root, layout="stage")  → data_root/stages/nettoye/02_fin/<source>/... (+ stats.json)

Le texte est fabriqué à partir de gabarits (aucun texte réel) : de quoi tester le mix
(sélection, répétition, ordre) sans dépendre des vraies données.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pipeline.io import write_docs

PRENOMS = ["Jean", "Marie", "Pierre", "Claire", "Louis", "Sophie", "Henri", "Lucie", "Paul", "Jeanne",
           "Victor", "Émile", "Hélène", "Antoine", "Camille", "Gaston", "Mathilde", "Étienne"]
NOMS = ["Dupont", "Martin", "Bernard", "Lefèvre", "Moreau", "Laurent", "Girard", "Roux", "Fournier",
        "Mercier", "Blanc", "Garnier", "Chevalier", "Rousseau", "Faure", "Gauthier"]
VILLES = ["Amiens", "Rouen", "Lyon", "Nantes", "Rennes", "Dijon", "Brest", "Nancy", "Tours", "Angers",
          "Limoges", "Nîmes", "Arles", "Metz", "Reims", "Caen", "Pau", "Annecy", "Québec", "Liège", "Genève"]
REGIONS = ["Hauts-de-France", "Normandie", "Bretagne", "Occitanie", "Grand Est", "Auvergne-Rhône-Alpes",
           "Nouvelle-Aquitaine", "Pays de la Loire", "Bourgogne-Franche-Comté", "Centre-Val de Loire"]
RIVIERES = ["la Somme", "la Seine", "la Loire", "le Rhône", "la Garonne", "la Meuse", "la Vilaine", "l'Oise"]
METIERS = ["peintre", "écrivain", "médecin", "ingénieur", "poète", "architecte", "musicien", "historien"]
SUJETS = ["le vieux marchand", "la jeune femme", "l'enfant", "le capitaine", "la servante", "le notaire",
          "le curé du village", "la comtesse", "le meunier", "l'étranger", "le soldat", "la veuve"]
VERBES = ["regardait", "traversait", "attendait", "quittait", "rejoignait", "observait", "écoutait",
          "cherchait", "longeait", "retrouvait"]
LIEUX = ["la place du marché", "le vieux pont", "la forêt sombre", "l'église", "le port", "la grande salle",
         "le jardin", "la route de la ville", "les champs", "la maison du notaire"]
MOMENTS = ["au lever du jour", "à la tombée de la nuit", "ce soir-là", "le lendemain", "depuis longtemps",
           "sans rien dire", "avec inquiétude", "d'un pas lent", "malgré la pluie", "jusqu'à l'aube"]
SENTIMENTS = ["la tristesse", "une joie profonde", "l'inquiétude", "la colère", "l'espérance", "la honte"]
SUJETS_PRESSE = ["le conseil municipal", "la préfecture", "le gouvernement", "la région", "le syndicat",
                 "l'association des riverains", "la direction de l'hôpital", "le tribunal"]
OBJETS_PRESSE = ["un nouveau plan de circulation", "la rénovation de la gare", "une hausse des tarifs",
                 "la fermeture d'une école", "un budget de deux millions d'euros", "la construction d'un pont"]
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
THEMES_AN = ["le projet de loi de finances", "la réforme des retraites", "la santé publique",
             "l'éducation nationale", "la transition énergétique", "la sécurité routière", "le logement"]
NEWS_HOSTS = ["www.lemonde.fr", "www.ouest-france.fr", "ici.radio-canada.ca", "www.lesoir.be",
              "www.letemps.ch", "www.ledauphine.com", "www.lapresse.ca", "www.francetvinfo.fr"]
OTHER_HOSTS = ["www.blog-cuisine-exemple.fr", "boutique-exemple.com", "forum.exemple.org", "www.voyages-exemple.fr"]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _person(r):
    return f"{r.choice(PRENOMS)} {r.choice(NOMS)}"


def _narration(r) -> str:
    out = []
    for _ in range(r.randint(3, 6)):
        k = r.random()
        if k < 0.5:
            out.append(f"{_cap(r.choice(SUJETS))} {r.choice(VERBES)} {r.choice(LIEUX)} {r.choice(MOMENTS)}.")
        elif k < 0.8:
            out.append(f"Il y avait dans son regard {r.choice(SENTIMENTS)}, et {r.choice(SUJETS)} le savait bien.")
        else:
            out.append(f"— {r.choice(['Bonjour', 'Venez', 'Écoutez', 'Partez'])}, dit {r.choice(SUJETS)}. "
                       f"Qu'attendez-vous de moi ?")
    return " ".join(out)


def _encyclo(r, i: int) -> tuple[str, str]:
    if i % 2 == 0:
        v = r.choice(VILLES) + f"-sur-{r.choice(['Mer', 'Loire', 'Seine', 'Marne'])}" + (f" {i}" if i > 20 else "")
        parts = [f"{v} est une commune française située dans la région {r.choice(REGIONS)}. "
                 f"Elle compte {r.randint(300, 90000)} habitants en {r.randint(1990, 2022)}."]
        for sec in ["Géographie", "Histoire", "Économie", "Patrimoine"][: r.randint(1, 4)]:
            parts.append(f"== {sec} ==")
            for _ in range(r.randint(1, 4)):
                parts.append(f"La commune est traversée par {r.choice(RIVIERES)}. Au {r.randint(10, 19)}e siècle, "
                             f"elle dépend de l'abbaye voisine. Le village compte une église du {r.randint(11, 18)}e "
                             f"siècle et {r.randint(2, 9)} moulins. L'activité principale est l'agriculture.")
        return v, "\n\n".join(parts)
    name = _person(r)
    parts = [f"{name}, né le {r.randint(1, 28)} mars {r.randint(1800, 1950)} à {r.choice(VILLES)}, est un "
             f"{r.choice(METIERS)} français. Il est connu pour ses travaux sur {r.choice(THEMES_AN)}."]
    for sec in ["Biographie", "Œuvre", "Postérité"][: r.randint(1, 3)]:
        parts.append(f"== {sec} ==")
        for _ in range(r.randint(1, 3)):
            parts.append(f"En {r.randint(1820, 1990)}, il s'installe à {r.choice(VILLES)} où il rencontre "
                         f"{_person(r)}. Ses œuvres sont exposées au musée de {r.choice(VILLES)}.")
    return name, "\n\n".join(parts)


def _press(r) -> str:
    v = r.choice(VILLES)
    parts = [f"{v} — {_cap(r.choice(SUJETS_PRESSE))} a annoncé {r.choice(JOURS)} {r.choice(OBJETS_PRESSE)}, "
             f"a indiqué {_person(r)} dans un communiqué."]
    for _ in range(r.randint(3, 9)):
        parts.append(f"« Nous avons reçu {r.randint(10, 500)} demandes depuis le début de l'année », précise "
                     f"{_person(r)}. Selon les chiffres de {r.choice(SUJETS_PRESSE)}, la hausse atteint "
                     f"{r.randint(2, 40)} % sur un an. Les travaux doivent commencer en {r.choice(['janvier', 'mai', 'septembre'])}.")
    return "\n\n".join(parts)


def _seance_part(r, n_turns: int) -> str:
    deputes = [_person(r) for _ in range(5)]
    turns = []
    for _ in range(n_turns):
        d = r.choice(deputes)
        turns.append(f"M. le président. La parole est à M. {d}.")
        turns.append(f"M. {d}. Monsieur le président, monsieur le ministre, mes chers collègues, "
                     f"{r.choice(THEMES_AN)} mérite mieux que ce texte. {_narration(r)}")
        if r.random() < 0.5:
            turns.append(f"M. {r.choice(deputes)}. Très bien !")
    return "\n\n".join(turns)


def _theatre_act(r, act: int, chars: list[str]) -> str:
    out = [f"ACTE {act}", ""]
    for sc in range(1, r.randint(3, 5)):
        out.append(f"SCÈNE {sc}")
        for _ in range(r.randint(6, 12)):
            who = r.choice(chars)
            out.append(f"{who}. — {_narration(r)}")
    return "\n\n".join(out)


def corpus(seed: int = 7) -> dict[str, list[dict]]:
    """Documents par source, au format de docs/ARCHITECTURE.md."""
    r = random.Random(seed)
    docs: dict[str, list[dict]] = {}

    wiki = []
    for i in range(70):
        title, text = _encyclo(r, i)
        if i % 9 == 0:
            text = text[:300]  # ébauche trop courte (< 500 caractères)
        wiki.append({"id": f"wikipedia_fr:{1000 + i}", "source": "wikipedia_fr", "domain": "encyclopedique",
                     "text": text, "url": f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                     "date": "2026-10-01", "license": "CC BY-SA 4.0", "meta": {"title": title}})
    docs["wikipedia_fr"] = wiki

    authors = [_person(r) for _ in range(10)]
    guten = []
    for i in range(10):
        title = f"Le Roman numéro {i}" if i not in (2, 5) else f"Les Mémoires d'un témoin {i}"
        text = "\n\n".join(_narration(r) for _ in range(r.randint(40, 70)))
        guten.append({"id": f"gutenberg_fr:{5000 + i}", "source": "gutenberg_fr", "domain": "litteraire",
                      "text": text, "url": f"https://www.gutenberg.org/ebooks/{5000 + i}", "date": None,
                      "license": "domaine public", "meta": {"title": title, "author": authors[i]}})
    docs["gutenberg_fr"] = guten

    ws = []
    for w in range(8):
        # les œuvres 0 et 1 existent aussi dans Gutenberg (même auteur, même titre)
        title, author = (f"Les Mémoires d'un témoin {2 if w == 0 else 5}", authors[2 if w == 0 else 5]) \
            if w < 2 else (f"Contes de la vallée {w}", _person(r))
        for ch in range(1, r.randint(3, 6)):
            text = "\n\n".join(_narration(r) for _ in range(r.randint(8, 16)))
            ws.append({"id": f"wikisource_fr:{title}/Chapitre {ch}", "source": "wikisource_fr",
                       "domain": "litteraire", "text": text, "url": None, "date": None,
                       "license": "domaine public",
                       "meta": {"title": title, "author": author, "is_theatre": False}})
    for w in range(3):
        title = f"La Comédie des erreurs {w}"
        chars = [n.upper() for n in r.sample(PRENOMS, 4)]
        for act in range(1, 4):
            ws.append({"id": f"wikisource_fr:{title}/Acte {act}", "source": "wikisource_fr", "domain": "litteraire",
                       "text": _theatre_act(r, act, chars), "url": None, "date": None, "license": "domaine public",
                       "meta": {"title": title, "author": _person(r), "is_theatre": True}})
    docs["wikisource_fr"] = ws

    cc = []
    for i in range(90):
        news = i % 3 != 0
        host = r.choice(NEWS_HOSTS) if news else r.choice(OTHER_HOSTS)
        text = _press(r) if news else ("Recette du jour : mélangez la farine et le beurre. " * r.randint(10, 30))
        cc.append({"id": f"commoncrawl:{i:05d}", "source": "commoncrawl", "domain": "web", "text": text,
                   "url": f"https://{host}/article/{i}-{r.randint(1000, 9999)}.html", "date": "2025-06-01",
                   "license": "Common Crawl ToU", "meta": {}})
    docs["commoncrawl"] = cc

    an = []
    for s in range(8):
        sid = f"2024-{s + 1:02d}-1{s}"
        for part in range(1, 5):
            an.append({"id": f"assemblee_nationale:{sid}-{part}", "source": "assemblee_nationale",
                       "domain": "dialogue", "text": _seance_part(r, r.randint(4, 8)), "url": None,
                       "date": f"2024-{s + 1:02d}-10", "license": "Licence Ouverte 2.0",
                       "meta": {"seance_id": sid}})
    docs["assemblee_nationale"] = an
    return docs


def make_corpus(data_root, layout: str = "raw", seed: int = 7) -> dict[str, list[dict]]:
    """Écrit le corpus sous `data_root` et renvoie les documents par source."""
    data_root = Path(data_root)
    docs = corpus(seed)
    if layout == "raw":
        base = data_root / "raw"
    elif layout == "stage":
        base = data_root / "stages" / "nettoye" / "02_fin"
        # étape intermédiaire terminée et étape finale : le banc doit lire la dernière
        (data_root / "stages" / "nettoye" / "01_debut").mkdir(parents=True, exist_ok=True)
        (data_root / "stages" / "nettoye" / "01_debut" / "stats.json").write_text("{}")
    else:
        raise ValueError(layout)
    for src, items in docs.items():
        write_docs(base / src / "part-00000.jsonl.zst", items)
    if layout == "stage":
        (base / "stats.json").write_text(json.dumps({s: {"docs_out": len(d)} for s, d in docs.items()}))
    return docs


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_fixture")
    d = make_corpus(out, sys.argv[2] if len(sys.argv) > 2 else "raw")
    for s, items in d.items():
        print(s, len(items), sum(len(x["text"].encode()) for x in items))


# --------------------------------------------------------------------------- configs de test

def run_config(root, run: str = "t", **over) -> dict:
    """Config de run du banc pointée sur `root` (data/, runs/, experiments/), sans fichier YAML."""
    from bench.common import deep_merge
    from bench.config import load_run_config
    root = Path(root)
    base = {
        "run": run, "seed": 3,
        "paths": {"data": str(root / "data"), "runs": str(root / "runs"), "experiments": str(root / "experiments")},
        "mix": {"sources": [{"name": "nettoye", "pipeline": "nettoye"}], "workers": 1,
                "tokenizer_sample_bytes": 200_000},
    }
    over_sources = (over.get("mix") or {}).get("sources")
    cfg = deep_merge(base, over)
    if over_sources is not None:
        cfg["mix"]["sources"] = over_sources
    return load_run_config(None, cfg)
