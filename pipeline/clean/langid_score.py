"""Étape `langid_score` : la note de langue de chaque document, calculée une fois et gardée en cache.

C'est le calcul cher du filtre de langue (~1 Mo/s par cœur). L'étape ne retire rien : elle
écrit dans `meta` ce que dit le modèle, et l'étape suivante (`langid`) applique le seuil.
Changer de seuil ou de règle ne relance donc pas le modèle (le cache de cette étape reste valide).

Modèle : GlotLID v3 (fastText, 2 102 étiquettes, cis-lmu/glotlid), sinon fastText lid.176.
Prédiction sur le texte mis sur une seule ligne, distribution complète des probabilités.

Ce qui est écrit dans `meta` :
- `langid_label`, `langid_score` : 1re étiquette et sa probabilité ;
- `langid_fr` : probabilité du français (`fra_Latn`) ;
- `langid_family` : probabilité « français + parlers proches » (normand, franco-provençal,
  picard, wallon) : GlotLID partage souvent la probabilité d'un texte français entre eux ;
- `langid_group` : groupe de la 1re étiquette (`fr`, `family` = parler proche, `other`) ;
- `langid_top` : les 5 premières étiquettes (probabilité >= 0,01), pour le diagnostic.

Téléchargement manuel : uv run python -m pipeline.clean.langid_score --download glotlid
"""
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pipeline.clean.base import Outcome, Stage

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "glotlid": {
        "path": "data/models/glotlid/model.bin",
        "repo": "cis-lmu/glotlid",
        "filename": "model.bin",
        "fr_labels": ("fra_Latn",),
        # Parlers d'oïl et franco-provençal présents dans GlotLID (pas les créoles ni l'occitan,
        # langues à part entière). L'annotation dira si cette somme sauve du bon français.
        "family_labels": ("fra_Latn", "frp_Latn", "nrm_Latn", "pcd_Latn", "wln_Latn"),
        "license": "Apache-2.0 (+ notices), cis-lmu/glotlid",
    },
    "lid176": {
        "path": "data/models/lid176/lid.176.bin",
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "fr_labels": ("fr",),
        "family_labels": ("fr", "wa"),
        "license": "CC-BY-SA 3.0, fastText (Meta)",
    },
}
TOP_K, TOP_MIN = 5, 0.01  # étiquettes gardées dans `langid_top`
DIGITS = 6  # décimales des scores écrits dans meta (le seuil est appliqué sur ces valeurs)


def model_path(name, root=ROOT):
    return Path(root) / MODELS[name]["path"]


def download(name, root=ROOT):
    """Télécharge le modèle dans data/models/ (fait dans un sous-processus par le runner)."""
    spec = MODELS[name]
    dest = model_path(name, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if name == "glotlid":
        from huggingface_hub import hf_hub_download

        for fname in ("README.md", "LICENSE", spec["filename"]):
            hf_hub_download(repo_id=spec["repo"], filename=fname, local_dir=str(dest.parent))
    else:
        import requests

        tmp = dest.with_name(dest.name + ".part")
        with requests.get(spec["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
    return dest


def resolve_model(params, root=ROOT):
    """Choisit le modèle : celui demandé, téléchargé si besoin, sinon le repli."""
    wanted = params.get("model", "glotlid")
    order = [wanted] + [m for m in (params.get("fallback", "lid176"),) if m and m != wanted]
    for name in order:
        path = model_path(name, root)
        if not path.exists() and params.get("download", True):
            # Sous-processus : évite d'initialiser le réseau dans le processus qui va forker.
            subprocess.run([sys.executable, "-m", "pipeline.clean.langid_score", "--download", name],
                           cwd=str(root), check=False)
        if path.exists():
            return name, path
    raise FileNotFoundError(
        f"Aucun modèle de langue disponible ({', '.join(order)}). "
        "Lancer : uv run python -m pipeline.clean.langid_score --download glotlid"
    )


def summarize_probs(probs, fr_labels, family_labels):
    """Résumé d'une distribution {étiquette: probabilité} : ce que l'étape écrit dans meta."""
    label, score = max(probs.items(), key=lambda kv: (kv[1], kv[0]))
    group = "fr" if label in fr_labels else ("family" if label in family_labels else "other")
    top = sorted(((p, lab) for lab, p in probs.items() if p >= TOP_MIN), reverse=True)[:TOP_K]
    return {
        "langid_label": label,
        "langid_score": round(score, DIGITS),
        "langid_fr": round(min(1.0, sum(probs.get(lab, 0.0) for lab in fr_labels)), DIGITS),
        "langid_family": round(min(1.0, sum(probs.get(lab, 0.0) for lab in family_labels)), DIGITS),
        "langid_group": group,
        "langid_top": {lab: round(p, 4) for p, lab in top},
    }


class LangIdScoreStage(Stage):
    name = "langid_score"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.max_chars = int(self.params.get("max_chars", 0))  # 0 = tout le texte
        self.model_name, self.path = resolve_model(self.params, ctx.root)
        spec = MODELS[self.model_name]
        self.fr_labels = tuple(sorted(self.params.get("fr_labels") or spec["fr_labels"]))
        self.family_labels = tuple(sorted(set(self.params.get("family_labels") or spec["family_labels"])
                                          | set(self.fr_labels)))
        self.model = None

    def setup(self):
        import fasttext

        fasttext.FastText.eprint = lambda *a, **k: None  # message d'avertissement inutile
        self.model = fasttext.load_model(str(self.path))

    def teardown(self):
        self.model = None

    def fingerprint_extra(self, source=None):
        st = self.path.stat()  # taille et date : un modèle re-téléchargé invalide le cache
        return {"model": self.model_name, "model_size": st.st_size, "model_mtime": st.st_mtime_ns,
                "fr_labels": self.fr_labels, "family_labels": self.family_labels}

    def probs(self, text):
        """{étiquette: probabilité} sur tout le texte mis sur une ligne (None si vide)."""
        line = " ".join(text.split())
        if self.max_chars and len(line) > self.max_chars:
            line = line[: self.max_chars]
        if not line:
            return None
        labels, probs = self.model.predict(line, k=-1, threshold=0.0)
        return {lab.replace("__label__", ""): min(1.0, max(0.0, float(p))) for lab, p in zip(labels, probs)}

    def score(self, text):
        """Ce que l'étape écrit dans meta (sans `langid_model`), ou None si le texte est vide."""
        probs = self.probs(text)
        return summarize_probs(probs, self.fr_labels, self.family_labels) if probs else None

    def apply(self, doc):
        meta = doc.setdefault("meta", {})
        res = self.score(doc.get("text") or "")
        meta["langid_model"] = self.model_name
        if res is None:
            meta.update(langid_label=None, langid_score=0.0, langid_fr=0.0, langid_family=0.0,
                        langid_group="other", langid_top={})
            return Outcome(doc, counters={"empty": 1})
        meta.update(res)
        return Outcome(doc, counters={f"group:{res['langid_group']}": 1})

    def summarize(self, counters):
        return {"model": self.model_name, "fr_labels": list(self.fr_labels),
                "family_labels": list(self.family_labels), **dict(sorted(counters.items()))}

    def describe(self, st):
        c = Counter(st.get("counters", {}))
        n = max(1, st["docs_in"])
        return (f"{st['docs_in']} docs notés ({self.model_name}) : 1re étiquette française "
                f"{100 * c['group:fr'] / n:.1f} %, parler proche {100 * c['group:family'] / n:.1f} %, "
                f"autre {100 * c['group:other'] / n:.1f} %, vides {c['empty']}").replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description="Gestion des modèles d'identification de langue")
    ap.add_argument("--download", choices=sorted(MODELS), required=True)
    args = ap.parse_args()
    print(download(args.download))


if __name__ == "__main__":
    main()
