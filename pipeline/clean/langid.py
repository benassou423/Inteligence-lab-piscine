"""Étape `langid` : on ne garde que le français, d'après les notes de l'étape `langid_score`.

Étape instantanée (aucun modèle) : elle lit les probabilités écrites dans `meta` par
`langid_score`, qui la précède toujours (le runner le vérifie). Changer le seuil ou la
règle ne relance que cette étape.

Deux règles, au choix (`rule`) :
- `fr` (défaut) : probabilité du français >= seuil. Dès que le seuil est >= 0,5, c'est
  exactement l'ancienne règle « 1re étiquette fra_Latn et score >= seuil ».
- `family` : probabilité « français + parlers proches » (normand, franco-provençal,
  picard, wallon) >= seuil. GlotLID partage souvent un texte français entre ces
  étiquettes (comptes rendus de foot de Wikinews : fra 0,58 + nrm 0,35).

Le seuil et la règle se choisissent sur l'annotation de Ben (tools.annotate_langid puis
tools.langid_report, qui compare les deux règles sur les mêmes annotations).

Téléchargement du modèle : uv run python -m pipeline.clean.langid_score --download glotlid
"""
from collections import Counter

from pipeline.clean.base import Outcome, Stage, utf8_len

DEFAULT_THRESHOLD = 0.65
RULES = {"fr": "langid_fr", "family": "langid_family"}  # règle -> score de meta comparé au seuil
# Strates pour l'annotation manuelle (tirage stratifié de tools.annotate_langid) :
KEPT_EDGES = (0.80, 0.90, 0.95, 0.99)  # gardés : tranches du score de la règle, au-dessus du seuil
FAMILY_EDGES = (0.40, 0.65)  # rejetés : selon le score « famille » (les cas que la règle family sauverait)
REJECTED_HIGH = 0.90  # rejetés hors famille : autre langue sûre (>= 0,90) ou incertaine


def kept_strata(threshold, edges=KEPT_EDGES):
    """Bornes des strates de documents gardés : [(nom, bas, haut)]."""
    bounds = [threshold] + [e for e in edges if e > threshold] + [1.0]
    return [(f"kept_{lo:.2f}-{hi:.2f}", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def stratum_of(keep, score, family, top_score, threshold, edges=KEPT_EDGES):
    """Strate d'un document : gardé (tranche du score de la règle) ou rejeté (selon la famille)."""
    if keep:
        for name, lo, hi in kept_strata(threshold, edges):
            if lo <= score < hi:
                return name
        return kept_strata(threshold, edges)[-1][0]
    if family >= FAMILY_EDGES[1]:
        return "rejected_fam_high"  # français + parlers proches >= 0,65 : la règle family le garderait
    if family >= FAMILY_EDGES[0]:
        return "rejected_fam_mid"
    return "rejected_other_high" if top_score >= REJECTED_HIGH else "rejected_other_low"


class LangIdStage(Stage):
    name = "langid"
    filters = True
    sample_sizes = {"strata": 200}

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.threshold = float(self.params.get("threshold", DEFAULT_THRESHOLD))
        self.rule = self.params.get("rule", "fr")
        if self.rule not in RULES:
            raise ValueError(f"langid : règle inconnue {self.rule!r} (connues : {', '.join(RULES)})")
        self.sample_chars = int(self.params.get("sample_chars", 1500))

    def apply(self, doc):
        meta = doc.get("meta") or {}
        if RULES[self.rule] not in meta:
            raise RuntimeError(f"langid : {doc['id']} n'a pas de note de langue ; "
                               "l'étape langid_score doit précéder langid dans la config")
        label = meta.get("langid_label")
        if label is None:
            return Outcome(doc, keep=False, reason="empty")
        score, family, top = meta[RULES[self.rule]], meta["langid_family"], meta["langid_score"]
        group = meta.get("langid_group", "other")
        keep = score >= self.threshold
        frenchish = group == "fr" or (self.rule == "family" and group == "family")
        bin_ = min(99, int(score * 100))
        hist = "kept" if keep else ("rejected_fr" if frenchish else "rejected_other")
        stratum = stratum_of(keep, score, family, top, self.threshold)
        counters = {f"hist:{hist}:{bin_:02d}": 1, f"stratum:{stratum}": 1}
        if not keep and not frenchish:
            counters[f"label:{label}"] = 1
        text = doc.get("text") or ""
        record = {"id": doc["id"], "source": doc["source"], "domain": doc.get("domain"), "url": doc.get("url"),
                  "text": text[: self.sample_chars], "text_bytes": utf8_len(text), "keep": keep,
                  "stratum": stratum, **{k: v for k, v in meta.items() if k.startswith("langid_")}}
        samples = [("strata", stratum, record)]
        if keep:
            return Outcome(doc, counters=counters, samples=samples)
        return Outcome(doc, keep=False, reason="low_score" if frenchish else "not_french",
                       counters=counters, samples=samples)

    def summarize(self, counters):
        hist = {k: [0] * 100 for k in ("kept", "rejected_fr", "rejected_other")}
        strata, labels, other = Counter(), Counter(), {}
        for k, v in counters.items():
            if k.startswith("hist:"):
                _, kind, b = k.split(":")
                hist[kind][int(b)] += v
            elif k.startswith("stratum:"):
                strata[k[8:]] += v
            elif k.startswith("label:"):
                labels[k[6:]] += v
            else:
                other[k] = v
        return {
            "rule": self.rule,
            "threshold": self.threshold,
            "strata": dict(sorted(strata.items())),
            "rejected_labels": dict(labels.most_common(30)),
            "score_hist": hist,
            **other,
        }

    def describe(self, st):
        labels = Counter({k[6:]: v for k, v in st.get("counters", {}).items() if k.startswith("label:")})
        r = st["removed_by_reason"]
        parts = []
        if r.get("not_french"):
            top = ", ".join(f"{k} {v}" for k, v in labels.most_common(3))
            parts.append(f"autre langue {r['not_french']} ({top})")
        if r.get("low_score"):
            what = "français" if self.rule == "fr" else "français + parlers proches"
            parts.append(f"{what} < {self.threshold:.2f} : {r['low_score']}".replace(".", ","))
        if r.get("empty"):
            parts.append(f"vides {r['empty']}")
        return " ; ".join(parts) or "rien retiré"
