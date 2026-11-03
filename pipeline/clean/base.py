"""Socle commun des étapes du pipeline : interface d'une étape et résultat par document."""
from dataclasses import dataclass


@dataclass
class Outcome:
    """Ce qu'une étape renvoie pour un document."""

    doc: dict  # document après l'étape (même s'il est retiré : il sert à l'échantillon)
    keep: bool = True
    reason: str | None = None  # raison du retrait (clé de removed_by_reason)
    counters: dict | None = None  # compteurs additifs propres à l'étape
    samples: list | None = None  # [(groupe, strate, enregistrement)] candidats d'échantillon
    capture: dict | None = None  # info renvoyée au processus principal (étapes globales)


class Stage:
    """Étape document par document (map/filter), exécutée en parallèle.

    Une étape « globale » (kind = "global") a besoin de voir tout le corpus
    avant de décider : voir pipeline/dedup/exact.py.
    """

    name = "?"
    kind = "map"
    uses_html = False  # l'étape lit doc["html"] : le runner garde ce champ jusqu'à elle
    filters = False  # l'étape peut retirer des documents
    sample_sizes: dict = {}  # groupe d'échantillon -> taille max par (source, strate)

    def __init__(self, params, ctx):
        self.params = dict(params or {})
        self.ctx = ctx
        self.later_stages = []

    def bind(self, later_stages):
        """Appelé par le runner une fois toutes les étapes construites."""
        self.later_stages = list(later_stages)

    def setup(self):
        """Charge les ressources lourdes (modèle, listes). Appelé une fois, avant le fork."""

    def teardown(self):
        """Libère les ressources lourdes après l'étape."""

    def apply(self, doc):
        """Traite un document et renvoie un Outcome."""
        raise NotImplementedError

    def replaces_text_from_html(self, source):
        """True si l'étape remplace `text` par une extraction du HTML pour cette source."""
        return False

    def fingerprint_extra(self, source=None):
        """Informations qui invalident le cache d'une source (liste d'IDs, modèle…). Rapide."""
        return {}

    def summarize(self, counters):
        """Met en forme les compteurs bruts d'une source pour stats.json."""
        return dict(sorted(counters.items()))

    def describe(self, source_stats):
        """Une phrase courte pour la colonne « détail » de funnel.md."""
        return ""

    def finalize(self, captures, samples):
        """Fin d'étape dans le processus principal : renvoie (rapport, rapport_markdown)."""
        return None, None


def utf8_len(text):
    """Taille en octets UTF-8 d'un texte."""
    return len(text.encode("utf-8")) if text else 0


def common_prefix_len(a, b):
    """Longueur du plus long préfixe commun (recherche dichotomique, comparaisons en C)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def excerpt_diff(before, after, width=160):
    """Extraits avant/après autour de la première différence (pour les échantillons)."""
    i = common_prefix_len(before, after)
    start = max(0, i - width // 4)
    return before[start:start + width], after[start:start + width]
