"""Heuristiques prudentes sur le web, inspirées de Gopher, adaptées au français.

Pas de filtre de mots interdits. Pas de seuil de longueur sur livres, poésie,
brèves Wikinews, débats ou encyclopédie. Les seuils sont des choix à évaluer.
"""
import re
from collections import Counter

from pipeline.clean.base import Outcome, Stage


WORDS = re.compile(r'\w+', re.UNICODE)
SHORT_SHELL = re.compile(
    r"(?:nous utilisons des cookies|votre formulaire a expiré|aucun produit disponible|"
    r"retour à la page d'accueil|copiez.collez (?:cette URL|ce code)|"
    r"rempliss(?:ez|ant) le formulaire|recevez par email les bons plans)", re.I)


def quality_signals(text):
    words = WORDS.findall(text.lower())
    visible = sum(not c.isspace() for c in text)
    lines = [' '.join(line.lower().split()) for line in text.splitlines() if line.strip()]
    line_counts = Counter(lines)
    line_chars = sum(len(s) * n for s, n in line_counts.items())
    duplicate_chars = sum(len(s) * (n - 1) for s, n in line_counts.items())
    grams = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    # Nombre de mots couverts par le trigramme le plus fréquent ; plafonné à 1.
    top = max(grams.values(), default=0)
    return {'words': len(words),
            'alpha_fraction': sum(c.isalpha() for c in text) / max(1, visible),
            'mean_word_length': sum(map(len, words)) / max(1, len(words)),
            'duplicate_line_char_fraction': duplicate_chars / max(1, line_chars),
            'top_trigram_word_fraction': min(1., 3 * top / max(1, len(words)))}


class QualityStage(Stage):
    name = 'quality'
    filters = True

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.sources = set(self.params.get('sources', ['commoncrawl', 'fineweb2_fra']))
        self.min_words = int(self.params.get('min_words', 10))
        self.min_alpha = float(self.params.get('min_alpha_fraction', .55))
        self.max_duplicate = float(self.params.get('max_duplicate_line_char_fraction', .5))
        self.max_trigram = float(self.params.get('max_top_trigram_word_fraction', .3))

    def apply(self, doc):
        if doc['source'] not in self.sources:
            return Outcome(doc, counters={'outside_web_scope': 1})
        text = doc.get('text') or ''
        signals = quality_signals(text)
        reason = None
        if signals['words'] < self.min_words:
            reason = 'quality_short_web'
        elif signals['words'] <= 60 and SHORT_SHELL.search(text):
            reason = 'quality_interface_only'
        elif signals['alpha_fraction'] < self.min_alpha:
            reason = 'quality_low_letters'
        elif not 2 <= signals['mean_word_length'] <= 20:
            reason = 'quality_word_length'
        elif signals['duplicate_line_char_fraction'] > self.max_duplicate:
            reason = 'quality_repeated_lines'
        elif signals['top_trigram_word_fraction'] > self.max_trigram:
            reason = 'quality_repeated_trigram'
        else:
            masks = (doc.get('meta', {}).get('privacy') or {}).get('masks', {})
            contacts = masks.get('EMAIL', 0) + masks.get('TELEPHONE', 0)
            if contacts >= 5 and contacts / max(1, signals['words']) > .02:
                reason = 'quality_contact_directory'
        meta = {**(doc.get('meta') or {}), 'quality': signals}
        return Outcome({**doc, 'meta': meta}, keep=reason is None, reason=reason,
                       counters={'web_evaluated': 1})

    def describe(self, st):
        return f"{st['docs_in'] - st['docs_out']} pages web retirées"
