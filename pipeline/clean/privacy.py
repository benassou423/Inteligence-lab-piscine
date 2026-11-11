"""Masquage déterministe de coordonnées structurées, sans journaliser leurs valeurs.

Ce filtre ne constitue pas une anonymisation : noms, adresses postales et formes
obfusquées ne sont pas détectés. Les dates, populations et références restent utiles.
"""
import ipaddress
import re
from collections import Counter

from pipeline.clean.base import Outcome, Stage


EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?)*\.[A-Z]{2,24}(?![\w-]|\.[A-Z0-9])", re.I)
PHONE = re.compile(r"(?<![\w+.,])(?:0[1-9]|\+33[ .\-\u00a0\u202f]?[1-9])(?P<sep>[ .\-\u00a0\u202f]?)\d{2}(?P=sep)\d{2}(?P=sep)\d{2}(?P=sep)\d{2}(?![\w.,]\d|\w)")
PHONE_ID_PREFIX = re.compile(r"\b(?:antenne|référence|ref|identifiant|ISBN|ISSN|EAN|code)\s*(?:n[°o]|[:=#])?\s*$", re.I)
PHONE_HINT = re.compile(r"\b(?:t[ée]l(?:[ée]phone)?|portable|fax|mobile|sms|joignable|appeler|appelez|appellez|contact(?:ez)?)\b", re.I)
CURATED = {'wikipedia_fr', 'wikinews_fr', 'wikisource_fr', 'gutenberg_fr', 'assemblee_nationale'}
IBAN_FR = re.compile(r"(?<!\w)FR\d{2}(?:[ \u00a0\u202f]?[A-Z0-9]){23}(?!\w)", re.I)
IP_CONTEXT = re.compile(r"(?P<prefix>\b(?:adresse\s+IP|IP(?:v4|v6)?)\s*[:=]?\s*)(?P<address>[0-9a-f:.]+)", re.I)


def valid_iban(value):
    compact = ''.join(value.split()).upper()
    if len(compact) != 27 or not compact.startswith('FR'):
        return False
    digits = ''.join(str(ord(c) - 55) if c.isalpha() else c for c in compact[4:] + compact[:4])
    return int(digits) % 97 == 1


def mask_text(text, source=None):
    counts = Counter()

    def replace(kind, check=None):
        def sub(match):
            if check is not None and not check(match.group()):
                return match.group()
            counts[kind] += 1
            return f'[{kind}]'
        return sub

    text = EMAIL.sub(replace('EMAIL'), text)
    text = IBAN_FR.sub(replace('IBAN', valid_iban), text)
    def phone_sub(match):
        # Un vrai contre-exemple du pilote : « antenne n° 0422700190 ».
        # Même forme qu'un téléphone, mais libellé d'identifiant explicite.
        before = text[max(0, match.start() - 160):match.start()]
        after = text[match.end():match.end() + 80]
        if PHONE_ID_PREFIX.search(before):
            return match.group()
        # Ne pas extraire cinq paires au milieu d'une longue ligne de chiffres,
        # ni une fin de décimale. Les séparateurs doivent tous être identiques.
        hinted = bool(PHONE_HINT.search(before + after))
        directory_line = (source in {'commoncrawl', 'fineweb2_fra'} and not match['sep']
                          and not before.rsplit('\n', 1)[-1].strip()
                          and bool(re.search(r'\bannuaire\b', text, re.I))
                          and bool(re.search(r'[A-Za-zÀ-ÿ]{3,}', after.split('\n', 1)[0])))
        dense = re.search(r'\d[ .\-\u00a0\u202f]?$', before) or re.match(r'^[ .\-\u00a0\u202f]?\d', after)
        if dense and not (hinted or directory_line):
            return match.group()
        if source in CURATED and not hinted:
            return match.group()
        counts['TELEPHONE'] += 1
        return '[TELEPHONE]'

    text = PHONE.sub(phone_sub, text)

    def ip_sub(match):
        value = match['address'].rstrip('.')
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return match.group()
        counts['IP'] += 1
        return match['prefix'] + '[IP]' + match['address'][len(value):]

    return IP_CONTEXT.sub(ip_sub, text), counts


class PrivacyStage(Stage):
    name = 'privacy'

    def apply(self, doc):
        before = doc.get('text') or ''
        after, counts = mask_text(before, doc['source'])
        if not counts:
            return Outcome(doc)
        meta = dict(doc.get('meta') or {})
        scores = {k: meta.pop(k) for k in list(meta) if k.startswith('langid_')}
        meta['privacy'] = {'masks': dict(counts), 'previous_langid': scores}
        return Outcome({**doc, 'text': after, 'meta': meta},
                       counters={**dict(counts), 'documents_masked': 1})

    def describe(self, st):
        return f"{st.get('docs_modified', 0)} textes avec coordonnées masquées"
