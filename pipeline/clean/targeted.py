"""Nettoyage structurel prudent, sans changer le filtre de langue.

Travaille sur le corpus déjà nettoyé : aucune liste d'IDs, aucun choix guidé
par la validation. La poésie, les noms étrangers et les petits textes restent.
"""
from collections import Counter
import re

from pipeline.clean.base import Outcome, Stage, excerpt_diff, utf8_len

WIKI = {"wikipedia_fr", "wikinews_fr", "wikisource_fr"}
TABLE_SOURCES = {"wikipedia_fr", "wikinews_fr"}
LITERARY = {"wikisource_fr", "gutenberg_fr"}
TABLE_TOKEN = re.compile(r"\{\||\|\}")
ATTR = re.compile(r"\b(?:rowspan|colspan|bgcolor|style|align|width)\s*=", re.I)
MATH = re.compile(r"\\[A-Za-z]+")
TABLE_OPEN = re.compile(r'^\{\|(?:[ \t]*(?:class|style|rowspan|colspan|align|width)\s*=|[ \t]*\|[-+!\s]|[ \t]*\n[ \t]*\|)', re.I)
REPEAT_WORD = re.compile(r"\b([^\W\d_]{2,})\b(?:[ \t]+\1\b){9,}", re.I)
IMAGE_LINE = re.compile(r"\s*(?:(?:centré|center|droite|gauche|thumb|vignette)\|)*\d+px\s*", re.I)
CONTENTS_ENTRY = re.compile(r"(?:lettre|chapitre|livre|chant|tome)\s+(?:[IVXLCDM]+|\d+)[.\s–—-]*", re.I)
EMPTY_HEADINGS = {"statistiques", "effectif", "victoires", "classement uci", "filmographie",
                  "biographie", "télévision", "clips", "sources", "liens externes"}


def remove_balanced_tables(text, counts):
    """Retire seulement les tableaux dont l'ouverture et la fermeture sont présentes."""
    depth, start, spans = 0, 0, []
    for token in TABLE_TOKEN.finditer(text):
        if token.group() == "{|":
            if not depth:
                start = token.start()
            depth += 1
        elif depth:
            depth -= 1
            if not depth:
                block = text[start:token.end()]
                prose_inside = any(len(line) >= 80 and not table_line(line) and
                                   re.search(r'[.!?]', line) for line in block.splitlines())
                line_start = text.rfind('\n', 0, start) + 1
                line_end = text.find('\n', token.end())
                context = text[line_start:line_end if line_end >= 0 else len(text)]
                boundary = start == 0 or text[start-1].isspace()
                if (boundary and len(block) <= 20000 and not prose_inside and TABLE_OPEN.match(block)
                        and not MATH.search(block) and not MATH.search(context)):
                    spans.append((start, token.end()))
    if not spans:
        return text
    chunks, start = [], 0
    for a, b in spans:
        chunks.extend([text[start:a], "\n"])
        start = b
    counts['balanced_tables'] += len(spans)
    return ''.join(chunks) + text[start:]


def table_line(line):
    """Un opérateur || cité dans une phrase n'est pas un tableau."""
    if MATH.search(line):
        return False
    if ('{|' in line and line.strip() != '{|' and
            not any(TABLE_OPEN.match(line[m.start():]) for m in re.finditer(r'\{\|', line))):
        return False
    return ('|' in line and (line.lstrip().startswith(('|}', '{|')) or
            re.search(r'\|[-+](?=\s|\||$)', line) or '|}' in line or ATTR.search(line)))


def clean_structure(text, source):
    counts = Counter()
    if source not in WIKI:
        return text, counts
    original = text
    # Wikisource emploie aussi les tableaux pour la prose et les traductions.
    # Le premier essai a révélé des fermetures éloignées dans ces livres :
    # la suppression de tableaux y reste désactivée.
    if source in TABLE_SOURCES and '{|' in text:
        text = remove_balanced_tables(text, counts)
    if '{{' in text and re.search(r'\{\{\s*infobox', text, re.I):
        import mwparserfromhell
        code = mwparserfromhell.parse(text)
        for template in list(code.filter_templates(recursive=False)):
            if str(template.name).strip().casefold().startswith('infobox'):
                code.remove(template)
                counts['infoboxes'] += 1
        text = str(code)
    lines = []
    previous, repeat = None, 0
    raw_lines = text.splitlines(keepends=True)
    # Sans délimiteurs fiables, une cellule peut continuer sur la ligne suivante.
    # La retirer seule laisserait « Réélu en 1900 » sans le nom correspondant.
    ambiguous_rows = any(table_line(a) and '|}' not in a and b.strip() and
                         not table_line(b) and any(c.isalpha() for c in b)
                         for a, b in zip(raw_lines, raw_lines[1:]))
    for line in raw_lines:
        if (source in TABLE_SOURCES and line.lstrip().startswith('|') and line.count('|') == 1
                and len(line.split()) >= 5 and re.search(r'[.!?]\s*$', line)):
            line = line.lstrip()[1:].lstrip()
            counts['stray_leading_pipes'] += 1
        if source in TABLE_SOURCES and not ambiguous_rows and table_line(line):
            # Une ligne de tableau peut finir par |} suivi d'un vrai paragraphe.
            parts = line.split('|}')
            kept = []
            for part in parts:
                if table_line(part) or (len(parts) > 1 and part.lstrip().startswith('|') and '||' in part):
                    prefix = part.split('|', 1)[0]
                    if len(prefix.split()) >= 5 and re.search(r'[.!?]\s*$', prefix):
                        kept.append(prefix)
                elif part.strip() and not (len(parts) == 1):
                    kept.append(part)
            lines.append('\n'.join(kept) + '\n' if kept else '\n')
            counts['table_lines'] += 1
            continue
        stripped = line.strip()
        if re.fullmatch(r'Q\d{4,}', stripped) or IMAGE_LINE.fullmatch(stripped):
            counts['layout_lines'] += 1
            continue
        if source in TABLE_SOURCES and REPEAT_WORD.search(line):
            line, n = REPEAT_WORD.subn(lambda m: m[1], line)
            counts['repeated_word_runs'] += n
        # Les suites de lignes très courtes identiques sont comptées ; garder
        # les neuf premières évite de retirer les répétitions courtes légitimes.
        short = line.strip()
        repeat = repeat + 1 if short and short == previous else 1
        previous = short
        if source in TABLE_SOURCES and repeat >= 10 and len(short.split()) <= 3 and short.isalpha():
            counts['repeated_lines'] += 1
            continue
        lines.append(line)
    if not counts:
        return original, counts
    text = ''.join(lines)
    if source == 'wikipedia_fr':
        lines = text.rstrip().split('\n')
        while lines and (not lines[-1].strip() or lines[-1].strip().casefold() in EMPTY_HEADINGS):
            lines.pop()
            counts['empty_trailing_lines'] += 1
        text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, counts


def contents_only(text, title=''):
    """Seulement un titre et une suite de numéros de chapitres ; pas de vers ni de prose."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    entries = [x for x in lines if CONTENTS_ENTRY.fullmatch(x)]
    allowed = {title.strip().casefold(), 'sommaire', 'table des matières'}
    return len(entries) >= 5 and all(CONTENTS_ENTRY.fullmatch(x) or x.casefold() in allowed for x in lines)


class TargetedCleanStage(Stage):
    name = 'targeted_clean'
    filters = True
    sample_sizes = {'changed': 200}

    def apply(self, doc):
        before = doc.get('text') or ''
        after, counts = clean_structure(before, doc['source'])
        reason = None
        if not any(c.isalpha() for c in after):
            reason = 'no_text_after_structure'
        elif doc['source'] in LITERARY and contents_only(after, doc.get('meta', {}).get('title', '')):
            reason = 'contents_only'
        if reason:
            return Outcome(doc, keep=False, reason=reason, counters=dict(counts))
        if after == before:
            return Outcome(doc)
        # Les scores décrivent l'ancien texte : conserver leur provenance,
        # pas les laisser passer pour des scores recalculés sur le texte nettoyé.
        result = {**doc, 'text': after, 'meta': dict(doc.get('meta', {}))}
        scores = {k: result['meta'].pop(k) for k in list(result['meta']) if k.startswith('langid_')}
        result['meta']['targeted_clean'] = {'rules': dict(counts), 'previous_langid': scores}
        counts['bytes_removed'] = utf8_len(before) - utf8_len(after)
        def sample():
            a, b = excerpt_diff(before, after, width=1200)
            return {'id': doc['id'], 'source': doc['source'], 'before': a, 'after': b,
                    'bytes_before': utf8_len(before), 'bytes_after': utf8_len(after), 'rules': dict(counts)}
        return Outcome(result, counters=dict(counts), samples=[('changed', '', sample)])

    def describe(self, st):
        c = st.get('counters', {})
        return f"{st.get('docs_modified', 0)} modifiés ; {c.get('bytes_removed', 0)} octets retirés"
