"""Jeu de données « théâtre » du chatbot : des scènes découpées en répliques (qui parle, que dit-il).

Entrées (lecture seule) :
  - corpus/moliere_train.txt et corpus/moliere_val.txt (kit officiel, 25 + 3 pièces) ;
  - data/moliere/moliere.txt (33 pièces de Molière tirées de Wikisource, autre édition) ;
  - les sources wikisource_fr et gutenberg_fr du corpus nettoyé (on y cherche les autres pièces).

Sorties (une ligne JSON = une scène) dans data/stages/chatbot_20261112/theatre/ :
  moliere_train.jsonl.zst, moliere_val.jsonl.zst, theatre_train.jsonl.zst, stats.json.

    uv run python -m chatbot.theatre                    # exécution complète
    uv run python -m chatbot.theatre --limit-files 4    # essai rapide sur 4 fichiers du corpus

Règle d'or : les 3 pièces de validation (Le Misanthrope, Le Sicilien, L'Avare) ne doivent se
retrouver dans AUCUN fichier d'entraînement, quelle que soit l'édition. On le vérifie en comptant
les suites de mots communes avec ces pièces (13 mots comme ftb/contamination.py pour les documents,
12 mots scène par scène : un alexandrin repris fait 12 mots), et en écartant les scènes qui
reprennent leurs personnages (suites et pastiches).

Les documents du corpus listés dans chatbot/exclusions_theatre.json (audit à l'œil, liste unique) sont
écartés. moliere_val est figé (empreinte VAL_SHA256) : la référence BPB y a été mesurée.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import xxhash

from pipeline.io import read_docs, write_docs

ROOT = Path(__file__).resolve().parent.parent
KIT_TRAIN = ROOT / "corpus" / "moliere_train.txt"
KIT_VAL = ROOT / "corpus" / "moliere_val.txt"
KIT_MANIFEST = ROOT / "corpus" / "moliere_manifest.json"
WS_MOLIERE = ROOT / "data" / "moliere" / "moliere.txt"
WS_README = ROOT / "data" / "moliere" / "README.md"
CORPUS = ROOT / "data" / "stages" / "nettoye" / "08_exact_dedup"
SOURCES = ("wikisource_fr", "gutenberg_fr")
OUT = ROOT / "data" / "stages" / "chatbot_20261112" / "theatre"
EXCLUSIONS = ROOT / "chatbot" / "exclusions_theatre.json"

MAX_TURNS = 60        # une scène plus longue est coupée en morceaux de 60 répliques au plus
NGRAM = 13            # taille des suites de mots comparées (même valeur que le kit)
NGRAM_VAL = 12        # scène par scène contre la validation : un alexandrin repris = 12 mots
MIN_SHARED = 3        # une ÉDITION DE MOLIÈRE qui partage 3 13-grammes distincts ou plus est exclue entière
MIN_DOC_TURNS = 10    # en dessous, un document du corpus n'a « pas assez de répliques »
DUP_SHARE = 0.8       # scène dont 80 % des 13-grammes viennent d'un autre document : doublon d'édition
MOLIERE_SHARE = 0.2   # document dont 20 % des 13-grammes sont chez Molière : une édition de Molière
WS_LICENSE = "Domaine public (œuvres) ; transcription Wikisource CC BY-SA 4.0"
MAX_PARA = 300        # mode strict : un paragraphe isolé plus long (et pas en vers) est du récit, pas une réplique

# Empreinte SHA-256 de moliere_val.jsonl.zst, figée : le BPB de référence 1,4746 (modèle non affiné) a été
# mesuré sur CE fichier. La validation est découpée sans les nettoyages ajoutés ensuite (cleanup=False).
VAL_SHA256 = "6e7d26d14598666ea6c298104cd19c63fb566a13b9fb3a064ebc7ed06b2c5a6b"

# ---------------------------------------------------------------------------------------------
# 1. Reconnaître le rôle de chaque ligne : nom de personnage, titre, didascalie ou texte
# ---------------------------------------------------------------------------------------------

UP = "A-ZÀ-ÖØ-ÞŒÆŸ"                                   # lettres capitales, accents compris
TOKEN = rf"[{UP}][{UP}'’\-]*"                            # un mot en capitales : LÉLIE, D'ARTAGNAN
NAME = rf"{TOKEN}(?:\.?\s+{TOKEN}){{0,5}}"               # 1 à 6 mots : M. JOURDAIN, LE DOCTEUR
DIDA = r"(?P<dida>\s*,[^\n]{1,100}?|\s*\([^()\n]{1,100}\))?"   # « , à part » ou « (lisant) »
# Nom seul sur sa ligne : « HARPAGON. », « HARPAGON, à part. », « LÉLIE », « CYRANO (riant): »,
# « JODELET. (Ils s'embrassent.) »
ALONE = re.compile(rf"^(?P<name>{NAME}){DIDA}\s*(?P<end>[.:])?(?:\s*\([^()\n]{{1,150}}\)[.:]?)?$")
# Nom et réplique sur la même ligne : « ANTONIO.--Qui est là? », « POÈTE (riant): Exquis ! »
INLINE = re.compile(rf"^(?P<name>{NAME}){DIDA}\s*(?:\.\s*(?:--|—|–)|\s(?:—|–)|:)\s*(?P<rest>\S.*)$")

# Titres qui ferment la scène en cours (acte, scène, fin, liste des personnages, préface…)
HEADING_UP = re.compile(r"^(?:ACTE|SC[ÈE]NE|TABLEAU|PROLOGUE|[ÉE]PILOGUE|INTERM[ÈE]DE|ENTR['’]ACTE|"
                        r"FIN|RIDEAU|PERSONNAGES|ACTEURS|CHAPITRE|LIVRE|PARTIE|TOME|NOTES?|"
                        r"PR[ÉE]FACE|AVERTISSEMENT|NOTICE|TABLE|INTRODUCTION|D[ÉE]DICACE|[ÉE]P[ÎI]TRE"
                        r"|RECUEILS|VARIANTES?|[ÉE]DITIONS?\s+S[ÉE]PAR[ÉE]ES?)\b"   # notes de variantes (Corneille)
                        r"|^\S+\s+INTERM[ÈE]DE\b")
HEADING_MIXED = re.compile(r"^(?:Acte|Scène|Scene)\s+(?:[IVXLC\d]+\b|premier|première|unique|dernière|\w+ième)"
                           r"|^[Ff]in\s+(?:de|du|des)\b"
                           r"|^(?:Prologue|[ÉE]pilogue|Interm[èe]de)\.?$")   # « Prologue. » seul (Psyché)
# Liste des personnages : majuscule initiale obligatoire. Sans cela, la ligne de prose coupée
# « personnages qu'il représente… » (L'Impromptu de Versailles) ouvrait la liste et avalait 80 lignes.
CAST = re.compile(r"^(?:PERSONNAGES|ACTEURS|DRAMATIS|Personnages|Acteurs|Dramatis)\b")
# Étiquettes de chant ou de mise en scène écrites comme un nom : ce ne sont pas des personnages.
# On les traite en didascalie : les paroles qui suivent restent au chanteur précédent.
LABEL = re.compile(r"^(?:LES?\s+M[ÊE]MES?|AIR|(?:PREMIER|DEUXI[ÈE]ME|SECOND|TROISI[ÈE]ME|DERNIER)\s+COUPLET"
                   r"|COUPLETS?|RONDEAU|STROPHE|ANTISTROPHE|[ÉE]PODE|R[ÉE]CITATIF|ROMANCE|ARIETTE|CAVATINE"
                   r"|DUO|TRIO|QUATUOR|FINALE?|VAUDEVILLE|CHANSON|REFRAIN|D[ÉE]COR|NOTA)\b")
# Étiquettes de section d'un livre (Spinoza, poésies, flore, catéchisme) : si elles font plus de 20 %
# des répliques, le document n'est pas une pièce.
SECTION = re.compile(r"^(?:D[ÉE]MONSTRATION|SCOLIE|COROLLAIRE|PROPOSITION|AXIOME|EXPLICATION|D[ÉE]FINITION"
                     r"|LEMME|BALL?ADE|L['’]ENVOY|ENVOI|RONDEAU|CHANSON|OBSERVATIONS|EXSICCATA|DISTRIB"
                     r"|DEMANDE|R[ÉE]PONSE|NOTE|EXEMPLE)")
# Ligne de distribution d'un ballet dans Molière : « SIX DIEUX MARINS dansans. »
BALLET_CAST = re.compile(rf"^[{UP}][{UP}\s,'’\-]+\s(?:dansans|chantans|dansants|chantants)\.?$")
# Courte didascalie sans parenthèses (mode strict) : « Il sort. », « Elles entrent. », « Il frappe à sa porte. »
SHORT_DIDA = re.compile(r"^(?:Il|Elle|Ils|Elles|On)\s+(?:sort|sortent|entre|entrent|rentre|rentrent|frappe|frappent"
                        r"|s['’]en va|s['’]en vont|se retire|se retirent|dispara[îi]t|disparaissent)\b[^.!?]{0,40}\.$")
# Même chose avec un personnage pour sujet : « Lélie sort. », « Sganarelle sort, et rentre aussitôt… ».
# Le sujet doit être un nom déjà locuteur dans le document (voir named_stage). Pas de « fait » ni de
# « revient » : « L'Amour fait perdre le repos. », « Amphitryon revient, qui me paroît content. » sont des vers.
NAMED_DIDA = re.compile(r"^(?P<subj>[^\s,.!?;:]+(?:\s+[^\s,.!?;:]+){0,3}?)\s+(?:sort|sortent|entre|entrent|rentre"
                        r"|rentrent|frappe|s['’]en va|se retire|dispara[îi]t)\b[^!?]{0,150}\.$")
# Ligne de texte qui commence comme une réplique « NOM. texte » ou « NOM (2), à Duchotel… » (format que
# ALONE et INLINE ne reconnaissent pas) : ce n'est jamais la suite de la réplique en cours.
REPLY_LIKE = re.compile(rf"^(?P<name>{NAME})(?:\s\(\d+\))?\s*[.,:]")
POS_NUM = re.compile(rf"^({NAME})\s*\(\d{{1,2}}\)")      # numéro de position sur scène après le nom (Feydeau)
NOTE_START = re.compile(r"^\[(?:Note\s*)?\d+")            # note de bas de page « [Note 24: … » (Gutenberg)
# Marque de parole : question, exclamation ou pronom de dialogue. Une scène où 90 % des répliques n'en ont
# aucune n'est pas un dialogue (distribution « BLOUNT : Dailly. », sommaire de revue, page de titre).
SPEECH = re.compile(r"[?!]|\b(?:je|j['’]|tu|vous|nous|moi|me|m['’]|te|t['’]|toi|mon|ma|mes|ton|ta|tes|votre|vos"
                    r"|notre)\b", re.I)
LEAD_NUM = re.compile(r"^(\d{1,4})\s+(?=\S)")          # numéro de vers en début de ligne (Psyché : « 540 À ne… »)
HEAD_WORDS = {"ACTE", "SCÈNE", "SCENE", "FIN", "PERSONNAGES", "ACTEURS", "CHAPITRE", "LIVRE", "TOME",
              "PARTIE", "PRÉFACE", "NOTE", "NOTES", "TABLE", "AVERTISSEMENT", "NOTICE", "INTRODUCTION"}
INTERJ = {"OUI", "NON", "AH", "OH", "HÉ", "HO", "EH", "HA", "HEIN", "BON", "FIN", "ET", "MAIS", "OR",
          "DE", "DU", "DES", "LE", "LA", "LES", "UN", "UNE", "EN", "AU", "AUX", "À", "A"}   # mots seuls, pas des noms
# Vrai chiffre romain (« IV », « XII »), pas un nom fait des mêmes lettres (« MIMI », « LILI »)
ROMAN = re.compile(r"^(?=[IVXLCDM])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
# Numéro de scène en chiffres romains (1 à 99) : seul sur sa ligne (« XVII. ») c'est un titre ;
# en tête de ligne (« II. La côte d'Afrique ») il ouvre un nouveau bloc.
SCENE_NUM = r"(?=[IVXL])(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
NUM_LINE = re.compile(rf"^{SCENE_NUM}\.?$")
NUM_START = re.compile(rf"^{SCENE_NUM}\.\s")
# mots qui signalent un titre (ballet, section de livre, « PREMIÈRE JOURNÉE » du théâtre espagnol) et pas un personnage
BALLET = re.compile(r"\b(?:ENTR[ÉE]E|BALLET|MENUET|ACTE|SC[ÈE]NE|INTERM[ÈE]DE|DIALOGUE|PARTIE|CHAPITRE|LIVRE|TOME"
                    r"|JOURN[ÉE]ES?)\b")
# Numéro de scène écrit en toutes lettres sans le mot SCÈNE : « PREMIÈRE. Un salon. » (Mérimée, Sardou, Wagner)
ORDINAL = re.compile(r"(?:PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME"
                     r"|HUITI[ÈE]ME|NEUVI[ÈE]ME|DIXI[ÈE]ME|DERNI[ÈE]RE)")
# Courte didascalie en minuscule placée en tête de réplique, juste sous le nom : « entrant. Une dame… »,
# « à part. Peste ! » (Mérimée). Première phrase de 60 caractères au plus, terminée par un point.
LEAD_DIDA = re.compile(r"^[a-zà-ÿœæ][^.!?]{0,58}\.\s+(?=\S)")
# Mode strict : marques d'un paragraphe de ROMAN dialogué (Ségur…), pas de théâtre. Tiret ou guillemet
# de dialogue en tête, ou incise de narration (« dit-il », « s'écria », « répondit », « se leva »…).
DIALOGUE_START = re.compile(r"^(?:—|–|--|«)")
NARRATION = re.compile(r"\b(?:dit|répondit|reprit|demanda|ajouta|s['’]écria|cria|murmura|continua)(?:-t)?-"
                       r"(?:il|elle|ils|elles|on)\b|\bs['’]écri(?:a|èrent)\b|\brépondit\b|\bse (?:leva|tourna)\b",
                       re.I)

VERSE_NUM = re.compile(r"\s{2,}\d{1,5}$")              # numéro de vers en fin de ligne (Corneille)
NOTE_REF = re.compile(r"\[\d+\]")                       # appel de note « [441] »
NOTE_LINE = re.compile(r"^\[\d+\]")                     # ligne de note « [238] Col rabattu… »
DIDA_LINE = re.compile(r"^(?:\([^()]*\)|_[^_]+_|\[[^\[\]]*\])[.,;:]?$")   # ligne qui n'est QUE didascalie


def strip_accents(s):
    """« Élide » → « elide » : sert aux titres et aux 13-grammes (éditions d'orthographes variées)."""
    s = s.lower().replace("œ", "oe").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean_name(name):
    """Nom normalisé : capitales, sans chiffre (« POURCEAUGNAC1 »), espaces simples, sans point final."""
    name = re.sub(r"\d+", "", name)
    return re.sub(r"\s+", " ", name).strip(" .:,").upper()


def valid_name(name):
    """Écarte ce qui ressemble à un nom mais n'en est pas un : titres, interjections, chiffres romains."""
    words = name.split()
    return (sum(c.isupper() for c in name) >= 2 and len(name) <= 40 and words
            and words[0] not in HEAD_WORDS and name not in INTERJ and not ROMAN.match(name.replace(" ", "")))


def classify(line):
    """Rôle d'une ligne (déjà nettoyée) : (genre, contenu).

    Genres : blank, skip (numéro de page, note), heading, cast, dida, speaker, text.
    Pour un speaker, le contenu est (nom, texte sur la même ligne, forme « nue » sans ponctuation).
    """
    if not line:
        return "blank", None
    line = POS_NUM.sub(r"\1", line)            # « LE GÉNÉRAL (3), devant la table. » (Feydeau) → sans « (3) »
    if line.isdigit() or NOTE_LINE.match(line):
        return "skip", None
    if CAST.match(line):
        return "cast", None
    if HEADING_UP.match(line) or HEADING_MIXED.match(line) or NUM_LINE.match(line):
        return "heading", None
    if DIDA_LINE.match(line) or BALLET_CAST.match(line):
        return "dida", None
    for pattern in (ALONE, INLINE):
        m = pattern.match(line)
        if m and BALLET.search(m.group("name")):
            return "heading", None              # « TROISIÈME ENTRÉE DE BALLET », « DANS LE SECOND ACTE »
        if m and LABEL.match(clean_name(m.group("name"))):
            return "dida", None                 # « LES MÊMES, X entre. », « AIR : … » : pas un personnage
        if m and valid_name(clean_name(m.group("name"))):
            rest = m.groupdict().get("rest") or ""
            bare = pattern is ALONE and not m.group("dida") and not m.group("end")
            return "speaker", (clean_name(m.group("name")), rest, bare)
    return "text", line


def clean_line(line):
    """Retire l'indentation, les numéros de vers et les appels de note."""
    return NOTE_REF.sub("", VERSE_NUM.sub("", line.strip())).strip()


def strip_verse_numbers(lines):
    """Retire les numéros de vers placés en tête de ligne (« 540 À ne vous point mentir »).

    Décision au niveau du document : au moins 10 lignes numérotées, 90 % de multiples de 5 et une suite
    croissante. Une date (« 14 novembre ») ou un montant (« 162 francs ») isolés ne sont donc pas touchés.
    """
    nums = [int(m.group(1)) for m in map(LEAD_NUM.match, lines) if m]
    if len(nums) < 10 or sum(n % 5 == 0 for n in nums) < 0.9 * len(nums):
        return lines
    if sum(b > a for a, b in zip(nums, nums[1:])) < 0.9 * (len(nums) - 1):
        return lines
    return [LEAD_NUM.sub("", x) for x in lines]


def wrapped_dida(lines, i):
    """Nombre de lignes de suite d'une didascalie coupée à 72 colonnes après « NOM, didascalie… » (0 sinon).

    Exemple du kit : « SGANARELLE, … sans » puis « apercevoir Valère ni Lucas. » puis une ligne vide.
    Conditions : nom suivi de « , », pas de ponctuation finale, ligne longue (≥ 55 caractères : texte
    recoupé), suite en minuscule (un vers commence par une capitale), ligne vide au plus 3 lignes après.
    """
    line = lines[i]
    m = ALONE.match(line)
    if not (m and (m.group("dida") or "").lstrip().startswith(",") and not m.group("end")
            and len(line) >= 55 and not re.search(r"[.:)]$", line)):
        return 0
    if i + 1 >= len(lines) or not lines[i + 1] or not lines[i + 1][0].islower():
        return 0
    end = next((j for j in range(i + 1, min(i + 5, len(lines) + 1)) if j == len(lines) or not lines[j]), None)
    return end - i - 1 if end is not None else 0


def classify_text(text, cleanup=True):
    """Classe toutes les lignes, en sautant les didascalies écrites sur plusieurs lignes « (… \\n …) ».
    cleanup=False : sans retirer la fin des didascalies coupées (règles d'origine, pour la validation figée)."""
    lines = strip_verse_numbers([clean_line(x) for x in text.split("\n")])
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("(") and ")" not in line:
            # didascalie ouverte : on cherche la parenthèse fermante dans les 5 lignes suivantes
            end = next((j for j in range(i + 1, min(i + 6, len(lines)))
                        if re.search(r"\)[.,;:]?$", lines[j])), None)
            if end is not None and all(classify(lines[j])[0] in ("text", "dida") for j in range(i + 1, end + 1)):
                out.append(("dida", None))
                i = end + 1
                continue
        kind = classify(line)
        k = wrapped_dida(lines, i) if kind[0] == "speaker" and cleanup else 0
        if kind[0] == "speaker" and cleanup and not k and not kind[1][1] and i + 1 < len(lines):
            # « DOÑA FRANCISCA. » puis « serrant dans ses bras doña Maria. Chère Maria ! » : on retire la
            # didascalie en tête de la ligne suivante (une vraie réplique commence par une capitale).
            lines[i + 1] = LEAD_DIDA.sub("", lines[i + 1], count=1)
        out.append(kind)
        out += [("dida", None)] * k           # la fin de la didascalie coupée n'entre pas dans la réplique
        i += 1 + k
    return out


# ---------------------------------------------------------------------------------------------
# 2. Assembler les lignes en répliques, puis en scènes
# ---------------------------------------------------------------------------------------------

def turn_text(lines):
    """Texte d'une réplique. Vers : un vers par ligne. Prose coupée à 72 colonnes : lignes recollées.

    On reconnaît la prose recoupée à une ligne (après la première) qui commence par une minuscule :
    en vers classiques, chaque vers commence par une capitale.
    """
    lines = [x for x in (re.sub(r"\s+", " ", x).strip() for x in lines) if x]
    if lines:
        lines[0] = re.sub(r"^[.\s—–-]+", "", lines[0])     # « . — Voilà » (Guignol) → « Voilà »
    lines = [x for x in lines if x]
    if any(x[0].islower() for x in lines[1:]):
        return " ".join(lines)
    return "\n".join(lines)


def finalize_scene(turns, strict=True):
    """Nettoie une scène : retire les répliques vides, fusionne deux répliques de suite du même
    personnage, et renvoie None si la scène a moins de 2 répliques ou de 2 personnages.
    Le retrait des échos de vers partagés (remove_echo) ne se fait qu'en mode strict (corpus) : le texte
    de Molière (kit, Wikisource) n'a pas cet artefact, et ses échos voulus (« Comment ? ») doivent rester."""
    merged = []
    for speaker, lines in turns:
        text = turn_text(lines)
        if not re.search(r"[a-zà-ÿœ]", text):     # vide, ou tout en capitales : un titre, pas une réplique
            continue
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += "\n" + text
        else:
            merged.append({"speaker": speaker, "text": text})
    if strict:
        remove_echo(merged)
    if len(merged) < 2 or len({t["speaker"] for t in merged}) < 2:
        return None
    return merged


def remove_echo(turns):
    """Vers partagé mal rendu (Cyrano, Chantecler) : la réplique N+1 recopie la dernière ligne de la
    réplique N avant sa propre moitié de vers. On retire ce préfixe recopié.

    On compare au texte ORIGINAL de la réplique précédente (pour nettoyer les chaînes de 3 répliques ou
    plus). Un écho complet voulu par l'auteur (« Bonjour. » → « Bonjour. ») est gardé, et on ne retire
    rien s'il ne resterait que de la ponctuation (« Les autres ? » → « ? ») : la réplique serait vide.
    """
    originals = [t["text"] for t in turns]
    for i in range(1, len(turns)):
        prev = originals[i - 1].rsplit("\n", 1)[-1].strip()
        text = turns[i]["text"]
        if len(prev) >= 6 and text.startswith(prev):
            rest = text[len(prev):].lstrip()
            if any(c.isalpha() for c in rest):          # il reste au moins une lettre
                turns[i]["text"] = rest


def known_speaker(line, counts, dot=True):
    """« D’ÉPERNON Allons, courage ! » (nom sans ponctuation), « SOSIE. Mais quelle idée ! » (nom, point,
    réplique ; seulement si dot=True) ou « POITRINAS, entrant … » (didascalie de plus de 100 caractères) :
    reconnu comme un nom SEULEMENT si ce nom est déjà au moins 3 fois locuteur dans le document.
    Renvoie (nom, texte de la réplique) ou None."""
    words = line.split()
    for k in range(min(6, len(words) - 1), 0, -1):          # du nom le plus long au plus court
        name, rest = " ".join(words[:k]), " ".join(words[k:])
        bare = name.rstrip(",.") if dot else name.rstrip(",")
        if not re.fullmatch(NAME, bare) or counts[clean_name(name)] < 3:
            continue
        if name.endswith(","):                               # longue didascalie : pas de réplique sur la ligne
            return clean_name(name), ""
        if re.match(rf"[{UP}«—]", rest):
            return clean_name(name), rest
    return None


def named_stage(line, counts):
    """« Lélie sort. » : didascalie si le sujet est un personnage déjà au moins 2 fois locuteur (accents ignorés)."""
    m = NAMED_DIDA.match(line)
    return bool(m) and any(c >= 2 and strip_accents(n) == strip_accents(m.group("subj")) for n, c in counts.items())


def reply_like(line):
    """« MÉNECHME. Comment cela ? » : la ligne commence comme une réplique d'un autre personnage
    (ou comme un bloc numéroté « II. La côte d'Afrique », ou une note « [Note 24: … »)."""
    m = REPLY_LIKE.match(line)
    return (bool(NUM_START.match(line) or NOTE_START.match(line))
            or (bool(m) and valid_name(clean_name(m.group("name")))))


def paragraph_at(events, i):
    """Lignes du paragraphe qui commence en i : les lignes de texte consécutives."""
    j = i
    while j < len(events) and events[j][0] == "text":
        j += 1
    return [payload for _, payload in events[i:j]]


def narrative_paragraph(lines):
    """Paragraphe de récit d'un roman dialogué (Ségur…), à ne pas mettre dans la bouche d'un personnage :
    il commence par un tiret ou un guillemet de dialogue, contient une incise (« dit-il », « s'écria »…),
    ou dépasse MAX_PARA caractères sans être des vers (vers : lignes courtes qui commencent par une capitale)."""
    text = " ".join(lines)
    verse = all(len(x) <= 80 and x[0].isupper() for x in lines)
    return bool(DIALOGUE_START.match(lines[0]) or NARRATION.search(text) or (len(text) > MAX_PARA and not verse))


def lookahead(events):
    """Pour chaque position i : genre du prochain événement de structure (speaker, heading, title, cast), et
    nombre de lignes et de caractères de texte d'ici là. Sert au mode strict (regarder en avant)."""
    n = len(events)
    next_kind, gap_lines, gap_chars = [None] * n, [0] * n, [0] * n
    kind_after, lines, chars = None, 0, 0
    for i in range(n - 1, -1, -1):                           # parcours à l'envers : on connaît la suite
        kind, payload = events[i]
        if kind in ("speaker", "heading", "title", "cast"):
            kind_after, lines, chars = kind, 0, 0
        elif kind == "text":
            lines, chars = lines + 1, chars + len(payload)
        next_kind[i], gap_lines[i], gap_chars[i] = kind_after, lines, chars
    return next_kind, gap_lines, gap_chars


def parse_scenes(text, strict=True, cleanup=True):
    """Découpe un texte de théâtre en scènes (listes de répliques {speaker, text}).

    strict=True (corpus inconnu) : un paragraphe qui suit une ligne vide dans une réplique est gardé
    seulement si un nom de personnage arrive peu après (5 lignes, 1 500 caractères au plus), s'il ne
    commence pas lui-même comme une réplique (« MÉNECHME. Comment ? ») et s'il n'a pas l'air d'un récit
    (narrative_paragraph : « — Trente francs ! s'écria Thérèse ») : c'est un deuxième paragraphe ou
    un vers séparé par une ligne vide. Sinon on ferme la scène : mieux vaut perdre un paragraphe que
    mettre de la narration dans la bouche d'un personnage.
    strict=False (Molière, source sûre) : il prolonge toujours la réplique.
    cleanup=False : règles d'origine, sans les nettoyages de didascalies (validation figée, voir VAL_SHA256).
    """
    events = classify_text(text, cleanup)
    # Un nom « nu » (sans point ni didascalie) vu une seule fois est plutôt un titre : « LE PETIT VOYAGE ».
    # Genre « title » (et non « heading ») : dans la liste des personnages, un rôle muet cité une seule fois
    # (ZÉPHYRE dans Psyché) ne doit pas fermer la liste ; hors de la liste, il ferme la scène comme un titre.
    counts = Counter(p[0] for kind, p in events if kind == "speaker")
    for i, (kind, payload) in enumerate(events):
        if kind == "speaker" and payload[2] and counts[payload[0]] == 1:
            events[i] = ("title", None)
        elif kind == "text":
            found = known_speaker(payload, counts, dot=cleanup)
            if found and strict and ORDINAL.fullmatch(found[0]):
                events[i] = ("heading", None)    # « PREMIÈRE. Un salon. » : « SCÈNE » manquant (Mérimée)
            elif found:
                events[i] = ("speaker", (found[0], found[1], False))
            # « Il sort. » : didascalie en mode strict seulement (chez Molière, « Il entre. » de Dorine est une
            # vraie réplique) ; « Lélie sort. » (sujet = un personnage de la pièce) : dans les deux modes.
            elif (strict and SHORT_DIDA.match(payload)) or (cleanup and named_stage(payload, counts)):
                events[i] = ("dida", None)
    next_kind, gap_lines, gap_chars = lookahead(events)
    scenes, turns, current = [], [], None
    state = "outside"          # outside | wait (nom lu, réplique attendue) | reply | after (ligne vide après réplique)
    in_cast, cast_lines = False, 0

    def close():
        nonlocal turns, current
        if current:
            turns.append(current)
        scene = finalize_scene(turns, strict)
        if scene:
            scenes.append(scene)
        turns, current = [], None

    for i, (kind, payload) in enumerate(events):
        if in_cast:                              # liste des personnages : ignorée jusqu'au premier titre
            if kind == "heading" or cast_lines > 80:   # (un « title », nom nu de la liste, ne la ferme pas)
                in_cast = False
            else:
                cast_lines += kind != "blank"
                continue
        if kind == "cast":
            close()
            state, in_cast, cast_lines = "outside", True, 0
        elif kind in ("heading", "title"):
            close()
            state = "outside"
        elif kind == "speaker":
            name, rest, _ = payload
            if current:
                turns.append(current)
            current = (name, [rest] if rest else [])
            state = "reply" if rest else "wait"
        elif kind == "blank":
            if state == "reply":
                state = "after"
        elif kind == "text":
            if state in ("wait", "reply"):
                current[1].append(payload)
                state = "reply"
            elif state == "after":
                near_reply = (next_kind[i] == "speaker" and gap_lines[i] <= 5 and gap_chars[i] <= 1500
                              and not reply_like(payload)    # « LOUISE. Les Prussiens. » : autre personnage
                              and not (strict and narrative_paragraph(paragraph_at(events, i))))
                if strict and not near_reply:      # vrai bloc de prose qui s'intercale : on ferme la scène
                    close()
                    state = "outside"
                else:                              # paragraphe ou vers de la même réplique
                    current[1].append(payload)
                    state = "reply"
        # dida et skip : ignorées, l'état ne change pas
    close()
    return scenes


def split_long(turns, size=MAX_TURNS):
    """Coupe une scène trop longue en morceaux réguliers de « size » répliques au plus."""
    if len(turns) <= size:
        return [turns]
    k = math.ceil(len(turns) / size)
    step = math.ceil(len(turns) / k)
    chunks = [turns[i:i + step] for i in range(0, len(turns), step)]
    return [c for c in chunks if len(c) >= 2 and len({t["speaker"] for t in c}) >= 2]


def scene_records(scenes, source, ident, play, doc_id, url, license, is_moliere):
    """Transforme les scènes d'un document en lignes du contrat de sortie (ids stables)."""
    records = []
    for n, scene in enumerate(scenes, 1):
        chunks = split_long(scene)
        for k, chunk in enumerate(chunks, 1):
            suffix = f"-{k}" if len(chunks) > 1 else ""
            records.append({"id": f"{source}:{ident}:{n:04d}{suffix}", "source": source, "play": play,
                            "doc_id": doc_id, "url": url, "license": license, "is_moliere": is_moliere,
                            "turns": chunk})
    return records


# ---------------------------------------------------------------------------------------------
# 3. Détecteur : ce document est-il vraiment du théâtre ?
# ---------------------------------------------------------------------------------------------

def doc_metrics(scenes, text):
    """Mesures qui distinguent une pièce d'un dictionnaire ou d'une table (où chaque « nom » est unique)."""
    turns = [t for s in scenes for t in s]
    if not turns:
        return {"n_turns": 0}
    counts = Counter(t["speaker"] for t in turns)
    body = "".join(t["text"] for t in turns)
    letters = sum(c.isalpha() for c in body) or 1
    return {
        "n_turns": len(turns),
        "n_speakers": len(counts),
        "recurrent": round(sum(c for c in counts.values() if c >= 3) / len(turns), 3),  # noms qui reviennent
        "distinct_ratio": round(len(counts) / len(turns), 3),       # ≈ 1 pour un dictionnaire
        "top_share": round(max(counts.values()) / len(turns), 3),   # un seul « nom » partout : pas une pièce
        "coverage": round(len(body) / max(1, len(text)), 3),        # part du document en répliques
        "median_len": statistics.median(len(t["text"]) for t in turns),
        "digits": round(sum(c.isdigit() for c in body) / letters, 3),  # tables de chiffres
        # part des répliques dont le « personnage » est une étiquette de section (SCOLIE, BALADE…)
        "sections": round(sum(c for s, c in counts.items() if SECTION.match(s)) / len(turns), 3),
    }


def theatre_verdict(m):
    """None si c'est du théâtre, sinon la raison du rejet."""
    if m["n_turns"] < MIN_DOC_TURNS:
        return "trop_peu_de_repliques"
    if m["sections"] > 0.2:
        return "etiquettes_de_section"          # Spinoza (SCOLIE), Charles d'Orléans (BALADE)…
    ok = (m["recurrent"] >= 0.7 and m["distinct_ratio"] <= 0.5 and m["top_share"] <= 0.7
          and (m["coverage"] >= 0.1 or m["n_turns"] >= 300)   # gros recueil (revue) : peu de théâtre mais beaucoup
          and m["median_len"] <= 400 and m["digits"] <= 0.05)
    return None if ok else "pas_du_theatre"


# ---------------------------------------------------------------------------------------------
# 4. Recouvrement : 13-grammes de mots partagés (minuscules, sans accents ni ponctuation)
# ---------------------------------------------------------------------------------------------

WORD = re.compile(r"\w+")


def fold_spelling(text):
    """Repli d'orthographe pour reconnaître deux éditions d'un même texte : « trouvoit » ≈ « trouvait »,
    « yeux » ≈ « ieux », « & » = « et », numéros de vers retirés (« 1035 » coupait les suites de mots).
    Sert SEULEMENT au dédoublonnage entre documents, jamais aux contrôles contre la validation."""
    text = re.sub(r"\d+", " ", strip_accents(text).replace("&", " et "))
    return text.replace("oi", "ai").replace("y", "i")


def ngram_hashes(text, n=NGRAM, fold=False):
    """Empreintes (entiers 64 bits) de toutes les suites de n mots du texte, sans doublon.

    Chaque mot reçoit une empreinte xxhash ; l'empreinte d'une suite combine les n empreintes
    (polynôme modulo 2^64, calculé d'un coup par numpy). fold=True : orthographe repliée (fold_spelling).
    """
    words = WORD.findall(fold_spelling(text) if fold else strip_accents(text))
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    h = np.fromiter((xxhash.xxh64_intdigest(w.encode()) for w in words), dtype=np.uint64, count=len(words))
    m = len(words) - n + 1
    out = np.zeros(m, dtype=np.uint64)
    for j in range(n):
        out = out * np.uint64(1099511628211) + h[j:j + m]
    return np.unique(out)


def turns_text(scenes):
    """Texte des répliques seules (sans les noms) : les 13-grammes ne sont pas coupés par les noms."""
    return "\n".join(t["text"] for s in scenes for t in s)


def shared(hashes, reference):
    """Nombre de 13-grammes distincts communs avec la référence (tableau trié)."""
    return int(np.isin(hashes, reference, assume_unique=True).sum()) if len(hashes) else 0


# ---------------------------------------------------------------------------------------------
# 5. Les pièces de Molière : kit (train/val) et Wikisource
# ---------------------------------------------------------------------------------------------

def load_kit(path):
    """Pièces du kit : séparées par une ligne « ===== », première ligne = titre."""
    plays = []
    for block in re.split(r"\n=====\n", path.read_text(encoding="utf-8")):
        block = block.strip("\n")
        if block:
            plays.append((block.split("\n", 1)[0].strip(), block))
    return plays


def load_ws_moliere(path, readme):
    """Pièces de data/moliere/moliere.txt : séparées par une ligne vide ; titre et URL lus dans le README
    (trois pièces commencent directement par la liste des personnages, sans titre)."""
    rows = re.findall(r"^\| (\d+) \| (.+?) \| \d+ \| \[1\]\((\S+?)\)", readme.read_text(encoding="utf-8"), re.M)
    blocks = [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    if len(rows) != len(blocks):
        raise ValueError(f"{len(blocks)} pièces dans moliere.txt mais {len(rows)} lignes dans le README")
    return [(title.strip(), url, block) for (_, title, url), block in zip(rows, blocks)]


def title_keys(title):
    """Formes comparables d'un titre : « Don Juan, ou le Festin de Pierre » → {« don juan festin de pierre »,
    « don juan », « festin de pierre »}. Articles, accents et ponctuation retirés."""
    t = strip_accents(title).replace("’", "'")
    parts = [t] + re.split(r",?\s+ou\s+", t)
    keys = set()
    for p in parts:
        p = re.sub(r"\bl'", "", p)
        p = re.sub(r"[^a-z' ]", " ", p)
        p = " ".join(w for w in p.split() if w not in {"le", "la", "les", "ou"})
        if p:
            keys.add(p)
    return keys


def same_title(a, b):
    return bool(title_keys(a) & title_keys(b))


def slug(title):
    return re.sub(r"[^a-z0-9]+", "_", strip_accents(title)).strip("_")


# ---------------------------------------------------------------------------------------------
# 6. Lecture du corpus en parallèle (un fichier .jsonl.zst par tâche)
# ---------------------------------------------------------------------------------------------

PREFILTER = re.compile(rf"^[ \t]*[{UP}][{UP}'’ .\-]{{1,40}}(?:[,(][^\n]{{0,100}})?(?:[.:]?[ \t]*$|\.(?:--|—))", re.M)
_REF = {}


def _init_worker(ref):
    """Références partagées par les processus : empreintes val (13 et 12 mots), Molière (13 mots),
    personnages des pièces de validation, liste d'exclusion {doc_id: raison}."""
    _REF.update(ref)


def load_exclusions(path=EXCLUSIONS):
    """Liste d'exclusion auditée {doc_id: raison} (une seule liste : le fichier JSON)."""
    return dict(json.loads(Path(path).read_text(encoding="utf-8"))["documents"])


def is_moliere_edition(doc, meta, turn_hashes):
    """Une édition de Molière : son nom dans l'id, le titre ou les auteurs, ou au moins 20 % des
    13-grammes de ses répliques présents chez Molière (kit, Wikisource) ou dans la validation.
    (« La Fille de Molière » de Fournier a le nom dans le titre mais pas les 13-grammes : elle n'est
    exclue entière que si elle partage aussi au moins MIN_SHARED 13-grammes.)"""
    about = strip_accents(" ".join(str(x) for x in (doc["id"], meta.get("title"), meta.get("authors"))))
    if "moliere" in about or "poquelin" in about:
        return True
    return shared(turn_hashes, _REF["val_mol"]) >= MOLIERE_SHARE * max(1, len(turn_hashes))


def val_speaker_overlap(turns, val_speakers):
    """Titre de la pièce de validation dont la scène reprend au moins 2 personnages distinctifs
    (suite ou pastiche : « La Conversion d'Alceste » avec ALCESTE et PHILINTE), sinon None."""
    names = {t["speaker"] for t in turns}
    for play, cast in val_speakers.items():
        if len(names & cast) >= 2:
            return play
    return None


def scene_reason(turns):
    """Raison de retirer une scène d'un document non-Molière, ou None pour la garder."""
    text = turns_text([turns])
    if shared(ngram_hashes(text, NGRAM_VAL), _REF["val12"]) > 0:
        return "scenes_retirees_val"
    if shared(ngram_hashes(text), _REF["mol"]) > 0:
        return "scenes_retirees_moliere"
    if val_speaker_overlap(turns, _REF["val_speakers"]):
        return "scenes_retirees_personnages_val"
    # Distribution, sommaire de revue, page de titre. Pour une scène de 2 ou 3 répliques, 10 % veut dire
    # « aucune » : « JUAN MENDO: Don CÉSAR BELMONTE, cavalier portugais. » (Mérimée) est retirée.
    if sum(bool(SPEECH.search(t["text"])) for t in turns) <= 0.1 * len(turns):
        return "scenes_retirees_sans_parole"
    return None


def process_file(path):
    """Traite un fichier du corpus : renvoie (scènes retenues, bilan par document candidat, nb de docs,
    nb de docs non français, doc_id de la liste d'exclusion rencontrés)."""
    records, rows, n_docs, n_non_fr, excluded = [], [], 0, 0, []
    for doc in read_docs(path):
        n_docs += 1
        if doc["id"] in _REF["exclusions"]:
            excluded.append(doc["id"])                 # écarté après audit à l'œil (liste d'exclusion)
            continue
        text, meta = doc["text"], doc.get("meta") or {}
        if meta.get("langid_group", "fr") != "fr":
            n_non_fr += 1
            continue
        lines = [x for x in text.split("\n") if x.strip()]
        n_hits = len(PREFILTER.findall(text))
        if n_hits < 20 or n_hits / max(1, len(lines)) < 0.04:
            continue                                   # pas même candidat : rien qui ressemble à des noms
        scenes = parse_scenes(text, strict=True)
        m = doc_metrics(scenes, text)
        row = {"id": doc["id"], "title": meta.get("title"), **m}
        reason = theatre_verdict(m) if m["n_turns"] else "trop_peu_de_repliques"
        if reason is None:
            ht = ngram_hashes(turns_text(scenes))
            h = np.union1d(ngram_hashes(text), ht)
            row["shared_val"] = shared(h, _REF["val"])
            row["shared_moliere"] = shared(h, _REF["mol"])
            # Exclusion du document ENTIER réservée aux éditions de Molière (orthographe parfois ancienne :
            # un filtre scène par scène pourrait y laisser passer une scène de validation).
            if is_moliere_edition(doc, meta, ht):
                if row["shared_val"] >= MIN_SHARED:
                    reason = "recouvrement_val"
                elif row["shared_moliere"] >= MIN_SHARED:
                    reason = "doublon_moliere"
        if reason is None:
            ident = doc["id"].split(":", 1)[1] if doc["id"].startswith(doc["source"] + ":") else doc["id"]
            # is_moliere toujours faux ici : seules moliere_kit et moliere_wikisource sont de Molière.
            recs = scene_records(scenes, doc["source"], ident, meta.get("title"), doc["id"],
                                 doc.get("url"), doc.get("license"), False)
            # Les autres recueils ne perdent que les scènes qui recouvrent (validation, Molière, personnages).
            dropped = Counter()
            for r in recs:
                why = scene_reason(r["turns"])
                if why:
                    dropped[why] += 1
                else:
                    records.append(r)
            row.update(dropped)
            if recs and sum(dropped.values()) == len(recs):
                reason = "toutes_scenes_retirees"
        row["reason"] = reason or "retenu"
        rows.append(row)
    return str(path), records, rows, n_docs, n_non_fr, excluded


# ---------------------------------------------------------------------------------------------
# 7. Assemblage et statistiques
# ---------------------------------------------------------------------------------------------

def file_stats(records):
    """Pièces, scènes, répliques, caractères et personnages d'un fichier de sortie."""
    turns = [(r["doc_id"], t) for r in records for t in r["turns"]]
    return {"pieces_ou_documents": len({r["doc_id"] for r in records}), "scenes": len(records),
            "repliques": len(turns), "caracteres": sum(len(t["text"]) for _, t in turns),
            "personnages_distincts_par_piece": len({(d, t["speaker"]) for d, t in turns}),
            "noms_de_personnages_distincts": len({t["speaker"] for _, t in turns})}


GENERIC_ROLE = re.compile(r"MUSICIEN|ENSEMBLE|GARDE|COMMISSAIRE|S[ÉE]NATEUR|\bET\b|^(?:PREMI|DEUXI|SECOND|TROISI)")


def distinctive_speakers(scenes):
    """Noms de personnages propres à une pièce : sans les rôles génériques (PREMIER MUSICIEN, LE GARDE…)."""
    return {t["speaker"] for s in scenes for t in s if not GENERIC_ROLE.search(t["speaker"])}


def build_moliere():
    """Scènes Molière (train, val) et références pour filtrer le corpus : empreintes val (13 et 12 mots),
    Molière (13 mots), personnages distinctifs de chaque pièce de validation."""
    manifest = {p["titre"]: p for p in json.loads(KIT_MANIFEST.read_text(encoding="utf-8"))["pieces"]}
    kit_train, kit_val = load_kit(KIT_TRAIN), load_kit(KIT_VAL)
    ws_plays = load_ws_moliere(WS_MOLIERE, WS_README)

    def kit_records(title, text, cleanup=True):
        scenes = parse_scenes(text, strict=False, cleanup=cleanup)
        n = manifest.get(title, {}).get("ebook_gutenberg")
        url = f"https://www.gutenberg.org/ebooks/{n}" if n else None
        return scenes, scene_records(scenes, "moliere_kit", slug(title), title, title, url, "Domaine public", True)

    train, val, val_parts, train_parts, kit_info, val_speakers = [], [], [], [], [], {}
    for title, text in kit_val:
        scenes, recs = kit_records(title, text, cleanup=False)   # validation figée (voir VAL_SHA256)
        val += recs
        val_parts += [text, turns_text(scenes)]
        val_speakers[title] = distinctive_speakers(scenes)
    for title, text in kit_train:
        scenes, recs = kit_records(title, text)
        train += recs
        train_parts.append(turns_text(scenes))
        kit_info.append((title, scenes))

    # Les éditions Wikisource des pièces de validation rejoignent la référence « val » (autre orthographe).
    ws_parsed = [(t, u, b, parse_scenes(b, strict=False)) for t, u, b in ws_plays]
    val_titles = [t for t, _ in kit_val]
    for t, u, b, scenes in ws_parsed:
        for v in val_titles:
            if same_title(t, v):
                val_parts += [b, turns_text(scenes)]
                val_speakers[v] |= distinctive_speakers(scenes)
    val_ref = np.unique(np.concatenate([ngram_hashes(x) for x in val_parts]))
    val_ref12 = np.unique(np.concatenate([ngram_hashes(x, NGRAM_VAL) for x in val_parts]))

    kit_ref = np.unique(np.concatenate([ngram_hashes(x) for x in train_parts]))
    decisions, kit_val_check = [], []
    for title, scenes in kit_info:
        kit_val_check.append({"titre": title, "13grammes_communs_val": shared(ngram_hashes(turns_text(scenes)), val_ref),
                              "12grammes_communs_val": shared(ngram_hashes(turns_text(scenes), NGRAM_VAL), val_ref12)})
    for t, u, b, scenes in ws_parsed:
        h = np.union1d(ngram_hashes(b), ngram_hashes(turns_text(scenes)))
        ht = ngram_hashes(turns_text(scenes))
        d = {"titre": t, "url": u, "titre_dans_kit_train": any(same_title(t, k) for k, _ in kit_train),
             "titre_dans_kit_val": any(same_title(t, v) for v in val_titles),
             "part_13grammes_dans_kit_train": round(shared(ht, kit_ref) / max(1, len(ht)), 4),
             "13grammes_communs_val": shared(h, val_ref)}
        if d["titre_dans_kit_val"]:
            d["decision"] = "exclue_validation"
        elif d["titre_dans_kit_train"] or d["part_13grammes_dans_kit_train"] >= 0.2:
            d["decision"] = "deja_dans_le_kit"
        elif d["13grammes_communs_val"] >= MIN_SHARED:
            d["decision"] = "exclue_validation"
        else:
            d["decision"] = "ajoutee_moliere_train"
            train += scene_records(scenes, "moliere_wikisource", slug(t), t, t, u, WS_LICENSE, True)
        if not d["titre_dans_kit_val"]:
            # toute édition d'une pièce d'entraînement sert de référence « doublon Molière »
            train_parts += [b, turns_text(scenes)]
        decisions.append(d)
    mol_ref = np.unique(np.concatenate([ngram_hashes(x) for x in train_parts]))

    # Molière a repris des vers de Don Garcie de Navarre (kit train) dans Le Misanthrope (val) : on retire
    # de l'entraînement toute scène qui partage ne serait-ce qu'un 12-gramme avec la validation
    # (12 mots et non 13 : « Je suis, je suis trahi, je suis assassiné » ne fait que 12 mots).
    removed = [r["id"] for r in train if shared(ngram_hashes(turns_text([r["turns"]]), NGRAM_VAL), val_ref12) > 0]
    train = [r for r in train if r["id"] not in set(removed)]
    info = {"scenes_train_retirees_car_vers_communs_avec_val": removed,
            "titres_kit_train": [t for t, _ in kit_train], "titres_kit_val": val_titles,
            "personnages_distinctifs_val": {k: sorted(v) for k, v in val_speakers.items()},
            "pieces_wikisource": decisions, "kit_train_contre_val": kit_val_check}
    ref = {"val": val_ref, "val12": val_ref12, "mol": mol_ref, "val_mol": np.union1d(val_ref, mol_ref),
           "val_speakers": val_speakers}
    return train, val, ref, info


def dedupe_across_documents(records):
    """Retire les scènes qui sont des doublons d'une autre édition (Wikisource et Gutenberg, Œuvres et
    Théâtre choisi) : une scène dont au moins 80 % des 13-grammes appartiennent déjà à un document
    antérieur est retirée. Les répétitions À L'INTÉRIEUR d'un document (refrains) sont gardées.
    Les 13-grammes sont pris sur l'orthographe repliée (fold_spelling) : « trouvoit » et « trouvait »,
    ou un numéro de vers collé au texte, ne cachent plus une deuxième édition de la même scène.
    Renvoie (scènes gardées, Counter des paires (document gardé, document doublon))."""
    owner, kept, dropped = {}, [], Counter()       # owner : empreinte → document qui l'a vue le premier
    for r in records:
        h = [int(x) for x in ngram_hashes(turns_text([r["turns"]]), fold=True)]
        others = Counter(owner[x] for x in h if x in owner and owner[x] != r["doc_id"])
        if h and sum(others.values()) >= DUP_SHARE * len(h):
            dropped[(others.most_common(1)[0][0], r["doc_id"])] += 1
            continue
        for x in h:
            owner.setdefault(x, r["doc_id"])
        kept.append(r)
    return kept, dropped


def dedupe_both_ways(records):
    """Second passage, dans les deux sens. Le premier passage ne compare une scène qu'aux documents DÉJÀ
    vus : une scène gardée tôt peut être entièrement recopiée dans une scène plus longue lue ensuite.
    Ici, une scène dont au moins 80 % des 13-grammes sont dans un AUTRE document encore gardé est retirée,
    de la plus courte à la plus longue : de deux copies, on garde la plus longue (orthographe repliée aussi).
    Renvoie (scènes gardées, Counter des paires (document gardé, document doublon))."""
    n = len(records)
    hs = [ngram_hashes(turns_text([r["turns"]]), fold=True) for r in records]
    sizes = np.array([len(h) for h in hs])
    doc_num = {d: k for k, d in enumerate(dict.fromkeys(r["doc_id"] for r in records))}
    doc = np.array([doc_num[r["doc_id"]] for r in records], dtype=np.int64)
    # Une ligne par (empreinte, scène), triée par empreinte puis document.
    S = np.repeat(np.arange(n), sizes)
    H = np.concatenate(hs) if n else np.empty(0, dtype=np.uint64)
    order = np.lexsort((doc[S], H))
    H, S = H[order], S[order]
    D = doc[S]
    # Combien de documents différents contiennent chaque empreinte ? (≥ 2 : présente ailleurs)
    new_h = np.r_[True, H[1:] != H[:-1]]
    new_pair = new_h | np.r_[True, D[1:] != D[:-1]]
    group = np.cumsum(new_h) - 1
    n_docs = np.bincount(group, weights=new_pair)[group]
    covered = np.bincount(S, weights=n_docs >= 2, minlength=n)
    # Candidats : scènes couvertes à 80 % quand toutes les scènes sont là ; on revérifie une à une.
    alive, dropped, names = np.ones(n, dtype=bool), Counter(), list(doc_num)
    for i in sorted((i for i in range(n) if sizes[i] and covered[i] >= DUP_SHARE * sizes[i]),
                    key=lambda i: (sizes[i], -i)):              # la plus courte d'abord, à égalité la plus tardive
        lo, hi = np.searchsorted(H, hs[i], "left"), np.searchsorted(H, hs[i], "right")
        partners, n_cov = Counter(), 0
        for a, b in zip(lo, hi):
            others = [int(D[k]) for k in range(a, b) if alive[S[k]] and D[k] != doc[i]]
            if others:
                n_cov += 1
                partners[others[0]] += 1
        if n_cov >= DUP_SHARE * sizes[i]:
            alive[i] = False
            dropped[(names[partners.most_common(1)[0][0]], records[i]["doc_id"])] += 1
    return [r for r, a in zip(records, alive) if a], dropped


def exclusion_report(exclusions, excluded, partial=False):
    """Bilan de la liste d'exclusion : documents écartés par raison, et un avertissement (pas une erreur)
    pour chaque doc_id listé mais introuvable dans le corpus lu."""
    missing = sorted(set(exclusions) - set(excluded))
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} doc_id de la liste d'exclusion introuvables dans le corpus lu"
                        + (" (essai partiel --limit-files)" if partial else ""))
    return {"documents_listes": len(exclusions), "documents_ecartes": len(set(excluded)),
            "par_raison": dict(Counter(exclusions[d] for d in set(excluded))),
            "ids_ecartes": sorted(set(excluded)), "introuvables": missing, "avertissements": warnings}


def corpus_files(limit=None):
    files = [p for s in SOURCES for p in sorted((CORPUS / s).glob("part-*.jsonl.zst"))]
    if limit:   # essai rapide : on prend des fichiers des deux sources
        per = max(1, limit // len(SOURCES))
        files = [p for s in SOURCES for p in sorted((CORPUS / s).glob("part-*.jsonl.zst"))[:per]]
    return files


def build(out=OUT, limit_files=None, workers=6):
    t0 = time.time()
    out = Path(out)
    train, val, ref, info = build_moliere()
    print(f"Molière : {len(train)} scènes train, {len(val)} scènes val", flush=True)
    exclusions = load_exclusions()
    ref["exclusions"] = exclusions

    files = corpus_files(limit_files)
    theatre, rows, n_docs, n_non_fr, excluded = [], [], 0, 0, []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(ref,)) as pool:
        for i, (path, recs, rws, nd, nf, ex) in enumerate(pool.imap(process_file, files), 1):
            theatre += recs
            rows += rws
            n_docs += nd
            n_non_fr += nf
            excluded += ex
            print(f"\r{i}/{len(files)} fichiers, {len(theatre)} scènes", end="", flush=True)
    print(flush=True)
    n_before_dedup = len(theatre)
    theatre, dup_pairs = dedupe_across_documents(theatre)   # imap garde l'ordre des fichiers : déterministe
    n_first_pass = len(theatre)
    theatre, dup_pairs2 = dedupe_both_ways(theatre)
    dup_pairs += dup_pairs2
    print(f"Doublons entre éditions : {n_before_dedup - n_first_pass} + {n_first_pass - len(theatre)} "
          f"scènes retirées (deux passages)", flush=True)
    excl_report = exclusion_report(exclusions, excluded, partial=bool(limit_files))

    sizes = {
        "moliere_train": write_docs(out / "moliere_train.jsonl.zst", train),
        "moliere_val": write_docs(out / "moliere_val.tmp.jsonl.zst", val),
        "theatre_train": write_docs(out / "theatre_train.jsonl.zst", theatre),
    }
    # La validation doit rester celle sur laquelle le BPB de référence a été mesuré (empreinte figée).
    # Si le nouveau fichier est identique à l'octet près, on garde l'ancien sans le toucher (un entraînement
    # peut être en train de le lire) ; sinon on le remplace et un avertissement est écrit plus bas.
    tmp, val_path = out / "moliere_val.tmp.jsonl.zst", out / "moliere_val.jsonl.zst"
    val_sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if val_path.exists() and hashlib.sha256(val_path.read_bytes()).hexdigest() == val_sha:
        tmp.unlink()
    else:
        tmp.replace(val_path)
    warnings = list(excl_report["avertissements"])
    if val_sha != VAL_SHA256:
        warnings.append(f"moliere_val a changé (SHA-256 {val_sha}, attendu {VAL_SHA256}) : "
                        "le BPB de référence 1,4746 n'est plus comparable, à recalculer sur ce fichier")
    reasons = Counter(r["reason"] for r in rows)
    rng = random.Random(1337)
    kept = [r["id"] for r in rows if r["reason"] == "retenu"]
    detector = sorted((r for r in rows if r["reason"] == "pas_du_theatre"), key=lambda r: -r["n_turns"])
    few = [r for r in rows if r["reason"] == "trop_peu_de_repliques"]
    stats = {
        "fichiers": {"moliere_train": file_stats(train), "moliere_val": file_stats(val),
                     "theatre_train": file_stats(theatre)},
        "lignes_ecrites": sizes,
        "corpus": {"fichiers_lus": len(files), "documents_lus": n_docs, "documents_non_francais": n_non_fr,
                   "candidats": len(rows), "retenus": reasons["retenu"],
                   "exclus": {k: reasons[k] for k in ("recouvrement_val", "doublon_moliere", "toutes_scenes_retirees",
                                                      "etiquettes_de_section", "pas_du_theatre",
                                                      "trop_peu_de_repliques")},
                   "par_source": {s: dict(Counter(r["reason"] for r in rows if r["id"].startswith(s)))
                                  for s in SOURCES}},
        "scenes_retirees": {
            k: sum(r.get(k, 0) for r in rows) for k in ("scenes_retirees_val", "scenes_retirees_moliere",
                                                        "scenes_retirees_personnages_val",
                                                        "scenes_retirees_sans_parole")},
        "scenes_retirees_par_document": [
            {k: r.get(k, 0) for k in ("id", "title", "scenes_retirees_val", "scenes_retirees_moliere",
                                      "scenes_retirees_personnages_val")}
            for r in rows if any(r.get(k) for k in ("scenes_retirees_val", "scenes_retirees_moliere",
                                                     "scenes_retirees_personnages_val"))],
        "scenes_retirees_doublon_inter_documents": n_before_dedup - len(theatre),
        "scenes_retirees_doublon_par_passage": {"premier_passage": n_before_dedup - n_first_pass,
                                                "second_passage_deux_sens": n_first_pass - len(theatre)},
        "doublons_principales_paires": [{"garde": a, "doublon": b, "scenes": n}
                                        for (a, b), n in dup_pairs.most_common(20)],
        "liste_exclusion": {"fichier": str(EXCLUSIONS.relative_to(ROOT)), **excl_report},
        "moliere_val_sha256": val_sha,
        "moliere_val_identique_reference": val_sha == VAL_SHA256,
        "avertissements": warnings,
        "moliere": info,
        "titres_moliere_train": sorted({r["play"] for r in train}),
        "titres_moliere_val": sorted({r["play"] for r in val}),
        "exclus_recouvrement_val": [{k: r.get(k) for k in ("id", "title", "shared_val")}
                                    for r in rows if r["reason"] == "recouvrement_val"],
        "exclus_doublon_moliere": [{k: r.get(k) for k in ("id", "title", "shared_moliere")}
                                   for r in rows if r["reason"] == "doublon_moliere"],
        "exemples_retenus": rng.sample(kept, min(15, len(kept))),
        "exemples_rejetes_detecteur": [{k: v for k, v in r.items()} for r in
                                       detector[:10] + rng.sample(few, min(15 - min(10, len(detector)), len(few)))],
        "parametres": {"max_repliques_par_scene": MAX_TURNS, "ngram": NGRAM, "ngram_scene_val": NGRAM_VAL,
                       "seuil_13grammes_edition_moliere": MIN_SHARED, "part_doublon_scene": DUP_SHARE,
                       "dedoublonnage_orthographe_repliee": "&→et, oi→ai, y→i, nombres retirés (fold_spelling)",
                       "limite_dedoublonnage": "seules les copies quasi exactes (au moins 80 % des 13-grammes "
                                               "repliés dans un autre document) sont retirées ; deux éditions "
                                               "plus éloignées (didascalies en ligne, découpage différent) "
                                               "peuvent rester toutes les deux (50 à 80 % de 13-grammes communs)",
                       "max_caracteres_paragraphe_isole_strict": MAX_PARA,
                       "part_edition_moliere": MOLIERE_SHARE,
                       "min_repliques_document": MIN_DOC_TURNS, "limit_files": limit_files},
        "duree_secondes": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m chatbot.theatre", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-files", type=int, default=None, help="essai rapide : nombre de fichiers du corpus")
    p.add_argument("--out", default=str(OUT), help="dossier de sortie")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    stats = build(args.out, args.limit_files, args.workers)
    print(json.dumps({k: stats[k] for k in ("fichiers", "corpus", "duree_secondes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
