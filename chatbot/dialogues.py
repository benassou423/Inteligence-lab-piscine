"""Vrais dialogues en français moderne, extraits des romans du corpus nettoyé (format « tirets »).

Pourquoi ? Le chatbot Molière a appris à répondre sur des scènes de théâtre classique. Pour un
chatbot qui parle un français COURANT, il faut des échanges modernes. Les romans en contiennent
des millions, écrits ainsi :

    --Où demeure-t-elle, ta mère? demanda le laquais.
    --Je trouverai bien. Ouvrez-moi seulement cette grande porte.

Chaque paragraphe qui commence par un tiret (« -- », « — » ou « – ») est une prise de parole ; deux
paragraphes de dialogue qui se suivent sont, le plus souvent, deux interlocuteurs qui se répondent.
On en fait des échanges au contrat « tirets » : tours "A" / "B" en alternance stricte (A commence),
une seule ligne par tour, tiret et incises de narration (« demanda le laquais ») retirés.

Règle d'or : DANS LE DOUTE, ON COUPE. Un paragraphe qui garde du récit (verbe au passé simple, plus
de 400 caractères, incise non retirée, deux locuteurs dans un même paragraphe…) n'entre pas dans
l'échange : il le FERME. On perd des tours, mais ceux qui restent sont de vraies répliques.

Sorties dans data/stages/chatbot_20261113/dialogues/ :
- romans_train.jsonl.zst / romans_val.jsonl.zst : échanges de romans (Gutenberg + Wikisource) ;
  la validation = des documents ENTIERS (~2 %, choisis par l'empreinte SHA-256 de leur id) ;
- theatre_moderne_train.jsonl.zst : les scènes EN PROSE de theatre_train (chatbot.theatre),
  converties au même contrat ;
- stats.json : volumes, distributions, raisons d'écart, ids à relire, durée.

Commande (CPU seulement, aucun entraînement) :

    uv run python -m chatbot.dialogues --workers 6
    uv run python -m chatbot.dialogues --limit-files 2 --out /tmp/essai   # essai rapide
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import time
import unicodedata
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import zstandard

from chatbot.format import clean_text
# Détection du vocabulaire cru : on RÉUTILISE celle du pré-entraînement (même liste, même seuil),
# pour que les deux jeux de données soient filtrés pareil. (Import propre : pretrain.py ne fait
# rien à l'import ; il charge seulement torch, ce qui coûte ~1 s.)
from chatbot.pretrain import crude_count, is_crude
from chatbot.theatre import KIT_VAL, MIN_SHARED, load_exclusions, load_kit, ngram_hashes, shared
from pipeline.io import read_docs, write_docs

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "stages" / "nettoye" / "08_exact_dedup"
SOURCES = ("gutenberg_fr", "wikisource_fr")
THEATRE_TRAIN = ROOT / "data" / "stages" / "chatbot_20261112" / "theatre" / "theatre_train.jsonl.zst"
OUT = ROOT / "data" / "stages" / "chatbot_20261113" / "dialogues"

# Raisons de chatbot/exclusions_theatre.json qui écartent aussi un roman. « pas_du_dialogue »
# n'en fait PAS partie : ce sont surtout les romans de la comtesse de Ségur, très dialogués, mal
# découpés en répliques de théâtre… mais excellents au format tirets.
EXCLUDED_REASONS = {"erotique_ou_obscene", "pas_francais_ou_illisible", "orthographe_trop_ancienne"}

MIN_TURNS, MAX_TURNS = 2, 40   # un échange a de 2 à 40 tours ; au-delà on le découpe
MAX_TURN_CHARS = 400           # plus long : c'est presque toujours du récit collé à la parole
MIN_DOC_TURNS = 20             # un document qui donne moins de 20 tours n'est pas un roman dialogué
VAL_PER_10000 = 200            # 2 % des documents en validation
ARCHAIC_MAX = 1.0              # orthographe ancienne : au plus 1 marque pour 1 000 mots
LONG_TURN = 40                 # un tour d'au moins 40 lettres sert d'empreinte (contrôle de fuite)

# ---------------------------------------------------------------------------------------------
# 1. Orthographe ancienne (« il étoit », « connoître », « ſ », « & ») : score par document
# ---------------------------------------------------------------------------------------------
# Avant ~1835, l'imparfait et le conditionnel s'écrivaient -oit / -oient (« étoit », « auroit »).
# On compte les mots en -oit qui ne sont PAS des mots modernes (liste blanche mesurée sur deux
# fichiers Gutenberg le 13/11 : « détroit », « assoit », « échoit »…), plus le s long « ſ » et
# l'esperluette « & » (mise pour « et » dans les vieilles éditions). On ne compte pas -oient : trop
# de verbes modernes en -oyer (« ils emploient », « envoient ») ; -oit suffit, il est partout.
# Mesure sur 107 documents tirés au hasard : textes modernes ≤ 0,7 marque pour 1 000 mots, textes
# du XVIIe-XVIIIe ≥ 3. Seuil 1,0 entre les deux.
ARCHAIC_WORD = re.compile(r"\b[a-zà-öø-ÿœæç]+oit\b")
MODERN_OIT = set("""doit voit soit droit détroit toit endroit adroit maladroit étroit exploit reçoit
    aperçoit conçoit déçoit perçoit croit accroit boit prévoit revoit entrevoit pourvoit assoit
    rassoit asseoit rasseoit échoit choit déchoit benoit surseoit sursoit""".split())
ARCHAIC_SIGNS = ("ſ", "&")


def archaic_score(text):
    """Marques d'orthographe ancienne pour 1 000 mots (0 pour un texte moderne)."""
    words = max(1, len(text.split()))
    low = text.lower().replace("_", " ")
    marks = sum(1 for w in ARCHAIC_WORD.findall(low) if w not in MODERN_OIT)
    marks += sum(text.count(c) for c in ARCHAIC_SIGNS)
    return 1000 * marks / words


# ---------------------------------------------------------------------------------------------
# 1 bis. Œuvres érotiques et dictionnaires : par auteur / titre, puis par densité d'euphémismes
# ---------------------------------------------------------------------------------------------
# is_crude (mots crus) laisse passer les romans érotiques qui parlent par EUPHÉMISMES (« croupe »,
# « tétons », « décharger »). Une revue indépendante du 13/11 a trouvé ~2 000 échanges venant d'au
# moins 17 œuvres érotiques. Deux barrières de plus :
# (a) auteurs et titres connus, relus à la main sur les 90 documents retenus les plus chargés en
#     euphémismes (dans le doute, on coupe : quelques romans licencieux mais littéraires partent) ;
# (b) un score sans liste : euphémismes + 3 × mots crus, pour 1 000 mots.
def normalize(s):
    """Minuscules sans accents : « Sacher-Masoch » et « Virmaître » → « sacher-masoch », « virmaitre »."""
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


EROTIC_NAMES = re.compile(r"(?:" + "|".join([
    # auteurs (pseudonymes compris) de littérature érotique ou de flagellation
    r"nismois", r"momas", r"fuckwell", r"van rod", r"dunan", r"sacher", r"masoch", r"virmaitre",
    r"kellec", r"effe geache", r"fourqueuse", r"furrya", r"dormienne", r"fontanges", r"chancennes",
    r"balencon", r"varley", r"villiot", r"\brebell\b", r"la vrille", r"\bista\b", r"ribourg",
    r"\bluisa\b", r"\bcoote\b", r"gervaise de latouche",
    # titres
    r"volupt", r"flagell", r"fouet", r"lesbos", r"erotique", r"\borgies?\b", r"callipyge", r"callypige",
    r"debauche", r"demi-vierges", r"vierges en fleur", r"venus en rut", r"venus a la fourrure",
    r"venus indienne", r"hermaphrodite", r"odor di femina", r"chair molle", r"enfant vicieuse",
    r"vice de lily", r"caprices du sexe", r"paris-eros", r"maquerelles", r"dom bougre",
    r"portier des chartreux", r"ecole des filles", r"rideau leve", r"hic et hec", r"teleny",
    r"\beveline\b", r"\bevelyne\b", r"danseuse russe", r"chanteuse allemande", r"emotions de suzette",
    r"geoles de dentelles", r"petites et grandes filles", r"secrets de poste", r"pommes d.amour",
    r"stations de l.amour", r"amours d.un gentleman", r"intime d.une soubrette", r"esclave amoureuse",
    r"odette et martine", r"roman de violette", r"rosiere malgre elle", r"souper des douze",
    r"journal d.une masseuse", r"pudique albion", r"le tutu", r"cydalise", r"soeur agnes",
    r"jeune fille a la page", r"degre des ages du plaisir", r"femme aux chiens", r"batteuses d.hommes",
    r"czarine noire", r"nuits chaudes", r"croquis du vice",
    # ajouts après relecture des 160 documents gardés les plus chargés (13/11, seconde passe)
    r"schwaeble", r"vaudere", r"blessebois", r"nefzaoui", r"jardin parfume", r"meslon",
    r"petite pervertie", r"fargeze", r"memoires amoureux", r"des vignons", r"betty petite fille",
    r"\bprecoce\b", r"enamourees", r"\bemery\b", r"du saussay", r"\bdamian\b", r"coeur-brulant",
    r"cousines de la colonelle", r"pantalon", r"suzie kerry", r"michette", r"choiseul-meuse",
    r"sauve ma rose", r"grand.halte", r"belle alsacienne", r"fellatores", r"fille de joie",
    r"ecrits par elle-meme",
]) + r")")
# Rubriques Gutenberg (champ meta.subjects) : « Erotic literature », « Flagellation -- Fiction ».
EROTIC_SUBJECTS = re.compile(r"erotic|flagellation|pornograph")
# Dictionnaires, glossaires : des « définitions » au tiret ne sont pas des répliques
# (gutenberg_fr:48852, « Le parler populaire des Canadiens français » : 1 615 faux échanges).
DICTIONARY = re.compile(r"dictionnaire|glossaire|lexique|vocabulaire|parler populaire|larousse|\bargot\b")

EUPHEMISM = re.compile(r"\b(?:pines?|nichons|tétons?|fesses|fessées?|croupes?|cuisses|décharg\w+|"
                       r"jouissances?|voluptés?|voluptueu\w+|verges?|vulves?|pucelages?|dépucel\w+|"
                       r"tribades?|gousses?|flagell\w+|fouett\w+|conins?|culs?\b(?!-)|bandai\w+|bandant\w*|"
                       r"lubri\w+|lascif\w*|lascive\w*|priap\w+|phallus|lesbien\w*|saphi\w+|"
                       r"godemich\w+|gamahuch\w+|suç\w+|sucer|suce|braquemart|chibre)\b")
# « chatte », « minette », « motte » ne sont pas dans la liste : chats et mottes de beurre (Colette,
# « La Chatte métamorphosée en femme » de Scribe sortaient en tête). « cul-de-jatte » non plus.
# Mesure du 13/11 sur les 6 447 documents retenus (ancienne liste) : au-dessus de 3 pour 1 000 mots,
# presque tout est érotique ; en dessous, surtout de la littérature (Zola, Colette, Pergaud).
EROTIC_MAX = 3.0


def erotic_score(text):
    """(euphémismes + 3 × mots crus) pour 1 000 mots."""
    low = text.lower()
    return 1000 * (len(EUPHEMISM.findall(low)) + 3 * crude_count(low)) / max(1, len(low.split()))


def about(doc):
    """Ce qu'on sait du document en dehors de son texte : id, titre, auteurs, rubriques."""
    meta = doc.get("meta") or {}
    return normalize(" ".join(str(x) for x in (doc.get("id"), meta.get("title"), meta.get("authors"))))


# ---------------------------------------------------------------------------------------------
# 2. Les incises de narration : « , dit-il, » ; « ? demanda le laquais. » ; « s'écria-t-elle »
# ---------------------------------------------------------------------------------------------
# Verbes de parole du 1er groupe : radical + terminaison (passé simple -a, imparfait -ait, présent -e,
# 1re personne -ai / -ais, pluriel -èrent / -aient). [cç] et e? couvrent « commença », « interrogea ».
STEMS_1 = r"""demand ajout murmur continu répliqu observ expliqu déclar object insist soupir balbuti
    bégay hurl rican grogn grond souffl chuchot lan[cç] affirm protest exclam écri cri avou pens
    songe? remarqu rétorqu interroge? questionn gliss plaisant raill risqu hasard commen[cç] achev
    termin suggér propos ordonn command suppli implor appel tonn bredouill articul pronon[cç] répét
    corrige? approuv opin acquies[cç] appuy précis rectifi conseill annon[cç] racont interpell
    sanglot bougonn grommel marmonn marmott jur rassur encourage? insinu persifl concéd inform
    étonn emport impatient indign excus ripost tranch coup relev minaud fredonn beugl vocifér
    repren glapiss gémiss aboy hésit clam""".split()
VERB_1 = r"(?:" + "|".join(STEMS_1) + r")(?:a|ait|e|ai|ais|èrent|aient)"
# Verbes irréguliers (et 2e groupe) sous leurs formes courantes.
IRREGULAR = r"""dit disait dis dirent disaient fit faisait fait fis firent faisaient répondit répondait
    répond répondis répondirent reprit reprenait repris reprirent poursuivit poursuivait poursuit
    poursuivis interrompit interrompait interrompt interrompis conclut concluait conclus gémit
    rugit glapit enquit avertit renchérit repartit sourit souriait souris rit riait ajoutai
    s'enquit repartis repartirent disais faisais répondais reprenais fît geignit geignait""".split()
# (« repartis-je », « disais-je » : incises du narrateur à la 1re personne ; « fît-elle » : coquille
# d'OCR fréquente pour « fit-elle ».)
VERB = r"(?:" + VERB_1 + r"|" + "|".join(re.escape(v) for v in IRREGULAR) + r")"
# Pronom complément ou réfléchi devant le verbe : « lui dit-il », « se dit-elle », « s'écria ».
CLITIC = r"(?:(?:se|lui|leur|me|nous|vous|te)\s+|[smtl]['’])?"
# Sujet inversé : pronom soudé par un tiret (« dit-il », « demanda-t-elle », « dis-je ») ou nom
# (« demanda le laquais », « fit Paul », « reprit M. de Pomereux »).
PRONOUN_SUBJ = r"(?:-t-|-)(?P<pro>il|elle|ils|elles|on|je|nous|vous)\b"
DETERMINERS = r"le|la|les|l['’]|un|une|mon|ma|mes|son|sa|ses|ce|cet|cette|notre|votre|leur|leurs|des|du"
TITLES = r"M\.|MM\.|Mme|Mlle|monsieur|madame|mademoiselle|maître|sir|lord|lady|miss|mistress|don|dona"
UPPER = "A-ZÀ-ÖØ-ÞŒÆ"
NAME = rf"(?:d['’]|de\s+)?[{UPPER}][\w'’-]+(?:\s+(?:de\s+|d['’]|du\s+)?[{UPPER}][\w'’-]+)*"   # « d'Harmental »
# Sujet nominal : « le laquais » (déterminant + nom) ou « Suzanne », « M. de Pomereux ».
# Un adverbe peut s'intercaler : « cria de nouveau Hélène », « répondit tranquillement Paul ».
# De même un gérondif : « dit en rougissant Carmainges ».
ADVERB = (r"(?:\s+(?:de\s+nouveau|encore|alors|aussitôt|enfin|soudain|tout\s+à\s+coup|à\s+son\s+tour|"
          r"à\s+voix\s+basse|en\s+[a-zà-ÿ]+ant|[a-zà-ÿ]{3,}ment))?")
NOUN_SUBJ = (rf"{ADVERB}\s+(?P<noun>(?:{DETERMINERS})\s*[\w'’-]+|"
             rf"(?:(?:{TITLES})\s+(?:de\s+|d['’]|du\s+)?)?{NAME})")
# Formes au présent qui se confondent avec l'impératif (« demande la permission ! ») : elles ne
# sont acceptées qu'avec un pronom soudé (« demande-t-il »), jamais avec un nom.
PRESENT_ONLY = re.compile(r"(?:" + "|".join(STEMS_1) + r")e$")
# Formes de 1re personne (« dis », « repris », « demandai ») : sujet « -je » obligatoire. Avec un
# nom derrière, c'est un impératif (« Allons, dis Paul… ») et non une incise.
FIRST_PERSON = re.compile(r".*(?:ai|ais|is|us)$")

# Une incise commence juste après une virgule (ou un point-virgule), un point d'exclamation,
# d'interrogation ou des points de suspension, par un verbe EN MINUSCULE (après « ? », une incise
# ne prend pas de capitale). Un guillemet fermant ou un tiret peut s'intercaler :
# « Ivanitch,» dit Grinewitch », « Nous quitter!… — s'écria Clémence ».
INCISE = re.compile(rf"(?P<pre>,|;|!|\?|…|\.\.\.)\s*(?:»|—|–)?\s+(?P<verb>{CLITIC}{VERB})(?:{PRONOUN_SUBJ}|{NOUN_SUBJ})")
# Incise encadrée de tirets, sans virgule (Eugène Sue) : « Oui — dit-il — c'est cela. », parfois
# collés : « Je viens seulement—expliqua Coryse—pour vous dire ».
DASH_INCISE = re.compile(rf"\s?[—–]\s*(?P<verb>{CLITIC}{VERB})(?:{PRONOUN_SUBJ}|{NOUN_SUBJ})[^—–.!?…]{{0,120}}[—–]\s*")
# Fin de l'incise : prochaine ponctuation, sauf le point d'une abréviation (« M. de Pomereux »).
INCISE_END = re.compile(r"(?<!\bM)(?<!\bMM)(?<!\bMme)(?<!\bMlle)(?<!\bDr)(?<!\bSt)(?<!\bMgr)[.;:!?…,]")
# Bouts de récit qui prolongent souvent une incise après une virgule : « , dit-il, en riant, ».
# (« riant aux larmes », « en tournant la tête », « de sa lucarne », « dont les dents claquaient »,
# « qui riait »…)
NOT_PARTICIPLE = r"(?!(?:avant|maintenant|pendant|cependant|devant|durant|autant|pourtant|suivant|quant|moyennant)\b)"
INCISE_TAIL = re.compile(r"\s*(?:en\s+[\wà-ÿ]+ant\b|" + NOT_PARTICIPLE + r"[\wà-ÿ]{2,}ant\b|d['’]une?\s|avec\s|sans\s|"
                         r"tout\s+(?:bas|haut)\b|à\s+(?:voix|mi-voix|part|son\s+tour)\b|puis\s|"
                         r"les\s+yeux\s|l['’]air\s|non\s+sans\s|de\s+(?:sa|son|ses|leur|leurs)\s|dont\s|"
                         r"qui\s+(?:(?:ne|se|le|la|les|lui)\s+|[nsl]['’])*\w+(?:ait|aient)\b)")
# Un petit morceau sans pronom de dialogue (« très rouge », « un peu calmé ») suivi d'un bout de
# récit fait partie de l'incise : « dit Meaulnes, très rouge, en tournant la tête ».
DIALOGUE_PRONOUN = re.compile(r"\b(?:je|j'|tu|nous|vous|me|m'|te|t'|moi|toi|on|il|elle|ils|elles)\b", re.I)
# Après « ? demanda X, », la suite est-elle encore du récit ? Oui si elle parle d'« il / elle /
# lui / son » à l'imparfait, sans aucun « je / tu / vous / mon / votre » : « au moment où ils
# passaient ». (« Oui, dit-il, il viendra demain » reste : futur, c'est de la parole.)
# Imparfait (« passaient ») mais pas conditionnel (« pourrait », « serait », « voudrait » : de la
# parole, « qu'il pourrait proposer »).
IMPARFAIT = re.compile(r"\b\w{2,}(?<!rr|er|ur|dr|vr|ir)(?:ait|aient)\b")
FIRST_SECOND = re.compile(r"\b(?:je|j'|tu|nous|vous|me|m'|te|t'|moi|toi|mon|ma|mes|ton|ta|tes|notre|nos|"
                          r"votre|vos)\b", re.I)
THIRD = re.compile(r"\b(?:il|elle|ils|elles|lui|leur|leurs|son|sa|ses|se|s')\b", re.I)
SENTENCE_END = re.compile(r"(?<!\bM)(?<!\bMM)(?<!\bMme)(?<!\bMlle)(?<!\bDr)(?<!\bSt)(?<!\bMgr)[.;:!?…]")


def _capitalize(s):
    """Première lettre en capitale (« mes compliments » → « Mes compliments »)."""
    for i, c in enumerate(s):
        if c.isalpha():
            return s[:i] + c.upper() + s[i + 1:]
    return s


def _subject(m):
    """Qui parle d'après l'incise : « je », un nom propre (« suzanne ») ou un groupe (« le laquais »).
    Les pronoms il/elle ne disent pas QUI : on renvoie None."""
    if m.group("pro"):
        return "je" if m.group("pro") == "je" else None
    words = m.group("noun").split()
    return " ".join(words[:3]).lower().strip(".,")


def remove_incises(text):
    """Retire les incises de narration. Renvoie (texte, sujets nommés trouvés dans l'ordre).

    « Tenez, dit-elle, voilà tout. »              → « Tenez, voilà tout. »
    « Où est-elle? demanda le laquais ironique. »  → « Où est-elle? »
    « Bigre! fit-il en souriant, mes compliments » → « Bigre! Mes compliments »
    """
    subjects, start = [], 0
    for m in DASH_INCISE.finditer(text):
        subjects.append(_subject(m))
    text = DASH_INCISE.sub(" ", text)
    for _ in range(12):  # garde-fou : au plus 12 essais par tour
        m = INCISE.search(text, start)
        if not m:
            break
        verb = m.group("verb").split()[-1].split("'")[-1]
        # Présent + nom après une virgule = souvent un impératif ; après « ! » ou « ? » suivi d'une
        # minuscule, c'est bien une incise (« Et le vôtre aussi! réplique le pilote »).
        imperative = PRESENT_ONLY.fullmatch(verb) and m.group("pre") in ",;"
        if m.group("noun") and (imperative or FIRST_PERSON.fullmatch(verb)):
            start = m.end("verb")  # « Écoute, demande la permission » : impératif, pas une incise
            continue
        subjects.append(_subject(m))
        # Fin de l'incise : ponctuation suivante ; si la suite est encore du récit (« en riant »),
        # on avale aussi ce morceau.
        pos = m.end()
        while True:
            e = INCISE_END.search(text, pos)
            if e and text[e.start()] == ",":
                if INCISE_TAIL.match(text, e.end()):
                    pos = e.end()
                    continue
                nxt = INCISE_END.search(text, e.end())
                seg = text[e.end():nxt.start() if nxt else len(text)]
                if (nxt and text[nxt.start()] == "," and len(seg.split()) <= 3
                        and not DIALOGUE_PRONOUN.search(seg) and INCISE_TAIL.match(text, nxt.end())):
                    pos = nxt.end()
                    continue
                stop = SENTENCE_END.search(text, e.end())
                rest = text[e.end():stop.start() if stop else len(text)]
                if THIRD.search(rest) and IMPARFAIT.search(rest) and not FIRST_SECOND.search(rest):
                    e = stop      # le reste de la phrase est du récit : on l'avale aussi
            break
        end = e.start() if e else len(text)
        delim = text[end] if e else ""
        after = text[end + 1:].lstrip() if e else ""
        pre = m.group("pre")
        head = text[:m.start()]
        if pre in ",;":
            # « Tenez, dit-elle, voilà » → « Tenez, voilà » ; « maman, dit-il. » → « maman. »
            if delim == ",":
                text = f"{head}, {after}"
            else:
                text = f"{head}{delim or '.'} {after}"
        else:
            # « ta mère? demanda le laquais. Puis » → « ta mère? Puis » ; après « ! » ou « ? », la
            # parole qui reprend après la virgule prend une capitale.
            text = f"{head}{pre} {_capitalize(after) if delim == ',' else after}"
        text = text.strip()
        start = len(head)
    return text, [s for s in subjects if s]


# ---------------------------------------------------------------------------------------------
# 3. Du récit resté dans un tour : on le reconnaît pour FERMER l'échange
# ---------------------------------------------------------------------------------------------
# Passé simple à la 3e personne, formes SANS ambiguïté (« dit », « vit », « sourit » sont aussi du
# présent : on ne les met pas). Un personnage qui raconte au passé simple sera coupé aussi : tant pis.
PS_FORMS = set("""fut furent eut eurent fit firent prit prirent mit mirent reprit reprirent sortit
    sortirent partit partirent sentit sentirent répondit répondirent descendit descendirent rendit
    rendirent attendit attendirent entendit entendirent perdit perdirent vint vinrent tint tinrent
    revint revinrent devint devinrent put purent dut parut parurent reconnut reconnurent aperçut
    aperçurent reçut reçurent mourut moururent courut coururent accourut accoururent voulut
    voulurent sut surent crut crurent tut turent ouvrit ouvrirent couvrit couvrirent offrit
    offrirent assit assirent suivit suivirent poursuivit vécut connut connurent lut battit
    conduisit écrivit écrivirent rejoignit joignit éteignit craignit plaignit naquit disparut
    disparurent apparut apparurent comprit comprirent apprit apprirent surprit promit permit
    remit remirent soumit dormit endormit servit ressentit consentit mentit revit revirent
    interrompit trouva arriva approcha leva tourna retourna regarda alla resta passa tomba
    commença parla monta descendit""".split())   # (« but » n'y est pas : c'est surtout le nom « le but »)
# Les formes en -a ajoutées le 13/11 (« Frédéric arriva », « le savant se trouva ») n'ont pas
# d'autre sens qu'un passé simple : on les compte quel que soit le sujet (nom propre, groupe nominal).
# Pronom réfléchi + verbe en -a (« s'approcha », « se leva ») : passé simple aussi (le présent
# finit en -e) ; les futurs en -ra (« se fera ») sont traités comme pour PS1.
REFLEXIVE_PS = re.compile(r"\b(?:se\s+|s['’])([a-zà-öø-ÿœç]{3,}a)\b")
# Narration placée entre tirets au milieu d'une réplique : « Oui — et il sortit — je reviens. »
DASHED_SEGMENT = re.compile(r"\s*[—–]([^—–]{1,200})[—–]\s*")
# Passé simple du 1er groupe (« il parla », « elle se leva ») : pronom sujet, petits mots
# éventuels, puis un mot en -a ou -èrent. Les mots en -ra sont exclus (futur : « il sera »,
# « elle fera ») sauf une liste de vrais passés simples (« il entra », « elle montra »).
PS1 = re.compile(r"\b(?:il|elle|on|ils|elles)\s+(?:(?:ne|n['’]|se|s['’]|lui|leur|le|la|les|l['’]|y|en|"
                 r"me|m['’]|te|t['’]|nous|vous)\s*)*([a-zà-öø-ÿœç]{3,}(?:a|èrent))\b")
RA_PS = set("""entra rentra montra démontra tira retira attira admira soupira déclara considéra pleura
    jura assura murmura demeura dévora ignora adora implora espéra préféra inspira respira sépara
    prépara répara compara empara éclaira figura mesura serra""".split())
NOT_VERB_A = {"cela", "déjà", "voilà", "holà", "delà"}
QUOTED = re.compile(r"«[^»]*»")
# Une incise QUI RESTE au passé simple du 1er groupe (« hasarda-t-il », verbe hors de la liste)
# trahit la narration. « va-t-il », « sera-t-il », « y a-t-il » sont des questions : exclues.
LEFT_INCISE = re.compile(r"\b([a-zà-öø-ÿœç]{2,}a)-t-(?:il|elle|on)\b")
# Référence de bibliographie ou de catalogue (« Pfeifferi Bernardi, p. 560. ») : pas une parole.
REFERENCE = re.compile(r"\b(?:p|pp|t|vol|chap|fig|pl)\.\s*\d")
# Deux prises de parole dans le même paragraphe : « Oui. — Non, monsieur. »
INNER_DASH = re.compile(rf"[.!?…»]\s*(?:--|—|–)\s*[«{UPPER}]")


def has_passe_simple(low):
    """Un verbe au passé simple de narration dans ce texte (en minuscules, guillemets retirés) ?"""
    words = re.findall(r"[a-zà-öø-ÿœç]+", low)
    # 3e personne du pluriel en -èrent (« ils s'écrièrent ») : passé simple presque toujours (le
    # présent de « préférer », « espérer » s'écrit pareil : rare dans un dialogue, on coupe quand même)
    if any(w in PS_FORMS or w.endswith("èrent") for w in words):
        return True
    for verb in PS1.findall(low) + REFLEXIVE_PS.findall(low):
        if verb in NOT_VERB_A:
            continue
        if verb.endswith("ra") and verb not in RA_PS:
            continue  # futur (« il sera ») ou passé simple en -ra hors liste : on ne tranche pas
        return True
    return False


def drop_dashed_narration(text):
    """Retire un segment « — … — » qui raconte au passé simple : « Oui — et il se leva — oui. »"""
    def repl(m):
        return " " if has_passe_simple(m.group(1).lower()) else m.group(0)
    return clean_text(DASHED_SEGMENT.sub(repl, text)).strip()


LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæÇç]")
EN_WORDS = set("the and of to was he she his her you is it that with for not but have had would my".split())
FR_WORDS = set("le la les de et un une est que qui je vous il elle pas ne ce en du des à au mon ma".split())


def language_reason(text):
    """« sans_lettre » (aucune lettre latine : ponctuation, grec « Παπαῖ! ») ou « anglais »."""
    letters = [c for c in text if c.isalpha()]
    latin = sum(1 for c in letters if LATIN.match(c))
    if not letters or latin < 0.8 * len(letters):
        return "sans_lettre"
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    if en >= 3 and en > sum(w in FR_WORDS for w in words):
        return "anglais"
    return None


def narrative_reason(text):
    """Raison pour laquelle ce tour sent le récit (None si c'est une vraie réplique)."""
    lang = language_reason(text)
    if lang:
        return lang
    if len(text) > MAX_TURN_CHARS:
        return "trop_long"
    if INNER_DASH.search(text):
        return "plusieurs_locuteurs"
    plain = QUOTED.sub(" ", text)   # ce qui est entre guillemets est une parole citée, pas du récit
    low = plain.lower()
    if any(not v.endswith("ra") for v in LEFT_INCISE.findall(low)):
        return "incise_restante"
    if REFERENCE.search(text):
        return "reference_biblio"
    if has_passe_simple(low):
        return "passe_simple"
    return None


# ---------------------------------------------------------------------------------------------
# 4. D'un document à ses échanges
# ---------------------------------------------------------------------------------------------
PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n\s*")
DASH = re.compile(r"^\s*(?:--|—|–)\s*")


LEADING_JUNK = re.compile(r"^[\s:\-—–.]+")


def strip_lead(text):
    """Retire ce qui traîne en tête de tour : tiret simple restant (« -Il faut y compter »), tiret
    cadratin, deux-points laissé après le nom du personnage (« : Vous avez bien fait. »)."""
    return LEADING_JUNK.sub("", clean_text(text)).strip()


def turn_from_paragraph(paragraph):
    """Un paragraphe de dialogue → (texte du tour, sujets des incises, raison de rejet ou None)."""
    text = DASH.sub("", paragraph, count=1)
    text = re.sub(r"\s*\n\s*", " ", text)          # une seule ligne par tour
    text = text.replace("--", "—")                   # tiret double de Gutenberg → tiret cadratin
    text = clean_text(text)
    if text.startswith("«"):
        # Dialogue à la Mérimée : « Il a pensé me tuer, » dit le colonel. « Figurez-vous… »
        # Les guillemets ne font que marquer la parole : on les retire pour voir l'incise.
        text = clean_text(text.replace("«", " ").replace("»", " "))
    text, subjects = remove_incises(text)
    text = drop_dashed_narration(text)
    text = strip_lead(text)
    if text.count("«") != text.count("»"):
        # guillemet orphelin (le « » ouvrant ou fermant est dans un autre paragraphe) : on les retire
        text = clean_text(text.replace("«", " ").replace("»", " ")).strip()
    if not text or not re.search(r"\w", text):
        return text, subjects, "vide_ou_ponctuation"
    first = next((c for c in text if c.isalpha()), "")
    if first.islower():
        return text, subjects, "minuscule_initiale"   # « — les membres du conseil ; » : une liste
    if text[-1] in ",;":
        return text, subjects, "fin_de_liste"
    return text, subjects, narrative_reason(text)


def split_long(turns, max_turns=MAX_TURNS, min_turns=MIN_TURNS):
    """Découpe un échange trop long en morceaux de taille presque égale (≤ max_turns, ≥ min_turns)."""
    if len(turns) <= max_turns:
        return [turns]
    k = -(-len(turns) // max_turns)            # nombre de morceaux (arrondi au-dessus)
    size = -(-len(turns) // k)
    return [turns[i:i + size] for i in range(0, len(turns), size) if len(turns[i:i + size]) >= min_turns]


def with_speakers(texts):
    """Tours A/B en alternance stricte, A commence."""
    return [{"speaker": "AB"[i % 2], "text": t} for i, t in enumerate(texts)]


def extract_exchanges(text):
    """Texte d'un document → (liste d'échanges [liste de textes de tours], compteur des coupures).

    Un échange = des paragraphes de dialogue CONSÉCUTIFS. Il se ferme sur un paragraphe de récit,
    sur un tour refusé (raison comptée), ou quand une incise nomme un 3e locuteur ou le même
    locuteur deux fois de suite (l'alternance A/B serait fausse).
    """
    exchanges, current, slots = [], [], [None, None]
    cuts = Counter()

    def close():
        nonlocal current, slots
        if len(current) >= MIN_TURNS:
            exchanges.extend(split_long(current))
        current, slots = [], [None, None]

    for para in PARAGRAPH_BREAK.split(str(text).replace("\r\n", "\n")):
        if not DASH.match(para):
            close()                                   # récit : l'échange s'arrête
            continue
        turn, subjects, reason = turn_from_paragraph(para)
        if reason:
            cuts[reason] += 1
            close()
            continue
        # Contrôle d'alternance par les noms des incises : le tour k appartient au locuteur k % 2.
        if subjects:
            who = subjects[0]
            me, other = len(current) % 2, 1 - len(current) % 2
            if who == slots[other] or (slots[me] is not None and slots[me] != who):
                cuts["alternance_fausse"] += 1
                close()
                me, other = 0, 1
            slots[me] = who
        current.append(turn)
    close()
    return exchanges, cuts


# ---------------------------------------------------------------------------------------------
# 5. Filtres par document, puis traitement d'un fichier (un processus par fichier)
# ---------------------------------------------------------------------------------------------
_REF = {}


def _init_worker(ref):
    _REF.update(ref)


def build_reference():
    """Ce que chaque processus doit connaître : liste d'exclusion et 13-grammes de la validation
    Molière du kit (aucun échange qui recopie Le Misanthrope, Le Sicilien ou L'Avare)."""
    hashes = [ngram_hashes(block) for _, block in load_kit(KIT_VAL)]
    return {"exclusions": load_exclusions(), "val_moliere": np.unique(np.concatenate(hashes))}


def doc_reason(doc, ref):
    """Raison d'écarter le document AVANT extraction (None = on le garde)."""
    reason = ref["exclusions"].get(doc["id"])
    if reason in EXCLUDED_REASONS:
        return f"liste_{reason}"
    who = about(doc)
    subjects = normalize((doc.get("meta") or {}).get("subjects"))
    if EROTIC_NAMES.search(who) or EROTIC_SUBJECTS.search(subjects):
        return "erotique_auteur_ou_titre"
    if DICTIONARY.search(who) or "dictionaries" in subjects:
        return "dictionnaire"
    text = doc.get("text", "")
    if is_crude(text):
        return "vocabulaire_cru"
    if erotic_score(text) > EROTIC_MAX:
        return "euphemismes_erotiques"
    if archaic_score(text) > ARCHAIC_MAX:
        return "orthographe_ancienne"
    return None


def process_doc(doc, ref, min_doc_turns=MIN_DOC_TURNS):
    """Un document → (raison d'écart ou None, échanges, coupures)."""
    reason = doc_reason(doc, ref)
    if reason:
        return reason, [], Counter()
    exchanges, cuts = extract_exchanges(doc["text"])
    if sum(len(e) for e in exchanges) < min_doc_turns:
        return "peu_de_dialogue", [], cuts
    hashes = ngram_hashes("\n".join(t for e in exchanges for t in e))
    if shared(hashes, ref["val_moliere"]) >= MIN_SHARED:
        return "recouvre_validation_moliere", [], cuts
    return None, exchanges, cuts


def process_file(args):
    """Un fichier du corpus → liste de résultats par document (ordre du fichier conservé)."""
    path, min_doc_turns = args
    out = []
    for doc in read_docs(path):
        reason, exchanges, cuts = process_doc(doc, _REF, min_doc_turns)
        # Score érotique des documents GARDÉS : les 30 plus hauts sont listés pour relecture.
        score = None if reason else round(erotic_score(doc["text"]), 2)
        out.append({"id": doc["id"], "source": doc.get("source"), "url": doc.get("url"),
                    "license": doc.get("license"), "reason": reason, "erotic": score,
                    "title": (doc.get("meta") or {}).get("title"),
                    "exchanges": exchanges, "cuts": dict(cuts)})
    return path, out


def is_val(doc_id, per_10000=VAL_PER_10000):
    """Validation choisie par l'empreinte SHA-256 de l'id : stable d'un lancement à l'autre."""
    return int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 10000 < per_10000


# ---------------------------------------------------------------------------------------------
# 6. Dédoublonnage et contrôle de fuite (éditions multiples d'un même roman)
# ---------------------------------------------------------------------------------------------
def fingerprint(text):
    """Texte normalisé (minuscules, lettres et chiffres seulement) → empreinte courte. Deux
    éditions qui diffèrent par la typographie (-- / —, ’ / ') donnent la même empreinte."""
    norm = "".join(c for c in text.lower() if c.isalnum())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()


def long_turn_prints(turns):
    """Empreintes des tours d'au moins LONG_TURN lettres (« Oui. » est partout : on l'ignore)."""
    return {fingerprint(t["text"]) for t in turns if sum(c.isalpha() for c in t["text"]) >= LONG_TURN}


class Deduper:
    """Échange identique déjà vu → doublon ; moitié ou plus de ses longs tours déjà vus → doublon
    d'une autre édition (découpée un peu autrement). Le premier rencontré est gardé."""

    def __init__(self):
        self.exchanges, self.turns = set(), set()

    def seen(self, turns):
        whole = fingerprint("\n".join(t["text"] for t in turns))
        prints = long_turn_prints(turns)
        if whole in self.exchanges or (prints and len(prints & self.turns) * 2 >= len(prints)):
            return True
        self.exchanges.add(whole)
        self.turns |= prints
        return False


# ---------------------------------------------------------------------------------------------
# 7. Théâtre en prose moderne (sous-ensemble de theatre_train)
# ---------------------------------------------------------------------------------------------
STAGE_DIRECTION = re.compile(r"\s*\([^()]{1,600}\)\s*")   # Feydeau, Leblanc : didascalies de 300 caractères et plus
TERMINAL = tuple(".!?…»)")
LONG_LINE = 90          # une ligne de plus de 90 caractères n'est presque jamais un vers
PROSE_MIN_LONG = 0.10   # pièce en prose : au moins 10 % de lignes longues
MAX_THEATRE_TURN = 1000  # une tirade plus longue est un monologue : elle coupe l'échange


def long_line_share(scenes):
    """Part des lignes de plus de LONG_LINE caractères dans une pièce.

    Mesuré le 13/11 sur les 1 000 pièces de theatre_train : pièces en vers (Corneille, Racine,
    Hugo, Rostand) ≤ 0,07 ; pièces en prose (Feydeau, Labiche, Marivaux, Musset) ≥ 0,12. Un
    alexandrin fait 40 à 55 caractères ; une réplique en prose tient sur une ligne souvent longue.
    """
    lines = [l for s in scenes for t in s["turns"] for l in t["text"].split("\n") if l.strip()]
    return sum(len(l) > LONG_LINE for l in lines) / len(lines) if lines else 0.0


def is_verse_play(scenes):
    """Pièce en vers (ou trop peu de prose pour être sûr) : on la refuse entière."""
    return long_line_share(scenes) < PROSE_MIN_LONG


def is_verse_turn(text):
    """Réplique en vers, p. ex. un couplet chanté au milieu d'un vaudeville en prose.

    En prose, chaque ligne d'une réplique est un paragraphe fini par une ponctuation forte. En vers,
    la ligne s'arrête au bout du vers, souvent sans ponctuation forte, et la suivante commence par
    une capitale :  « Monsieur Octave, elle écrit, j'imagine, / Pour sa société de secours. »
    Vers = au moins deux lignes et au moins la moitié des passages à la ligne de cette forme.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    verse = sum(1 for a, b in zip(lines, lines[1:]) if not a.endswith(TERMINAL) and b[:1].isupper())
    return 2 * verse >= len(lines) - 1


VERSE_PLAY_SHARE = 0.10  # pièce refusée si plus de 10 % de ses répliques sur plusieurs lignes sont en vers

# Didascalies SANS parenthèses, collées dans la réplique par certaines éditions (revue du 13/11 :
# 9,6 % des tours lus en contenaient) : « Ah! ça, qu'est-ce que ça signifie? Elle est à droite un
# peu au-dessous de la table. » On retire une PHRASE quand elle a la forme d'une indication de
# scène : sujet à la 3e personne (« il », « elle », un nom, « le régisseur ») + verbe de jeu au
# présent (« sort », « remonte », « s'assied »…), sans « je / tu / vous », sans « ? » ni « ! »,
# courte (≤ 160 caractères). Un personnage qui dit « Il sort. » perd sa phrase : tant pis.
STAGE_VERB = (r"(?:sort|sortent|entre|entrent|rentre|rentrent|remonte|remontent|descend|descendent|"
              r"redescend|s['’]assied|s['’]assoit|s['’]asseyent|se\s+lève|se\s+lèvent|se\s+retourne|"
              r"se\s+dirige|s['’]éloigne|s['’]approche|s['’]arrête|se\s+précipite|se\s+jette|s['’]incline|"
              r"se\s+tait|passe|passent|traverse|gagne|revient|reviennent|paraît|paraissent|disparaît|"
              r"salue|saluent|embrasse|regarde|regardent|montre|désigne|indique|tend|prend|pose|remet|"
              r"ouvre|ferme|frappe|sonne|lit|écrit|tire|court|tombe|éclate|touche|remercie|hésite|"
              r"repousse|coupe|dépose|bondit|s['’]enfuit|"
              r"va\s+(?:pour\s+sortir|rejoindre|vers|au\s+fond|à\s+la\s+porte)|"
              r"(?:est|sont)\s+(?:à\s+droite|à\s+gauche|au\s+fond|assise?s?|debout|seule?s?|en\s+scène))")
# Sujet : pronom (« Il », « Elle »…), groupe nominal court (« Le régisseur », « L'abbé ») ou nom
# propre (« Jean ») ; puis éventuellement un pronom complément (« Jean le remercie »), puis le verbe.
STAGE_OBJ = r"(?:(?:le|la|les|lui|leur|y|en)\s+|l['’])?"
STAGE_SENTENCE = re.compile(
    rf"^(?:(?:[Ii]l|[Ee]lle|[Ii]ls|[Ee]lles|On|Tous|Toutes)\s+|"
    rf"(?:Le|La|Les|L['’]|Un|Une)\s*[\w'’-]+(?:\s+[\w'’-]+){{0,2}}\s+|"
    rf"(?!(?:Ça|Cela|Ceci|Voilà|Voici|Tout|Rien|Personne|Chacun|Ce|Qui|Quoi|Où|Quand|Comment|Pourquoi|"
    rf"Si|Mais|Et|Ou|Donc|Alors|Puis|Oui|Non|Ah|Oh|Eh|Hé|Bon|Bien|Enfin|Aussi|Ainsi|Celui|Celle|"
    rf"Lequel|Laquelle|Rien|Jamais|Toujours|Demain|Aujourd)\b)"
    rf"[{UPPER}][a-zà-ÿ'’-]+(?:\s+(?:et\s+)?[{UPPER}][a-zà-ÿ'’-]+|\s+(?:père|fils|aîné|cadet|jeune))*\s+)"
    rf"{STAGE_OBJ}{STAGE_VERB}\b")   # « Jean », « Charolais père », « Godefroy et Césarine »
# Indications courtes : « À part. », « Haut. », « Bas, à Gérald. », « Lisant. »
SHORT_DIRECTION = re.compile(r"^(?:à\s+part|a\s+part|haut|bas|tout\s+bas|tout\s+haut|aparté|au\s+public|"
                             r"lisant|riant|seul|seule)\b[^.!?…]{0,60}[.:]$", re.I)
# Coupure de phrase : avant une capitale, ou avant « il / elle » en minuscule (certaines éditions,
# comme le Guignol de Mourguet, écrivent « Et t'as bien fait! ils sortent. »).
SENTENCE_SPLIT = re.compile(rf"(?<=[.!?…])\s+(?=[«{UPPER}]|(?:il|elle|ils|elles)\s)")
# Entrée ou sortie, verbe en tête : « Rentre Titus Andronicus, accompagné de… »
STAGE_ENTRANCE = re.compile(rf"^(?:Entre|Entrent|Rentre|Rentrent|Sort|Sortent|Paraît|Paraissent|Revient|"
                            rf"Reviennent)\s+[{UPPER}]")
# Un personnage collé dans la réplique d'un autre : « … L'abbé regarde. JEAN .—J'ai trop longtemps cru »
GLUED_SPEAKER = re.compile(rf"\b([{UPPER}][{UPPER}'’ -]{{1,40}}?)\s*\.\s*[—–]\s*")
ROMAN = re.compile(r"[IVXLCDM]+")
# Premier tour qui décrit le décor (« Une salle d'audience au Palais de Justice. Au lever du rideau… »)
DECOR = re.compile(r"\b(?:au\s+lever\s+du\s+rideau|décor|le\s+théâtre\s+représente|la\s+scène\s+représente|"
                   r"salon|chambre|salle|à\s+droite|à\s+gauche|au\s+fond|porte|fenêtre|cheminée|fauteuil|"
                   r"table|premier\s+plan)\b", re.I)
# Vers mis bout à bout sur une ligne : capitale de mot-outil en milieu de phrase (« vit le jour Et
# des bras maternels ») ; deux au moins = vers.
MID_CAPITAL = re.compile(r"(?<=[a-zà-ÿ,;])\s+(?:Et|Mais|Je|Tu|Il|Elle|Nous|Vous|Que|Qui|De|Du|La|Le|Les|"
                         r"Un|Une|Ou|Où|Pour|Dans|Sur|Si|Ne|Se|Ce|Mon|Ma|Mes|Son|Sa|Ses|Ton|Ta|À|Au|"
                         r"Aux|En|Quand|Comme)\b")
SONG = re.compile(r"\bAIR\s*:|\bAir\s*:")


def is_stage_sentence(s):
    """Phrase qui a la forme d'une indication de scène (voir STAGE_SENTENCE)."""
    if "?" in s or "!" in s or len(s) > 160 or FIRST_SECOND.search(s):
        return False
    return bool(STAGE_SENTENCE.match(s) or SHORT_DIRECTION.match(s) or STAGE_ENTRANCE.match(s))


def drop_stage_sentences(text):
    """Retire les phrases-didascalies, ligne par ligne (les retours à la ligne sont gardés pour le
    contrôle des vers)."""
    lines = []
    for line in text.split("\n"):
        kept = [s for s in SENTENCE_SPLIT.split(line.strip()) if s and not is_stage_sentence(s)]
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines)


def split_glued_speakers(speaker, text):
    """« Merci. JEAN .—J'ai trop longtemps cru » → [(speaker, « Merci. »), (« JEAN », « J'ai… »)]."""
    parts, pos, who = [], 0, speaker
    for m in GLUED_SPEAKER.finditer(text):
        name = m.group(1).strip()
        if ROMAN.fullmatch(name.replace(" ", "")) or sum(c.isalpha() for c in name) < 2:
            continue                       # « XXII. — » : numéro de chapitre, pas un personnage
        parts.append((who, text[pos:m.start()]))
        who, pos = name, m.end()
    parts.append((who, text[pos:]))
    return [(w, t) for w, t in parts if t.strip()]


def is_verse_flat(flat):
    """Vers recollés sur une ligne, ou couplet chanté (« Air : … »)."""
    return bool(SONG.search(flat)) or len(MID_CAPITAL.findall(flat)) >= 2


def verse_turn_share(scenes):
    """Part des répliques sur plusieurs lignes qui sont en vers (couplets, livrets) ; et leur nombre."""
    multi = [STAGE_DIRECTION.sub(" ", t["text"]) for s in scenes for t in s["turns"]]
    multi = [t for t in multi if len([l for l in t.split("\n") if l.strip()]) >= 2]
    verse = sum(is_verse_turn(t) for t in multi)
    return (verse / len(multi) if multi else 0.0), verse


def scene_to_tirets(scene):
    """Une scène {"turns": [{"speaker": NOM, "text": ...}]} → morceaux de conversation (listes de
    textes de tours ; le locuteur change à chaque tour).

    On garde l'ordre des répliques, on retire les didascalies (entre parenthèses, ou phrases de jeu
    de scène collées), on sépare un personnage collé dans la réplique d'un autre (« JEAN .— »), on
    met chaque réplique sur une ligne, et on FUSIONNE deux répliques consécutives du même
    personnage : le tour change de lettre (A ↔ B) seulement quand le locuteur change. Avec trois
    personnages, A et B ne sont donc pas toujours les mêmes personnes : simplification assumée.
    Une réplique en vers (couplet), pas en français ou de plus de 1 000 caractères COUPE la scène
    en deux morceaux, comme le récit coupe un échange de roman. Un premier tour qui décrit le décor
    est retiré.
    """
    pieces, merged = [], []   # merged = [(personnage, texte)] du morceau en cours
    for i, t in enumerate(scene["turns"]):
        raw = drop_stage_sentences(STAGE_DIRECTION.sub(" ", t["text"]).replace("--", "—"))
        for speaker, text in split_glued_speakers(t["speaker"], raw):
            flat = strip_lead(re.sub(r"\s*\n\s*", " ", text))
            flat = drop_stage_sentences(flat)            # phrases-didascalies révélées par la découpe
            if not re.search(r"[^\W\d_]", flat):
                continue                                 # réplique muette (didascalie seule)
            if (i == 0 and not merged and not re.search(r"[?!]", flat) and DECOR.search(flat)
                    and not FIRST_SECOND.search(flat)):
                continue                                 # description du décor en tête de scène
            if (is_verse_turn(text) or is_verse_flat(flat) or len(flat) > MAX_THEATRE_TURN
                    or language_reason(flat)):
                pieces.append(merged)
                merged = []
                continue
            if merged and merged[-1][0] == speaker:
                merged[-1] = (speaker, merged[-1][1] + " " + flat)
            else:
                merged.append((speaker, flat))
    pieces.append(merged)
    return [[text for _, text in m] for m in pieces if len(m) >= MIN_TURNS]


def leaks(turns, val_prints, val_whole):
    """L'échange recopie la validation : un de ses longs tours y est, ou l'échange entier (même court,
    au texte normalisé près : autre édition du même roman)."""
    return bool(long_turn_prints(turns) & val_prints) or fingerprint(
        "\n".join(t["text"] for t in turns)) in val_whole


def theatre_moderne(path, ref, deduper, val_prints=frozenset(), val_whole=frozenset(), val_docs=frozenset()):
    """theatre_train → (échanges au contrat tirets, pièces écartées par raison, ids écartés,
    ids retenus, compteur des morceaux de scène retirés). Un document de la validation des romans
    (val_docs) est écarté ici aussi : aucun document des deux côtés."""
    plays = defaultdict(list)
    for scene in read_docs(path):
        plays[(scene["doc_id"], scene.get("play"))].append(scene)
    rejected, rejected_ids, kept_ids, removed = Counter(), defaultdict(list), [], Counter()
    out = []
    for (doc_id, play), scenes in plays.items():
        text = "\n".join(t["text"] for s in scenes for t in s["turns"])
        who = normalize(f"{doc_id} {play}")
        share, n_verse = verse_turn_share(scenes)
        reason = None
        if ref["exclusions"].get(doc_id) in EXCLUDED_REASONS:
            reason = f"liste_{ref['exclusions'][doc_id]}"
        elif doc_id in val_docs:
            reason = "document_en_validation"
        elif any(s.get("is_moliere") for s in scenes):
            reason = "moliere"           # Molière a son propre chatbot : on garde les styles séparés
        elif EROTIC_NAMES.search(who):
            reason = "erotique_auteur_ou_titre"
        elif DICTIONARY.search(who):
            reason = "dictionnaire"
        elif is_crude(text):
            reason = "vocabulaire_cru"
        elif erotic_score(text) > EROTIC_MAX:
            reason = "euphemismes_erotiques"
        elif archaic_score(text) > ARCHAIC_MAX:
            reason = "orthographe_ancienne"
        elif is_verse_play(scenes):
            reason = "en_vers"
        elif n_verse >= 2 and share > VERSE_PLAY_SHARE:
            reason = "en_partie_en_vers"   # livrets, opéras-comiques, alexandrins mis bout à bout
        if reason:
            rejected[reason] += 1
            rejected_ids[reason].append(f"{doc_id} | {play}")
            continue
        kept_ids.append(f"{doc_id} | {play}")
        for scene in scenes:
            k = 0
            for piece in scene_to_tirets(scene):
                for chunk in split_long(piece):
                    turns = with_speakers(chunk)
                    if leaks(turns, val_prints, val_whole):
                        removed["fuite_vers_val"] += 1
                    elif deduper.seen(turns):
                        removed["doublon"] += 1
                    else:
                        out.append({"id": f"{scene['id']}:t{k}", "source": scene["source"],
                                    "doc_id": doc_id, "url": scene.get("url"),
                                    "license": scene.get("license"), "style": "moderne", "turns": turns})
                        k += 1
    return out, rejected, rejected_ids, kept_ids, removed


# ---------------------------------------------------------------------------------------------
# 8. Statistiques
# ---------------------------------------------------------------------------------------------
def quantile(values, q):
    return float(np.quantile(values, q)) if values else 0.0


def summary(ex):
    """Ce qu'il faut garder d'un échange pour les statistiques : (document, longueurs des tours)."""
    return ex["doc_id"], [len(t["text"]) for t in ex["turns"]]


def file_stats(summaries):
    """Documents, échanges, tours, caractères, tours par échange et longueur des tours (médiane, p90)."""
    n_turns = [len(lens) for _, lens in summaries]
    lengths = [n for _, lens in summaries for n in lens]
    return {"documents": len({d for d, _ in summaries}), "echanges": len(summaries),
            "tours": sum(n_turns), "caracteres": sum(lengths),
            "tours_par_echange": {"mediane": quantile(n_turns, 0.5), "p90": quantile(n_turns, 0.9)},
            "longueur_tour": {"mediane": quantile(lengths, 0.5), "p90": quantile(lengths, 0.9)}}


def sample_ids(ids, k=20):
    """k ids stables (les plus petites empreintes) : la même liste à chaque lancement."""
    return sorted(set(ids), key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:k]


# ---------------------------------------------------------------------------------------------
# 9. Le tout
# ---------------------------------------------------------------------------------------------
def corpus_files(limit_files=None):
    files = []
    for source in SOURCES:
        found = sorted(glob(str(CORPUS / source / "part-*.jsonl.zst")))
        files += found[:limit_files] if limit_files else found
    return files


def build(files, out, theatre_path=THEATRE_TRAIN, workers=6, ref=None,
          val_per_10000=VAL_PER_10000, min_doc_turns=MIN_DOC_TURNS, log=print):
    """Construit romans_train / romans_val / theatre_moderne_train et stats.json dans `out`."""
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref = ref if ref is not None else build_reference()
    tmp = out / "_romans_train_candidats.jsonl.zst"   # passe 1 : candidats train, sur disque

    rejected, rejected_ids, kept_ids = Counter(), defaultdict(list), []
    cuts = Counter()
    erotic = []          # (score, id, titre) des documents gardés : les 30 plus hauts à relire
    val, val_docs, train_docs = [], set(), set()
    n_candidates = 0
    jobs = [(f, min_doc_turns) for f in files]
    # Passe 1 : extraction en parallèle (un fichier par processus), écriture dans l'ordre des fichiers.
    with zstandard.open(tmp, "wt", encoding="utf-8") as cand:
        if workers > 1:
            pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(ref,))
            results = pool.imap(process_file, jobs)
        else:
            _init_worker(ref)
            pool, results = None, map(process_file, jobs)
        try:
            for i, (path, docs) in enumerate(results, 1):
                for d in docs:
                    cuts.update(d["cuts"])
                    if d["reason"]:
                        rejected[d["reason"]] += 1
                        rejected_ids[d["reason"]].append(d["id"])
                        continue
                    kept_ids.append(d["id"])
                    erotic.append((d["erotic"], d["id"], d.get("title")))
                    to_val = is_val(d["id"], val_per_10000)
                    (val_docs if to_val else train_docs).add(d["id"])
                    for k, texts in enumerate(d["exchanges"]):
                        ex = {"id": f"{d['id']}:{k:05d}", "source": d["source"], "doc_id": d["id"],
                              "url": d["url"], "license": d["license"], "style": "moderne",
                              "turns": with_speakers(texts)}
                        if to_val:
                            val.append(ex)
                        else:
                            cand.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            n_candidates += 1
                log(f"[{i}/{len(files)}] {Path(path).parent.name}/{Path(path).name} : "
                    f"{len(docs)} docs, {n_candidates} échanges train candidats, {len(val)} val "
                    f"({time.time() - t0:.0f} s)")
        finally:
            if pool:
                pool.close()
                pool.join()
    assert not (val_docs & train_docs), "un document est à la fois en train et en val"

    # Passe 2 : validation dédoublonnée d'abord (elle a la priorité), puis train sans doublon
    # ni fuite (un long tour déjà présent en validation = autre édition du même roman).
    removed = Counter()
    vdedup = Deduper()
    val_kept = []
    for ex in val:
        if vdedup.seen(ex["turns"]):
            removed["val_doublon"] += 1
        else:
            val_kept.append(ex)
    val_prints, val_whole = vdedup.turns, vdedup.exchanges
    dedup = Deduper()

    train_kept = []   # résumés seulement (pas les textes) : de quoi faire les statistiques

    def train_stream():
        for ex in read_docs(tmp):
            if leaks(ex["turns"], val_prints, val_whole):
                removed["train_fuite_vers_val"] += 1
            elif dedup.seen(ex["turns"]):
                removed["train_doublon"] += 1
            else:
                train_kept.append(summary(ex))
                yield ex

    write_docs(out / "romans_train.jsonl.zst", train_stream())
    write_docs(out / "romans_val.jsonl.zst", val_kept)
    tmp.unlink()
    log(f"romans : {len(train_kept)} échanges train, {len(val_kept)} val ; retirés {dict(removed)}")

    # Théâtre en prose : dédoublonné contre les romans train (rare, mais des romans dialogués
    # existent aussi en version scène) et contre lui-même.
    th, th_rejected, th_rejected_ids, th_kept, th_removed = ([], Counter(), {}, [], Counter())
    if theatre_path and Path(theatre_path).exists():
        th, th_rejected, th_rejected_ids, th_kept, th_removed = theatre_moderne(theatre_path, ref, dedup,
                                                                                val_prints, val_whole,
                                                                                val_docs)
        write_docs(out / "theatre_moderne_train.jsonl.zst", th)
        log(f"théâtre moderne : {len(th)} échanges ; pièces écartées {dict(th_rejected)}")

    stats = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "tirets",
        "fichiers_corpus": len(files),
        "parametres": {"min_tours_echange": MIN_TURNS, "max_tours_echange": MAX_TURNS,
                       "max_caracteres_tour": MAX_TURN_CHARS, "min_tours_document": min_doc_turns,
                       "validation_pour_10000": val_per_10000, "orthographe_ancienne_max": ARCHAIC_MAX,
                       "score_erotique_max": EROTIC_MAX},
        "fichiers": {"romans_train.jsonl.zst": file_stats(train_kept),
                     "romans_val.jsonl.zst": file_stats([summary(e) for e in val_kept]),
                     "theatre_moderne_train.jsonl.zst": file_stats([summary(e) for e in th])},
        "documents_romans": {"retenus": len(kept_ids), "train": len(train_docs), "val": len(val_docs),
                             "ecartes_par_raison": dict(rejected.most_common())},
        "tours_refuses_par_raison": dict(cuts.most_common()),
        "echanges_retires": dict(removed),
        "theatre_pieces": {"retenues": len(th_kept), "ecartees_par_raison": dict(th_rejected.most_common()),
                           "echanges_retires": dict(th_removed)},
        "a_relire": {"romans_retenus": sample_ids(kept_ids),
                     "romans_score_erotique_top30": [{"id": i, "titre": t, "score": sc}
                                                     for sc, i, t in sorted(erotic, key=lambda x: -x[0])[:30]],
                     "romans_ecartes": {r: sample_ids(ids) for r, ids in rejected_ids.items()},
                     "theatre_retenues": sample_ids(th_kept),
                     "theatre_ecartees": {r: sample_ids(ids) for r, ids in th_rejected_ids.items()}},
        "duree_s": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"terminé en {stats['duree_s']} s → {out}")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description="Dialogues en français moderne (format tirets).")
    p.add_argument("--limit-files", type=int, default=None, help="N premiers fichiers de chaque source")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--workers", type=int, default=6, help="processus (6 au plus)")
    args = p.parse_args(argv)
    workers = max(1, min(6, args.workers))
    build(corpus_files(args.limit_files), args.out, workers=workers,
          log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
