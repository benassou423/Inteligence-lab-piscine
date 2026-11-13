"""Générateur de conversations synthétiques (option, à valider par l'étudiant avant toute génération complète).

    uv run python -m chatbot.synthetic --style moderne --n 60 --out DOSSIER/moderne.jsonl.zst --seed 1
    uv run python -m chatbot.synthetic --style moliere --n 120 --out DOSSIER/moliere.jsonl.zst --seed 1
    uv run python -m chatbot.synthetic --maj-exemples       # choisit les répliques de Molière du prompt
    uv run python -m chatbot.synthetic --lisezmoi DOSSIER   # bilan chiffré + 15 exemples par style
    uv run python -m chatbot.synthetic --refiltrer --style moderne --out F  # filtres actuels sur F

Pourquoi ? Le théâtre et les romans apprennent le STYLE, mais aucune donnée réelle ne montre
« un message d'aujourd'hui -> une réponse à la manière de Molière », ni une conversation courante
d'aujourd'hui (« Salut, ça va ? » / « Ça va, et toi ? »). On les fait écrire par un modèle local à
POIDS OUVERTS déjà installé sur le Mac : qwen3:8b servi par Ollama (API HTTP locale). C'est de
l'inférence, pas un entraînement. Règles suivies (celles du concours pour le synthétique) : modèle
à poids ouverts, prompts publiés (chatbot/prompts_synthetiques.json), part bornée et déclarée.

Chemin d'une donnée :
1. prompt tiré au hasard mais REPRODUCTIBLE : graine + numéro d'appel -> mêmes sujets, même ton,
   même personnage, mêmes exemples, même graine Ollama ;
2. qwen3 écrit k conversations « A: ... / B: ... » séparées par « ### » ;
3. analyse (parse_sortie), puis filtres de qualité (raison_rejet), anti-doublons et anti-tics
   (Dedup), puis, pour le style Molière seulement, relecture du fond par qwen3 lui-même (juger :
   trois questions OUI/NON, prompt publié ; mesuré inutile en style moderne, voir juge_mesure
   dans le JSON) ; chaque rejet est compté par raison dans le rapport JSON, avec 5 exemples ;
4. écriture atomique (fichier temporaire puis os.replace) : on peut couper et relancer avec le
   même --out, la génération reprend au numéro d'appel suivant.

Contrat de sortie (une ligne JSON = une conversation) : id, source, doc_id, url, license, style,
turns [{speaker, text}], plus generator {model, prompt_id, seed, temperature}. Style « moderne » :
speakers A/B en alternance stricte (format « tirets »). Style « moliere » : l'utilisateur est
l'interlocuteur du personnage (LA GRANGE pour MOLIÈRE...), la réponse est dite par le personnage
(format « repliques »).
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from chatbot.format import PERSONAS, clean_text, normalize_speaker, render_scene
from chatbot.generate import WORD, echo_ratio
from chatbot.pretrain import CRUDE
from pipeline.io import read_docs, write_docs

PROMPTS_PATH = Path(__file__).with_name("prompts_synthetiques.json")
SOURCE = "synthetique_qwen3"
LICENSE = "synthétique : généré par qwen3:8b (poids ouverts, Apache-2.0), prompts publiés"
MOLIERE_TRAIN = Path("data/stages/chatbot_20261112/theatre/moliere_train.jsonl.zst")
TOKENIZER = Path("submissions/2026-11-11_153839/tokenizer.json")

# Jurons d'aujourd'hui, en plus de la liste CRUDE de chatbot/pretrain.py (vocabulaire sexuel cru).
# Ici « con » est compté : dans une conversation moderne il n'a pas les sens anciens/étrangers qui
# l'avaient fait exclure de CRUDE.
JURONS = re.compile(r"\b(?:merd\w*|emmerd\w*|connard\w*|conne|cons?|chier|chiant\w*|"
                    r"fout\w*|m'en fou\w*|enfoiré\w*|salaud\w*|niqu\w*|putes?|pétasse\w*)\b")
# Méta-texte : le générateur parle de ce qu'il fait au lieu d'écrire la conversation.
# « Bien sûr » et « Voici » seuls sont du français naturel (« Bien sûr, je viens ! ») : ils ne sont
# du méta-texte que lorsqu'ils annoncent la sortie (« Bien sûr ! Voici trois conversations »).
META = re.compile(r"[*`#<>\[\]{}()]|\bdialogues?\b|\brépliques?\b|\b[AB]\s*:|"
                  r"\bvoici (?:les|des|un|une|deux|trois|quelques) (?:conversations?|échanges?|exemples?)|"
                  r"^bien sûr\W+voici\b|\bconversation\s*\d|\béchange\s*\d", re.IGNORECASE)
# Tutoiement dans la réponse du personnage (voir prompts_synthetiques.json, tutoiement_justification).
# Les étiquettes « A » / « B » employées comme des prénoms (« Salut B, tu as reçu... ») : le modèle
# confond les noms de rôle du prompt avec les personnes. Sensible à la casse : « a » (avoir) passe.
NOM_AB = re.compile(r"(?<![\w'])[AB](?![\w'])")
TUTOIEMENT = re.compile(r"\b(?:tu|te|toi|ton|ta|tes)\b|\bt'", re.IGNORECASE)
# Tutoiement sans pronom : l'impératif singulier (« Attends », « Ne me dis pas », « et demande »).
# On ne le reconnaît qu'en DÉBUT de proposition (début, ponctuation, et/mais/alors...), avec
# « ne » et des pronoms compléments permis devant : « je vous dis » ou « il demande » ne sont pas
# des impératifs, et « dis-je » (inversion) est exclu. Liste courte des verbes les plus fréquents.
IMPERATIFS_TU = ("dis|fais|attends|écoute|demande|regarde|viens|va|sois|prends|crois|laisse|sache|"
                 "tiens|arrête|dépêche|songe|vois|parle|réponds|donne|ôte|cesse|pardonne")
IMPERATIF_TU = re.compile(
    r"(?:^|[.!?;:,—–]\s*|\b(?:et|mais|alors|donc|ou|puis|or)\s+)"  # début de proposition
    r"(?:(?:ne|n')\s*)?(?:(?:me|le|la|les|lui|leur|y|en|nous)\s+|m')*"  # « ne me », « ne les »
    rf"(?:{IMPERATIFS_TU})\b(?!-(?:je|il|elle|on|ils|elles|vous)\b)", re.IGNORECASE)
VOUS = re.compile(r"\bvous\b", re.IGNORECASE)
TROISIEME_PERSONNE = {"il", "elle", "lui", "son", "sa", "ses", "leur"}
# « monsieur » ou « mon ami » dit à une femme, « madame » dit (en apostrophe) à un homme.
VOCATIF_MASCULIN = re.compile(r"\bmonsieur\b|\bmon (?:jeune |cher |bon |pauvre )?ami\b", re.IGNORECASE)
VOCATIF_FEMININ = re.compile(r"(?:^|[,;:.!?]\s*)madame\s*[,;:.!?]|\bma (?:chère |bonne |pauvre )?amie\b",
                             re.IGNORECASE)
# Une ligne de réplique : « A: texte » (on tolère « **A** : », « - A: », espaces).
TURN_LINE = re.compile(r"^\s*[-–—]?\s*\**\s*([AB])\s*\**\s*[:：]\s*\**\s*(.*?)\s*$")
# Lignes de structure qu'on ignore sans pénalité (le modèle numérote parfois ses conversations).
STRUCTURE = re.compile(r"^\s*(?:#+|-{3,}|\**\s*(?:conversation|échange)\s*\d+\s*\**\s*:?)\s*$",
                       re.IGNORECASE)
SEPARATOR = re.compile(r"^\s*(?:#{3,}|-{3,})\s*$")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
OLD_SPELLING = re.compile(r"\b\w+(?:oit|oient|oître|oîtr\w+)\b|\btrès-", re.IGNORECASE)
PATOIS = re.compile(r"\bje \w+ons\b|\bmonsieu\b|\bmontsir\b|\bbene\b|\bdignus\b|\bbalordo\b|"
                    r"\b\w*(?:ét|av|voul|pouv|sav|dis|fais|croy|all|dev|ven|pren|ser|aur|fer)ois\b|"
                    r"\b(?!trois\b)\w{2,}rois\b|\bje voi\b")
MODERN_OIT = {"soit", "doit", "voit", "croit", "reçoit", "perçoit", "aperçoit", "déçoit", "conçoit",
              "boit", "droit", "endroit", "adroit", "étroit", "toit", "exploit", "accroît"}


def load_config(path=PROMPTS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. prompts reproductibles
def construire_appel(style, seed, call, cfg):
    """Prompt n° `call` : ne dépend QUE de (style, graine, numéro) -> reprise et tests exacts."""
    rng = random.Random(f"{style}:{seed}:{call}")
    gab = cfg["gabarits"][style]
    k = cfg["par_appel"][style]
    sujets = rng.sample(cfg["sujets"], k)
    meta = {"prompt_id": f"{gab['id']}#{call:06d}", "call": call, "sujets": sujets,
            "ollama_seed": rng.randrange(2**31)}
    # UN registre (tu/vous, court/long) par appel, rappelé en tête du prompt : dans l'échantillon
    # v1, un ton par conversation glissé dans la liste des sujets était ignoré (0 vouvoiement / 30).
    reg = rng.choice(cfg["registres"])
    meta["registres"] = [reg["id"]] * k
    lignes = [gab["ligne_sujet"].format(i=i + 1, sujet=s["texte"]) for i, s in enumerate(sujets)]
    if style == "moderne":
        prompt = "\n".join(gab["lignes"]).format(k=k, registre=reg["moderne"], sujets="\n".join(lignes))
    else:
        # MOLIÈRE une fois sur deux (demande du référent), sinon un autre personnage de la démo,
        # sauf ceux des pièces gardées pour la mesure (personas_exclus).
        autres = [p for p in PERSONAS[1:] if p["name"] not in cfg.get("personas_exclus", [])]
        p = PERSONAS[0] if rng.random() < 0.5 else rng.choice(autres)
        # Genre de l'interlocuteur : c'est son nom qui étiquettera les messages de A.
        genre = cfg["genres"][cfg["interlocuteurs"][p["interlocutor"]]]
        ex = rng.sample(cfg["exemples_moliere"], min(cfg["nb_exemples_par_prompt"],
                                                     len(cfg["exemples_moliere"])))
        demos = rng.sample(cfg["demonstrations_moliere"], cfg["nb_demonstrations_par_prompt"])
        meta.update(persona=p["name"], interlocutor=p["interlocutor"], persona_label=p["label"])
        prompt = "\n".join(gab["lignes"]).format(
            k=k, persona=p["label"], about=p["about"], registre=reg["moliere"],
            genre=genre["texte"], accord=genre["accord"], titre=genre["titre"],
            exemples="\n".join(gab["ligne_exemple"].format(texte=e["texte"]) for e in ex),
            demonstrations="\n".join(gab["ligne_demonstration"].format(**d) for d in demos),
            sujets="\n".join(lignes))
    return prompt, meta


def contexte(style, meta, j):
    """Ce que le prompt demandait pour la conversation n° j de l'appel (voir raison_rejet)."""
    sujet = meta["sujets"][j]["texte"] if j < len(meta["sujets"]) else None
    ctx = {"registre": meta["registres"][0], "sujet": sujet}
    if style == "moliere":
        ctx.update(persona=meta["persona"], interlocuteur=meta["interlocutor"],
                   persona_label=meta.get("persona_label", meta["persona"]))
    return ctx


def contexte_document(doc, cfg):
    """Même contexte, relu dans une ligne déjà écrite (pour --refiltrer)."""
    s = doc.get("sujet") or {}
    textes = {x["id"]: x["texte"] for x in cfg["sujets"]}
    ctx = {"registre": s.get("registre"), "sujet": textes.get(s.get("id"))}
    if doc["style"] == "moliere":
        label = next((p["label"] for p in PERSONAS if p["name"] == doc.get("persona")), doc.get("persona"))
        ctx.update(persona=doc.get("persona"), interlocuteur=doc["turns"][0]["speaker"], persona_label=label)
    return ctx


# ---------------------------------------------------------------- 2 bis. relecture par qwen3
def prompt_juge(style, tours, ctx, cfg):
    """Prompt publié du juge : la conversation, le contexte, trois questions (OUI = bon)."""
    g = cfg["gabarits"]["juge"]
    if style == "moderne":  # (le registre tu/vous est vérifié par le filtre « registre », pas ici)
        contexte_txt, question = g["contexte_moderne"], g["question_moderne"]
    else:
        contexte_txt = g["contexte_moliere"].format(persona=ctx.get("persona_label", ctx.get("persona")))
        question = g["question_moliere"]
    conversation = "\n".join(f"{s}: {t}" for s, t in tours)
    return "\n".join(g["lignes"]).format(contexte=contexte_txt, conversation=conversation,
                                          question=question)


def est_prompt_juge(prompt, cfg):
    return prompt.startswith(cfg["gabarits"]["juge"]["lignes"][0])


def juger(style, tours, ctx, cfg, generateur):
    """Verdict du juge : ("OUI" | "NON" | None si illisible, texte brut, réponse d'Ollama).

    « OUI » seulement si les trois questions ont reçu OUI ; une réponse sans ses trois lignes
    « 1: OUI » ... est illisible (et donc rejetée par generer).
    """
    j = cfg["juge"]
    rep = generateur(prompt_juge(style, tours, ctx, cfg),
                     {"temperature": j["temperature"], "num_predict": j["num_predict"], "seed": j["seed"]})
    texte = THINK.sub("", rep.get("response", "")).strip()
    reponses = dict(re.findall(r"^\s*([123])\s*[:.)-]\s*(OUI|NON)\b", texte, re.IGNORECASE | re.MULTILINE))
    if set(reponses) != {"1", "2", "3"}:
        return None, texte, rep
    return ("OUI" if all(v.upper() == "OUI" for v in reponses.values()) else "NON"), texte, rep


# ---------------------------------------------------------------- 2. appel au modèle local
def appel_ollama(prompt, options, cfg):
    """Un appel à Ollama (local). Renvoie le JSON de la réponse (texte + compteurs de tokens)."""
    body = json.dumps({"model": cfg["generateur"]["model"], "prompt": prompt, "stream": False,
                       "think": cfg["generateur"]["think"], "options": options}).encode()
    req = urllib.request.Request(cfg["generateur"]["url"], data=body,
                                 headers={"Content-Type": "application/json"})
    for essai in range(3):  # Ollama peut répondre une erreur pendant qu'il charge le modèle
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError):
            if essai == 2:
                raise
            time.sleep(5)


# ---------------------------------------------------------------- 3. analyse de la sortie
def nettoie_tour(text):
    """Un tour = un seul paragraphe, typographie du format (chatbot.format.clean_text)."""
    text = clean_text(text).replace("\n", " ")
    text = re.sub(r"^[-–—]\s*", "", text).strip()
    # Une réplique entièrement entre guillemets : on retire les guillemets (et rien d'autre).
    m = re.fullmatch(r"[«\"“]\s*(.*?)\s*[»\"”]", text)
    return m.group(1) if m else text


def parse_sortie(raw):
    """Texte brut -> (blocs, nb_blocs_sans_tour). Bloc = (tours, lignes_parasites).

    tours = [("A"|"B", texte)] dans l'ordre ; lignes_parasites = lignes qui ne sont ni une réplique
    ni une ligne de structure (titre, numéro) : elles rendent le bloc invalide (méta ou mal formé).
    Un bloc sans aucune réplique (préambule « Voici trois conversations : ») n'est pas une
    conversation : il est seulement compté.
    """
    raw = THINK.sub("", raw)
    blocs, courant, vides = [], [], 0
    for line in raw.split("\n") + ["###"]:
        if SEPARATOR.match(line):
            tours = [t for t in courant if isinstance(t, tuple)]
            if tours:
                blocs.append((tours, [t for t in courant if isinstance(t, str)]))
            elif any(courant):
                vides += 1
            courant = []
            continue
        if not line.strip() or STRUCTURE.match(line):
            continue
        m = TURN_LINE.match(line)
        courant.append((m.group(1), nettoie_tour(m.group(2))) if m else line.strip())
    return blocs, vides


def mots(text):
    return WORD.findall(text.lower())


def tutoie(text):
    """Pronom de 2e personne du singulier OU impératif singulier en début de proposition."""
    return bool(TUTOIEMENT.search(text) or IMPERATIF_TU.search(text))


def ngrammes(text, n):
    ws = mots(text)
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def textes_de_reference(cfg):
    """Ce que le prompt montre à qwen3 et qu'il ne doit pas recopier : les vraies répliques de
    Molière et les échanges de démonstration (message et réponse)."""
    return ([e["texte"] for e in cfg.get("exemples_moliere", [])]
            + [t for d in cfg.get("demonstrations_moliere", []) for t in (d["a"], d["b"])])


def raison_rejet(tours, parasites, style, cfg, ctx=None):
    """Première raison de rejet d'une conversation (None = gardée). Ordre : forme puis contenu.

    ctx (facultatif) : ce que le prompt demandait pour cette conversation, pour vérifier qu'il a
    été suivi : {"registre": "vous_court"..., "interlocuteur": "ELMIRE", "persona": "TARTUFFE",
    "sujet": "A a raté son train"}. Sans ctx, ces vérifications-là sont sautées.
    """
    f = cfg["filtres"]
    ctx = ctx or {}
    if parasites:
        return "meta" if any(META.search(p) for p in parasites) else "mal_formee"
    lo, hi = f["tours_moderne"] if style == "moderne" else f["tours_moliere"]
    speakers = [s for s, _ in tours]
    alterne = all(s == "AB"[i % 2] for i, s in enumerate(speakers))
    if not alterne or not lo <= len(tours) <= hi or (style == "moliere" and len(tours) % 2):
        return "mal_formee"  # moliere : un nombre pair de tours = A commence, B (le personnage) finit
    textes = [t for _, t in tours]
    tout = " ".join(textes)
    if any(c.isalpha() and not unicodedata.name(c, "").startswith("LATIN") for c in tout) \
            or any(ord(c) >= 0x2E80 for c in tout):  # chinois, cyrillique, émoji...
        return "non_latin"
    anglais = set(f["mots_anglais"])
    if any(sum(w in anglais for w in re.findall(r"[a-z']+", t.lower())) >= 2 for t in textes):
        return "anglais"
    if any(META.search(t) or NOM_AB.search(t) for t in textes):
        return "meta"
    if any(not mots(t) for t in textes):
        return "tour_vide"
    if any(len(mots(t)) > f["max_mots_par_tour"] for t in textes):
        return "trop_long"
    if any(echo_ratio(textes[i - 1], textes[i]) >= f["echo_max"] for i in range(1, len(textes))):
        return "echo"
    # L'utilisateur simulé (A) qui paraphrase la réplique d'avant (seuil plus bas) ou en reprend
    # les 3 premiers mots (« C'est comme si... » -> « C'est comme si... »).
    for i in range(2, len(textes), 2):
        if echo_ratio(textes[i - 1], textes[i]) >= f["echo_max_utilisateur"] \
                or mots(textes[i])[:3] == mots(textes[i - 1])[:3]:
            return "echo"
    # Une même réplique (de 4 mots ou plus) deux fois dans la conversation.
    longs = [" ".join(mots(t)) for t in textes if len(mots(t)) >= f["tour_repete_min_mots"]]
    if len(set(longs)) < len(longs):
        return "tour_repete"
    if CRUDE.search(tout.lower()) or JURONS.search(tout.lower()):
        return "vocabulaire_cru"
    if style == "moderne" and ctx.get("registre"):
        # Registre demandé par le prompt : « vous » -> aucun tutoiement ; « tu » -> pas un
        # échange entièrement au « vous » (« vous » seul peut être un pluriel : on le tolère
        # s'il y a aussi du tutoiement).
        if ctx["registre"].startswith("vous") and any(tutoie(t) for t in textes):
            return "registre"
        if ctx["registre"].startswith("tu") and VOUS.search(tout) and not any(tutoie(t) for t in textes):
            return "registre"
    if style == "moliere":
        if ctx.get("persona") in cfg.get("personas_exclus", []):
            return "persona_exclu"
        reponses = [t for s, t in tours if s == "B"]  # seulement le personnage
        messages = [t for s, t in tours if s == "A"]
        if any(tutoie(t) for t in reponses):
            return "tutoiement"
        modernes = [re.escape(m) for m in f["mots_modernes_reponse_moliere"]]
        pat = re.compile(r"\b(?:" + "|".join(modernes) + r")\b", re.IGNORECASE)
        if any(pat.search(t) for t in reponses):
            return "mot_moderne"
        # Le personnage recopie une réplique d'exemple du prompt (6 mots de suite en commun).
        n = f["copie_exemple_ngramme"]
        ref = set().union(*(ngrammes(t, n) for t in textes_de_reference(cfg)))
        if any(ngrammes(t, n) & ref for t in textes):
            return "copie_exemple"
        # A recopie la description du sujet (« un plat qu'il n'aime pas ») au lieu de parler en « je ».
        # Seules comptent les suites qui contiennent un mot de 3e personne : « à acheter une voiture
        # d'occasion » est une reprise normale, « un plat qu'il n'aime » est la description recopiée.
        if ctx.get("sujet"):
            sujet3 = {g for g in ngrammes(ctx["sujet"], 5) if set(g) & TROISIEME_PERSONNE}
            if any(ngrammes(t, 5) & sujet3 for t in messages):
                return "recopie_sujet"
        genre = cfg.get("interlocuteurs", {}).get(ctx.get("interlocuteur"))
        if genre == "f" and any(VOCATIF_MASCULIN.search(t) for t in reponses):
            return "genre"
        if genre == "m" and any(VOCATIF_FEMININ.search(t) for t in reponses):
            return "genre"
        if ctx.get("registre", "").startswith("vous") and any(tutoie(t) for t in messages):
            return "registre"
    return None


def ouverture(text):
    """Comment commence une réponse : son premier mot (« ah », « c » pour « c'est »), ou
    « <question courte> » si elle s'ouvre sur une question de 4 mots au plus (« Fatigué ? »)."""
    m = re.match(r"([^.!?]*)\?", text)
    if m and len(mots(m.group(1))) <= 4:
        return "<question courte>"
    return mots(text)[0]


class Dedup:
    """Ce qui dépend des conversations DÉJÀ gardées : doublons exacts, formules toutes faites
    (même suite de 5 mots dans trop de conversations), réplique déjà vue ailleurs, et tics
    d'ouverture (trop de réponses du personnage qui commencent par le même mot, « Ah »)."""

    def __init__(self, max_convs, pour_mille=0, ouverture_max=None, echauffement=20, min_mots=6,
                 style="moderne", tics=()):
        self.max_convs, self.pour_mille = max_convs, pour_mille
        # Tics plafonnés (« c'est comme ») : même plafond que les ouvertures, compté par réponse.
        self.tics = [re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE) for t in tics]
        self.ouverture_max, self.echauffement = ouverture_max, echauffement
        self.min_mots, self.style = min_mots, style
        self.exacts, self.tours = set(), set()
        self.ngrams = Counter()     # 5-gramme -> nombre de conversations GARDÉES qui le contiennent
        self.ouvertures = Counter()  # premier mot des réponses du personnage gardées
        self.nb_reponses = 0

    @classmethod
    def depuis(cls, cfg, style):
        f = cfg["filtres"]
        return cls(f["max_conversations_par_5gramme"], f.get("formules_pour_mille", 0),
                   (f.get("ouverture_max") or {}).get(style), f.get("ouverture_echauffement", 20),
                   f.get("tour_repete_min_mots", 6), style, (f.get("tics_plafonnes") or {}).get(style, []))

    def cles(self, textes):
        ws = [mots(t) for t in textes]
        exact = " | ".join(" ".join(w) for w in ws)
        grams = {tuple(w[i:i + 5]) for w in ws for i in range(len(w) - 4)}  # pas à cheval sur 2 tours
        tours = {" ".join(w) for w in ws if len(w) >= self.min_mots}
        reponses = textes[1::2] if self.style == "moliere" else []  # B = le personnage
        ouv = Counter(ouverture(t) for t in reponses if mots(t))
        # les tics vont dans le même compteur, sous une clé à part (« tic:c'est comme »)
        ouv.update(f"tic:{p.pattern}" for t in reponses for p in self.tics if p.search(t))
        return exact, grams, tours, ouv

    def raison(self, textes):
        exact, grams, tours, ouv = self.cles(textes)
        if exact in self.exacts:
            return "doublon_exact"
        # Seuil des formules : 3, ou 3 pour 1 000 conversations gardées quand le fichier grandit.
        limite = max(self.max_convs, self.pour_mille * len(self.exacts) / 1000)
        if any(self.ngrams[g] >= limite for g in grams):
            return "quasi_doublon"
        if tours & self.tours:
            return "tour_repete"
        if self.ouverture_max:
            nouvelles = sum(v for k, v in ouv.items() if not k.startswith("tic:"))
            total = max(self.nb_reponses + nouvelles, self.echauffement)
            trop = [w for w, c in ouv.items() if self.ouvertures[w] + c > self.ouverture_max * total]
            if trop:
                return "tic_repete" if all(w.startswith("tic:") for w in trop) else "ouverture_repetee"
        return None

    def ajoute(self, textes):
        exact, grams, tours, ouv = self.cles(textes)
        self.exacts.add(exact)
        self.ngrams.update(grams)
        self.tours |= tours
        self.ouvertures.update(ouv)
        self.nb_reponses += sum(v for k, v in ouv.items() if not k.startswith("tic:"))


def vers_document(style, seed, meta, j, tours, cfg, temperature):
    """Conversation gardée -> ligne du contrat commun."""
    if style == "moderne":
        turns = [{"speaker": s, "text": t} for s, t in tours]
    else:  # l'utilisateur (A) = l'interlocuteur du personnage ; la réponse (B) = le personnage
        noms = {"A": normalize_speaker(meta["interlocutor"]), "B": normalize_speaker(meta["persona"])}
        turns = [{"speaker": noms[s], "text": t} for s, t in tours]
    sujet = meta["sujets"][j] if j < len(meta["sujets"]) else None
    doc = {"id": f"{SOURCE}:{style}:s{seed}:{meta['call']:06d}:{j}", "source": SOURCE,
           "doc_id": f"{meta['prompt_id']}/{j}", "url": None, "license": LICENSE, "style": style,
           "turns": turns,
           "generator": {"model": cfg["generateur"]["model"], "prompt_id": meta["prompt_id"],
                         "seed": meta["ollama_seed"], "temperature": temperature},
           "sujet": {"id": sujet["id"], "theme": sujet["theme"], "registre": meta["registres"][j]}
           if sujet else None}
    if style == "moliere":
        doc["persona"] = normalize_speaker(meta["persona"])
    return doc


# ---------------------------------------------------------------- 4. boucle, reprise, journal
def chemins(out):
    out = Path(out)
    base = out.name[:-len(".jsonl.zst")] if out.name.endswith(".jsonl.zst") else out.stem
    return out, out.with_name(base + ".rapport.json"), out.with_name(base + ".journal.log")


def ecrit_atomique(out, rapport_path, docs, rapport):
    """Fichier temporaire puis os.replace : une coupure ne laisse jamais un fichier à moitié écrit."""
    write_docs(str(out) + ".tmp", docs)
    with open(str(rapport_path) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=1)
    os.replace(str(out) + ".tmp", out)
    os.replace(str(rapport_path) + ".tmp", rapport_path)


def generer(style, n, out, seed, generateur=None, cfg=None, max_appels=None, save_every=5,
            log=print, juge=None):
    """Génère jusqu'à n conversations gardées dans `out` (reprise si `out` existe déjà).

    juge : None = suivre cfg["juge"]["actif"][style] ; True/False pour forcer. Le juge passe par le même
    `generateur` (c'est le même qwen3) ; il ne voit que les conversations qui ont passé les filtres.
    """
    cfg = cfg or load_config()
    generateur = generateur or (lambda prompt, options: appel_ollama(prompt, options, cfg))
    if juge is None:  # actif par style (prompts_synthetiques.json : juge.actif, juge_mesure)
        juge = bool(cfg.get("juge", {}).get("actif", {}).get(style, False))
    out, rapport_path, journal_path = chemins(out)
    temperature = cfg["generateur"]["temperature"]
    docs, dedup = [], Dedup.depuis(cfg, style)
    rapport = {"style": style, "seed": seed, "model": cfg["generateur"]["model"],
               "gabarit": cfg["gabarits"][style]["id"], "temperature": temperature,
               "juge": cfg["gabarits"]["juge"]["id"] if juge else None,
               "appels": 0, "next_call": 0, "candidats": 0, "gardes": 0, "rejets": {},
               "blocs_sans_tour": 0, "tokens_prompt": 0, "tokens_generes": 0,
               "duree_generation_s": 0.0, "appels_juge": 0, "duree_juge_s": 0.0,
               "juge_rejets_exemples": [], "rejets_exemples": {}}
    if out.exists():  # reprise : on relit tout pour reconstruire l'anti-doublon et le compteur
        docs = list(read_docs(out))
        for d in docs:
            dedup.ajoute([t["text"] for t in d["turns"]])
        if rapport_path.exists():
            with open(rapport_path, encoding="utf-8") as f:
                ancien = json.load(f)
            if (ancien["style"], ancien["seed"]) != (style, seed):
                raise SystemExit(f"{out} a été généré avec style={ancien['style']} "
                                 f"seed={ancien['seed']} : même --style et --seed pour reprendre")
            rapport.update(ancien)
        last = max((int(d["generator"]["prompt_id"].split("#")[1]) for d in docs), default=-1)
        rapport["next_call"] = max(rapport["next_call"], last + 1)
        log(f"reprise : {len(docs)} conversations déjà gardées, appel suivant n° {rapport['next_call']}")
    max_appels = max_appels or 10 * n // cfg["par_appel"][style] + 10  # garde-fou anti-boucle
    debut_appels = rapport["appels"]
    journal = open(journal_path, "a", encoding="utf-8")
    try:
        while len(docs) < n and rapport["appels"] - debut_appels < max_appels:
            call = rapport["next_call"]
            prompt, meta = construire_appel(style, seed, call, cfg)
            options = {"temperature": temperature, "num_predict": cfg["num_predict"][style],
                       "seed": meta["ollama_seed"]}
            t0 = time.time()
            rep = generateur(prompt, options)
            duree = time.time() - t0
            blocs, vides = parse_sortie(rep.get("response", ""))
            rapport["blocs_sans_tour"] += vides
            duree_juge = 0.0
            for j, (tours, parasites) in enumerate(blocs):
                if len(docs) >= n:
                    break
                rapport["candidats"] += 1
                textes = [t for _, t in tours]
                ctx = contexte(style, meta, j)
                raison = raison_rejet(tours, parasites, style, cfg, ctx) or dedup.raison(textes)
                if not raison and juge:
                    t1 = time.time()
                    verdict, texte, _ = juger(style, tours, ctx, cfg, generateur)
                    duree_juge += time.time() - t1
                    rapport["appels_juge"] += 1
                    if verdict != "OUI":
                        raison = "juge" if verdict == "NON" else "juge_illisible"
                        if len(rapport["juge_rejets_exemples"]) < 40:  # pour relire le juge
                            rapport["juge_rejets_exemples"].append(
                                {"call": call, "j": j, "verdict": texte[:200],
                                 "turns": [f"{s}: {t}" for s, t in tours]})
                if raison:
                    rapport["rejets"][raison] = rapport["rejets"].get(raison, 0) + 1
                    # 5 rejets gardés par raison, pour relire ce que chaque filtre écarte
                    ex = rapport.setdefault("rejets_exemples", {}).setdefault(raison, [])
                    if len(ex) < 5:
                        ex.append({"call": call, "j": j, "turns": [f"{s}: {t}" for s, t in tours]})
                    continue
                dedup.ajoute(textes)
                docs.append(vers_document(style, seed, meta, j, tours, cfg, temperature))
            if len(blocs) != cfg["par_appel"][style]:  # le modèle n'a pas rendu k conversations
                rapport["rejets"]["nombre_de_blocs"] = rapport["rejets"].get("nombre_de_blocs", 0) + 1
            rapport["appels"] += 1
            rapport["next_call"] = call + 1
            rapport["gardes"] = len(docs)
            rapport["tokens_prompt"] += rep.get("prompt_eval_count", 0)
            rapport["tokens_generes"] += rep.get("eval_count", 0)
            rapport["duree_generation_s"] = round(rapport["duree_generation_s"] + duree, 2)
            rapport["duree_juge_s"] = round(rapport["duree_juge_s"] + duree_juge, 2)
            # Débit : tokens produits par seconde (compteurs d'Ollama) ; reste estimé au rythme moyen.
            tok_s = rep.get("eval_count", 0) / max(rep.get("eval_duration", 0) / 1e9, 1e-9)
            par_s = len(docs) / max(rapport["duree_generation_s"] + rapport["duree_juge_s"], 1e-9)
            reste = (n - len(docs)) / par_s if par_s else float("nan")
            ligne = (f"appel {call} : {duree:.1f} s, {tok_s:.1f} tokens/s, juge {duree_juge:.1f} s, "
                     f"{len(blocs)} blocs, gardées {len(docs)}/{n}, reste estimé {reste / 60:.1f} min")
            log(ligne)
            journal.write(time.strftime("%Y-%m-%d %H:%M:%S ") + ligne + "\n")
            journal.flush()
            if rapport["appels"] % save_every == 0:
                ecrit_atomique(out, rapport_path, docs, rapport)
    finally:
        journal.close()
    rapport["taux_rejet"] = round(1 - len(docs) / rapport["candidats"], 4) if rapport["candidats"] else None
    rapport["mots_gardes"] = sum(len(mots(t["text"])) for d in docs for t in d["turns"])
    ecrit_atomique(out, rapport_path, docs, rapport)
    return docs, rapport


def refiltrer(style, out, cfg=None):
    """Réapplique les filtres ACTUELS à un fichier déjà généré, sans rien régénérer.

    Utile quand on ajoute un filtre après coup : les conversations écartées sont comptées dans
    rapport["refiltrage"] (et ajoutées aux rejets), les autres sont réécrites dans le même ordre.
    """
    cfg = cfg or load_config()
    out, rapport_path, _ = chemins(out)
    rapport = json.loads(rapport_path.read_text(encoding="utf-8"))
    dedup, gardes, ecartes = Dedup.depuis(cfg, style), [], Counter()
    for d in read_docs(out):  # (le juge n'est pas rejoué : il faudrait Ollama)
        tours = [("AB"[i % 2], t["text"]) for i, t in enumerate(d["turns"])]
        textes = [t for _, t in tours]
        raison = raison_rejet(tours, [], style, cfg, contexte_document(d, cfg)) or dedup.raison(textes)
        if raison:
            ecartes[raison] += 1
            continue
        dedup.ajoute(textes)
        gardes.append(d)
    for k, v in ecartes.items():
        rapport["rejets"][k] = rapport["rejets"].get(k, 0) + v
        rapport.setdefault("refiltrage", {})[k] = rapport.get("refiltrage", {}).get(k, 0) + v
    rapport["gardes"] = len(gardes)
    rapport["taux_rejet"] = round(1 - len(gardes) / rapport["candidats"], 4)
    rapport["mots_gardes"] = sum(len(mots(t["text"])) for d in gardes for t in d["turns"])
    ecrit_atomique(out, rapport_path, gardes, rapport)
    return gardes, rapport


# ---------------------------------------------------------------- 5. exemples de style Molière
def choisir_exemples_moliere(path=MOLIERE_TRAIN, n=72, seed=2026, max_par_piece=3):
    """Vraies répliques courtes de Molière pour donner le ton (lecture seule de moliere_train).

    Critères : 8 à 25 mots, ponctuation finale, pas de chiffre, pas de tutoiement, pas de vocabulaire cru, orthographe
    moderne (« étoit » écarté), et pas de nom propre : aucune majuscule en milieu de vers ou de
    phrase, sauf Monsieur/Madame/Mademoiselle/Dieu/Ciel. Au plus 3 répliques par pièce (variété).
    """
    permis = {"Monsieur", "Madame", "Mademoiselle", "Dieu", "Ciel", "Messieurs", "Mesdames"}
    titres = {"monsieur", "madame", "mademoiselle", "maître", "seigneur", "dom", "don", "les",
              "des", "une", "deux", "trois", "quatre", "premier", "second", "autre", "père", "mère"}
    scenes = list(read_docs(path))
    # Mots des noms de rôle (« monsieur jourdain » -> « jourdain ») : interdits partout, même en
    # début de vers où la majuscule ne prouve rien (« Ergaste, que dis-tu... »).
    noms = {w for d in scenes for t in d["turns"] for w in mots(t["speaker"])
            if len(w) >= 3 and w not in titres} | {"molière"}
    candidats = []
    for d in scenes:
        for t in d["turns"]:
            text = clean_text(t["text"])
            lignes = text.split("\n")
            flat = " ".join(lignes)
            ws = mots(flat)
            if not 8 <= len(ws) <= 25 or not re.search(r"[.!?]$", flat) or set(ws) & noms:
                continue
            # Chiffres, guillemets (citation coupée), parenthèses (didascalie), réplique interrompue.
            if re.search(r"\d|[«»()]|\.\.\.$|…$", flat) or CRUDE.search(flat.lower()):
                continue
            # Tutoiement (pronom ou impératif) : le prompt demande au personnage de vouvoyer, et
            # le filtre rejette tout tutoiement ; un exemple qui tutoie pousserait vers le rejet.
            if tutoie(flat):
                continue
            # Didascalie restée dans le texte (« A part, après... », « Il sort. »), patois de
            # Pourceaugnac (« tout cho que je di », « chest mi »), sabir du Bourgeois gentilhomme
            # (« questa star ») et « connois » : mauvais modèles de ton.
            if re.search(r"(?:^|[.!?]\s)(?:[AÀ] part\b|(?:Il|Elle) sort\b)|\bje di\b|\bcho\b|"
                         r"\bchest\b|\bquesta\b|\bstar\b|\bconnoi", flat):
                continue
            # Patois paysan (« je venons »), latin de cérémonie, imparfaits en -ois (« j'étois »)
            # et refrains (« la, la, la ») : mauvais modèles de ton pour qwen3.
            # « Ah ! » en tête : le tic que le prompt interdit au personnage (37 % des réponses v1).
            if PATOIS.search(flat.lower()) or len(set(ws)) < 0.6 * len(ws) or ws[0] in ("ah", "ha"):
                continue
            if any(w.lower() not in MODERN_OIT for w in OLD_SPELLING.findall(flat)) \
                    or "très-" in flat.lower():
                continue
            propre = False  # un mot à majuscule qui n'ouvre ni une ligne (vers) ni une phrase
            for ligne in lignes:
                toks = ligne.split()
                for i, w in enumerate(toks[1:], 1):
                    nu = w.strip(",;:.!?'\"«»()")
                    if nu[:1].isupper() and not re.search(r"[.!?]$", toks[i - 1]) and nu not in permis:
                        propre = True
            if not propre:
                candidats.append({"texte": flat, "piece": d.get("play") or d.get("doc_id")})
    rng = random.Random(seed)
    rng.shuffle(candidats)
    choisis, par_piece = [], Counter()
    for c in candidats:
        if par_piece[c["piece"]] < max_par_piece:
            choisis.append(c)
            par_piece[c["piece"]] += 1
        if len(choisis) == n:
            break
    return choisis


def maj_exemples(config_path=PROMPTS_PATH, **kw):
    cfg = load_config(config_path)
    cfg["exemples_moliere"] = choisir_exemples_moliere(**kw)
    tmp = str(config_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, config_path)
    return cfg["exemples_moliere"]


# ---------------------------------------------------------------- 6. bilan de l'échantillon
def rendu(doc):
    """Texte tel que le modèle le lira : « — texte » (tirets) ou « NOM.\\ntexte » (répliques)."""
    if doc["style"] == "moderne":
        return "".join(f"— {t['text']}\n\n" for t in doc["turns"])
    return render_scene(doc["turns"])


def lisezmoi(dossier, n_exemples=15, seed=7, cible=20000):
    """Écrit DOSSIER/LISEZMOI.md : chiffres de l'échantillon et exemples tirés au hasard."""
    from ftb.tokenizer import load_tokenizer
    # check_texts=() : la vérification par défaut lirait data/validation.jsonl, réservée à la mesure.
    tok = load_tokenizer(TOKENIZER, check_texts=())
    dossier = Path(dossier)
    parts = ["# Échantillon de données synthétiques (qwen3:8b, local)\n",
             ("Généré par `chatbot/synthetic.py` ; prompts publiés dans `chatbot/prompts_synthetiques.json`. "
              "Inférence locale uniquement (aucun entraînement). Exemples tirés AU HASARD "
              f"(graine {seed}), pas choisis parmi les meilleurs.\n"),
             ("| Style | Gardées | Candidates | Rejet | Appels | Durée totale | Appel médian | "
              "Tokens qwen3 / s (médiane) | Tokens gardés (notre BPE) | Tokens gardés / heure "
              "(médiane) | 20 000 conversations (total ; médiane) |"),
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    details, exemples = [], []
    for style in ("moderne", "moliere"):
        out, rapport_path, journal_path = chemins(dossier / f"{style}.jsonl.zst")
        if not out.exists():
            continue
        docs = list(read_docs(out))
        r = json.loads(rapport_path.read_text(encoding="utf-8"))
        # Journal : « appel N : 20.4 s, 20.5 tokens/s, ... ». La MÉDIANE écarte les appels ralentis
        # par d'autres calculs sur le Mac (un appel a pris 411 s au lieu de ~25 s pendant la mesure).
        # Temps d'un appel = génération + relecture par le juge de ses conversations.
        lignes = re.findall(r"appel \d+ : ([\d.]+) s, ([\d.]+) tokens/s(?:, juge ([\d.]+) s)?",
                            journal_path.read_text())
        med_s = statistics.median(float(x) + float(z or 0) for x, _, z in lignes)
        med_tok = statistics.median(float(y) for _, y, _ in lignes)
        ntok = sum(len(tok.encode(rendu(d))) for d in docs)
        par_appel = len(docs) / r["appels"]  # conversations gardées par appel, rejets compris
        h_med = r["appels"] * med_s / 3600   # durée si chaque appel avait pris le temps médian
        t_tot = cible / len(docs) * (r["duree_generation_s"] + r.get("duree_juge_s", 0)) / 3600
        t_med = cible / par_appel * med_s / 3600
        parts.append(f"| {style} | {len(docs)} | {r['candidats']} | {100 * (r['taux_rejet'] or 0):.1f} % | "
                     f"{r['appels']} | {(r['duree_generation_s'] + r.get('duree_juge_s', 0)) / 60:.1f} min | {med_s:.1f} s | "
                     f"{med_tok:.1f} | {ntok} | {ntok / h_med:,.0f} | {t_tot:.0f} h ; {t_med:.0f} h |"
                     .replace(",", " "))
        rej = ", ".join(f"{k} {v}" for k, v in sorted(r["rejets"].items(), key=lambda x: -x[1])) or "aucun"
        details.append(f"- **{style}** : rejets par raison : {rej} (dont refiltrage après ajout de "
                       f"filtres : {r.get('refiltrage', {})}) ; blocs sans réplique (préambule) : "
                       f"{r['blocs_sans_tour']} ; juge : {r.get('appels_juge', 0)} relectures, "
                       f"{r.get('duree_juge_s', 0) / 60:.1f} min ; tokens générés par qwen3 : {r['tokens_generes']} ; "
                       f"tokens de prompt : {r['tokens_prompt']} ; {par_appel:.2f} conversation(s) "
                       "gardée(s) par appel.")
        rng = random.Random(seed)
        tirage = rng.sample(docs, min(n_exemples, len(docs)))
        exemples.append(f"\n## {n_exemples} exemples « {style} » tirés au hasard\n")
        for i, d in enumerate(tirage, 1):
            s = d.get("sujet") or {}
            entete = f"### {i}. {d['id']} — sujet {s.get('id')} ({s.get('theme')}, {s.get('registre')})"
            exemples.append(entete + "\n\n```\n" + rendu(d).rstrip() + "\n```\n")
    parts.append("")
    parts += details
    parts.append(f"\nTemps pour {cible} conversations, un appel à la fois : « total » = au rythme "
                 "réellement mesuré (Mac partagé avec d'autres calculs pendant la mesure) ; « médiane » "
                 "= si chaque appel prenait le temps d'un appel médian. Dans les deux cas SANS compter "
                 "le taux de quasi-doublons, qui MONTE avec la taille du fichier.\n")
    # Relecture humaine/IA de la qualité, écrite à part (QUALITE.md) pour survivre à une régénération.
    if (dossier / "QUALITE.md").exists():
        parts.append((dossier / "QUALITE.md").read_text(encoding="utf-8"))
    parts += exemples
    (dossier / "LISEZMOI.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--style", choices=["moderne", "moliere"])
    ap.add_argument("--n", type=int, help="nombre de conversations gardées visé")
    ap.add_argument("--out", help="fichier .jsonl.zst (relancer avec le même --out reprend)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-appels", type=int, default=None)
    ap.add_argument("--maj-exemples", action="store_true",
                    help="choisit les répliques de Molière du prompt et met à jour le JSON publié")
    ap.add_argument("--lisezmoi", metavar="DOSSIER", help="écrit le bilan de l'échantillon")
    ap.add_argument("--max-formule", type=int, default=None,
                    help="nombre max de conversations par 5-gramme (défaut : celui du JSON, 3) ; "
                         "à relever pour une génération de plusieurs milliers de conversations")
    ap.add_argument("--sans-juge", action="store_true",
                    help="ne pas faire relire chaque conversation gardée par qwen3 (juge OUI/NON)")
    ap.add_argument("--refiltrer", action="store_true",
                    help="réapplique les filtres actuels à --out (avec --style), sans régénérer")
    args = ap.parse_args(argv)
    if args.maj_exemples:
        ex = maj_exemples()
        print(f"{len(ex)} répliques de Molière écrites dans {PROMPTS_PATH}")
        return
    if args.lisezmoi:
        lisezmoi(args.lisezmoi)
        print(f"écrit : {Path(args.lisezmoi) / 'LISEZMOI.md'}")
        return
    if args.refiltrer:
        if not (args.style and args.out):
            ap.error("--refiltrer demande --style et --out")
        _, rapport = refiltrer(args.style, args.out)
        print(json.dumps(rapport, ensure_ascii=False, indent=1))
        return
    if not (args.style and args.n and args.out):
        ap.error("--style, --n et --out sont requis")
    cfg = load_config()
    if args.max_formule:
        cfg["filtres"]["max_conversations_par_5gramme"] = args.max_formule
    _, rapport = generer(args.style, args.n, args.out, args.seed, cfg=cfg, max_appels=args.max_appels,
                         log=lambda s: print(s, file=sys.stderr, flush=True),
                         juge=False if args.sans_juge else None)
    print(json.dumps(rapport, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
