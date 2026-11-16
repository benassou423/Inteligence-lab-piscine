"""Compose le fichier de consignes « fiches » de la campagne de synthèse (campagnes v4 et v5).

    PYTHONUTF8=1 .venv/Scripts/python.exe tools/composer_consignes_v4.py \
        --modele ETIQUETTE_OLLAMA --licence "Apache-2.0" [--sortie chatbot/prompts_synthetiques_v4.json]
        [--fiches 9000] [--graine 20261114] [--axes chatbot/axes_consignes_v4.json]
        [--origine chatbot/prompts_synthetiques.json]

Campagne v4 : les valeurs par défaut ci-dessus (9000 fiches, graine 20261114).
Campagne v5 (15/11/2026) : mêmes options, avec les axes de la v5 :
    --axes chatbot/axes_consignes_v5.json --sortie chatbot/prompts_synthetiques_v5.json
    --fiches 26000 --graine 20261115
Le fichier publié chatbot/prompts_synthetiques_v5.json garde, dans « composition », ces valeurs.
Les identifiants de fiche commencent par « v4- » dans les deux campagnes, v5 comprise.

Les deux fichiers de consignes publiés sortent de ce programme, octet pour octet, avec les mêmes
entrées et les mêmes arguments. Pour cette raison, n'y modifier que ce docstring : composer_fiches
et composer produisent le contenu des fichiers publiés.

Ce script n'écrit AUCUNE conversation : il croise des listes de consignes et recopie des clés.

Entrées (lues, jamais modifiées) :
- chatbot/axes_consignes_v4.json (v4), ou chatbot/axes_consignes_v5.json avec --axes (v5) : les axes de la campagne (thèmes, types de message et façons de
  répondre appariées, formes, déroulés, longueurs, suppléments, voix, gabarit du prompt, gabarit du
  juge), rédigés par des agents IA comme consignes ;
- chatbot/prompts_synthetiques.json : le fichier de consignes d'origine, dont on HÉRITE tout ce qui
  ne change pas (interlocuteurs, genres, mots_anglais, mots_modernes_reponse_moliere,
  personas_exclus, la configuration du juge, le style « moderne », les vraies répliques de Molière).

Sortie : un fichier de consignes complet au schéma lu par chatbot/synthetic.py (load_config).
- « sujets » = les fiches d'échange, une par conversation. Chaque fiche croise un thème, un type
  de message compatible avec ce thème, UNE façon de répondre appariée à ce type (la réponse est
  donc en rapport direct avec le message), une forme de premier message, un déroulé, une longueur
  de réponse et AU PLUS UN supplément (détail à glisser, début de la première réponse, interdit
  côté réponses ou interdit côté messages), choisi parmi ceux qui ne contredisent pas la fiche.
  L'id de la fiche code les valeurs tirées :
  v4-<n°>-t<thème>-<type><n° de la façon de répondre>-<forme>-<déroulé>-<longueur>-<supplément>
  (supplément : d1 = détail, e… = début imposé, r… = interdit côté réponses, m… = interdit côté
  messages, x0 = aucun ; un type dont l'id commence par « x » appartient au bloc « essentiels »).
- « registres » = les voix de la personne. « demonstrations_moliere » est vide : le code ne tire
  plus ni démonstration ni paire d'interdits, tout ce qui varie est dans la fiche.
- Tirage ÉQUILIBRÉ et déterministe : une part fixée des fiches va au bloc « essentiels » ; dans
  chaque population (essentiels, autres) les thèmes sont parcourus par tours complets mélangés
  (chaque note revient autant que les autres, à une fiche près) ; pour chaque autre axe, un
  « tourniquet » rend, parmi les valeurs compatibles, celle qui a été la moins servie au regard de
  son poids.
- L'id du gabarit finit par 8 caractères d'empreinte de TOUT le fichier composé : le prompt_id de
  chaque conversation et le rapport de chaque fichier portent ainsi l'identité des consignes.
- Même graine, mêmes fichiers d'entrée, mêmes arguments -> même sortie, octet pour octet (aucune
  date du jour, aucun ordre d'ensemble ou de dictionnaire non fixé).

Le modèle et sa licence sont des arguments : rien n'est téléchargé ni lancé ici.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
AXES = RACINE / "chatbot" / "axes_consignes_v4.json"
ORIGINE = RACINE / "chatbot" / "prompts_synthetiques.json"
SORTIE = RACINE / "chatbot" / "prompts_synthetiques_v4.json"
GRAINE = 20261114
URL_OLLAMA = "http://127.0.0.1:11434/api/generate"

MOT = re.compile(r"\w+")  # même découpage que chatbot.generate.WORD (l'apostrophe et le tiret coupent)
TROISIEME_PERSONNE = {"il", "elle", "lui", "son", "sa", "ses", "leur"}  # filtre « recopie_sujet »
TUTOIEMENT = {"tu", "te", "toi", "ton", "ta", "tes"}                     # filtres « tutoiement » / « registre »
# Ce que le filtre « meta » punit si le modèle le recopie : signes, mots, lettre d'étiquette isolée.
SIGNES_INTERDITS = re.compile(r"[*`#<>\[\]{}()]|\bdialogues?\b|\brépliques?\b", re.IGNORECASE)
LETTRE_ISOLEE = re.compile(r"(?<![\w'])[AB](?![\w'])")
MAX_MOTS_PAR_NOTE = 4
NOMS_DU_GABARIT = dict(k=1, persona="P", about="D", registre="V", genre="G", accord="C", titre="T",
                       exemples="E", demonstrations="I", sujets="F")
SUPPLEMENTS = ("detail", "entree", "interdit_reponse", "interdit_message", "aucun")
SANS_SUPPLEMENT = "x0"
# Garde-fou : rien, dans les axes, ne doit évoquer les trois pièces tenues à l'écart pour la mesure.
A_NE_PAS_EVOQUER = re.compile(r"\b(?:misanthrope|sicilien|avares?|avarice|harpagon|alceste|célimène|cassette)\b",
                              re.IGNORECASE)

# ---- vraies répliques de Molière : on ne garde que les « propres » (règles publiées ici) ----------
PIECES_A_L_ECART = ("misanthrope", "sicilien", "avare")  # pièces gardées pour la mesure
MAJUSCULES_PERMISES = {"Monsieur", "Madame", "Mademoiselle", "Messieurs", "Mesdames", "Dieu", "Ciel"}
DIDASCALIE = re.compile(r"\.\.\.\s+(?:Il|Elle)\s|(?:^|[.!?]\s)(?:[AÀ] part\b|(?:Il|Elle) (?:sort|entre|rit|lui parle)\b)")
GRAPHIE_ANCIENNE = re.compile(
    r"\bfoibl\w*|\w+aro[iî]t\w*|\bconno[iî]\w*|\bencor\b|très-|"
    r"\b(?:ét|av|aur|ser|fais|dis|voul|pouv|sav|dev|all|ven|croy|pens|parl|aim)o(?:is|it|ient)\b", re.IGNORECASE)
PATOIS = re.compile(r"\bbiau\b|\bça est\b|\bje \w+ons\b|\bmonsieu\b|\bpargu[ée]\b|\bmorgu[ée]\b|\btestigu[ée]\b",
                    re.IGNORECASE)
PONCTUATION_FAUTIVE = re.compile(r"(?<!\.)\.\.(?!\.)")
# Le chatbot est montré à un jury, et le gabarit demande au personnage de rester bienséant : pas de
# modèle de phrase sur l'enfer, l'autre vie ou les lavements.
BIENSEANCE = re.compile(r"\benfers?\b|\bautre vie\b|\blavements?\b", re.IGNORECASE)
# Tournures d'aujourd'hui relevées dans la carte de la chaîne : un exemple interdit ses suites de six
# mots partout, messages de la personne compris, et ces trois-là faisaient rejeter des messages naturels.
SUITE_BANALE = re.compile(r"qu'est-ce que c'est|je vous assure que c'est|je ne sais ce que vous voulez dire",
                          re.IGNORECASE)
# Règle générale, à la suite de la liste ci-dessus : le filtre « copie_exemple » interdit à TOUTES les
# conversations (messages de la personne compris) chaque suite de six mots de chaque vraie réplique
# gardée, montrée ou non. Une réplique dont une suite de six mots n'est faite que de mots très
# courants (liste fermée ci-dessous : articles, pronoms, prépositions, conjonctions, adverbes usuels,
# être, avoir et quelques verbes très fréquents, découpés comme le fait le filtre) ferait donc
# rejeter des phrases banales : elle est écartée.
MOTS_COURANTS = frozenset("""
le la les l un une des du de d au aux ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos
leur leurs je j tu il elle on nous vous ils elles me m te t se s lui y en moi toi soi eux ceci cela ça c qui
que qu quoi dont où à dans par pour sur sous avec sans chez vers entre et ou mais donc car ni si comme quand
lorsque puisque parce ne n pas point plus moins très bien mal tout tous toute toutes rien jamais toujours
encore déjà aussi même ici là oui non peu trop assez fort tant est es suis sommes êtes sont était étais
sera serai soit été être a ai as avons avez ont avait avais aura aurai ait eu avoir fait fais faire font
faites dit dis dire dites va vais vas aller allons allez vont veux veut voulez vouloir peux peut pouvez
pouvoir sais sait savez savoir faut dois doit devez voir vois voit voyez parle parler parlez crois croit
croyez viens vient venez prendre prends donne donner laisse laissez mettre remercie
""".split())

# Table « interdit incompatible avec telle valeur d'axe » : (motif dans le texte de l'interdit, côté,
# motif dans le texte de la fiche du même côté). Le compositeur n'associe jamais un interdit à une fiche
# qu'il heurte (clé « heurte » des axes) ; cette table est le contrôle indépendant qui ARRÊTE la
# composition si une fiche contient quand même une des contradictions relevées à la relecture.
# Côté « reponse » : façon de répondre, longueur, début imposé. Côté « message » : type de message,
# forme, détail, déroulé.
CONTRADICTIONS = [
    (r"voilà|voici", "reponse", r"voilà|voici"),
    (r"exclamation", "reponse", r"exclam|s'indigner|applaudir|féliciter"),
    (r"point-virgule|deux-points", "reponse", r"point-virgule|deux-points"),
    (r"comparaison|« comme »", "reponse", r"compar|rapproch"),
    (r"généralité|« toujours »|« jamais »", "reponse", r"maxime|leçon"),
    (r"juron|« Ah »", "reponse", r"interjection|juron"),
    (r"« Je »|« J' »|« Vous »|« Votre »", "reponse", r"pronom|possessif"),
    (r"question de moins de six mots", "reponse", r"demander|question|relancer"),
    (r"« une chose »", "reponse", r"chose"),
    (r"croyez-moi|sachez", "reponse", r"impératif"),
    (r"est-ce que", "message", r"est-ce que"),
    (r"« ça »|« c'est »", "message", r"« ça »|« on »"),
    (r"« merci »", "message", r"merci|remerci"),
    (r"« mais »", "message", r"object"),
    (r"que faire|comment faire", "message", r"infinitif"),
    (r"« je » en premier|commence par « je »", "message", r"."),  # interdit retiré : tournure la plus naturelle
]


def mots(texte):
    return MOT.findall(texte.lower())


def sha256(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def relatif(chemin):
    """Chemin écrit dans la sortie : relatif à la racine du clone quand c'est possible (stable)."""
    chemin = Path(chemin).resolve()
    try:
        return chemin.relative_to(RACINE).as_posix()
    except ValueError:
        return chemin.as_posix()


def erreur(message):
    sys.exit("ERREUR : " + message)


# ------------------------------------------------------------------ vraies répliques propres
def raison_exemple(e):
    """Pourquoi une vraie réplique n'est pas « propre » (None = gardée). Première raison seulement."""
    texte = e["texte"]
    piece = (e.get("piece") or "").lower()
    if any(p in piece for p in PIECES_A_L_ECART):
        return "piece_tenue_a_l_ecart"
    if LETTRE_ISOLEE.search(texte):
        return "lettre_majuscule_isolee"  # « A ce que je vois » : un A isolé recopié donne « meta »
    if DIDASCALIE.search(texte):
        return "didascalie"
    if GRAPHIE_ANCIENNE.search(texte):
        return "graphie_ancienne"
    if PATOIS.search(texte):
        return "patois"
    if PONCTUATION_FAUTIVE.search(texte):
        return "ponctuation_fautive"
    jetons = texte.split()
    for avant, mot in zip(jetons, jetons[1:]):
        nu = mot.strip(",;:.!?'\"«»")
        if nu[:1].isupper() and not re.search(r"[.!?]$", avant) and nu not in MAJUSCULES_PERMISES:
            return "vers_a_majuscule_interne"  # le personnage doit parler en prose
    if BIENSEANCE.search(texte):
        return "bienseance"
    if SUITE_BANALE.search(texte):
        return "suite_banale_d_aujourd_hui"
    ws = mots(texte)
    if any(all(w in MOTS_COURANTS for w in ws[i:i + 6]) for i in range(len(ws) - 5)):
        return "suite_de_six_mots_courants"
    return None


def exemples_propres(exemples):
    gardes, ecartes = [], Counter()
    for e in exemples:
        raison = raison_exemple(e)
        if raison:
            ecartes[raison] += 1
        else:
            gardes.append(e)
    return gardes, dict(sorted(ecartes.items()))


# ------------------------------------------------------------------ lecture et contrôle des axes
def valeurs(axes, cle):
    return axes[cle]["valeurs"]


def groupes_de(axes):
    return {g: v for g, v in axes["groupes"].items() if not g.startswith("_")}


def controle_texte(texte, ou, maxi=None):
    """Un texte destiné à une fiche ne doit rien contenir que les filtres puniraient s'il était recopié."""
    ws = set(mots(texte))
    if ws & TROISIEME_PERSONNE:
        erreur(f"{ou} : mot de troisième personne {sorted(ws & TROISIEME_PERSONNE)} dans « {texte} »")
    if ws & TUTOIEMENT:
        erreur(f"{ou} : mot du tutoiement {sorted(ws & TUTOIEMENT)} dans « {texte} »")
    if SIGNES_INTERDITS.search(texte) or LETTRE_ISOLEE.search(texte):
        erreur(f"{ou} : signe, mot ou lettre surveillés par le filtre « meta » dans « {texte} »")
    if maxi is not None and len(texte) > maxi:
        erreur(f"{ou} : {len(texte)} caractères, plus que le maximum de {maxi}, dans « {texte} »")


def themes_a_plat(axes):
    """[(code, famille, groupe, note)] dans l'ordre du fichier ; le code est la position (à partir de 1)."""
    plats, vus = [], {}
    groupes = groupes_de(axes)
    maxi = axes["fiche"]["max_par_axe"]["famille"]
    for bloc in axes["themes"]:
        if bloc["groupe"] not in groupes:
            erreur(f"thèmes « {bloc['famille']} » : groupe inconnu « {bloc['groupe']} »")
        controle_texte(bloc["famille"], "famille de thèmes", maxi)
        for note in bloc["notes"]:
            cle = " ".join(mots(note))
            if cle in vus:
                erreur(f"note de thème en double : « {note} » ({bloc['famille']}) et « {vus[cle]} »")
            vus[cle] = f"{note} ({bloc['famille']})"
            if len(mots(note)) > MAX_MOTS_PAR_NOTE:
                erreur(f"note de thème de plus de {MAX_MOTS_PAR_NOTE} mots : « {note} »")
            controle_texte(note, "note de thème")
            plats.append((len(plats) + 1, bloc["famille"], bloc["groupe"], note))
    return plats


def formes_permises(type_, formes):
    """Formes de premier message compatibles avec un type de message (ordre du fichier)."""
    connues = {f["id"] for f in formes}
    inconnues = [f for f in type_.get("formes", []) + type_.get("sans_formes", []) if f not in connues]
    if inconnues:
        erreur(f"type « {type_['id']} » : formes inconnues {inconnues}")
    if "formes" in type_:
        return [f for f in formes if f["id"] in type_["formes"]]
    return [f for f in formes
            if set(f["porte"]) & set(type_["porte"])
            and f["id"] not in type_.get("sans_formes", [])
            and (type_.get("court", True) or not f.get("courte"))
            and (type_.get("long", True) or not f.get("longue"))]


def deroules_compatibles(type_, groupe, forme, deroules):
    """Déroulés permis pour ce type de message, ce groupe de thèmes et cette forme (ordre du fichier)."""
    return [d["id"] for d in deroules
            if d["allers_retours"] <= type_.get("max_allers_retours", 99)
            and ("groupes" not in d or groupe in d["groupes"])
            and ("types" not in d or type_["id"] in d["types"])
            and ("deroules" not in type_ or d["id"] in type_["deroules"])
            and not (d.get("sauf_politesse") and type_.get("politesse"))
            and not (d.get("exige_question") and not type_.get("question"))
            and not (d.get("sauf_formes_courtes") and forme.get("courte"))]


def longueurs_compatibles(type_, reponse, longueurs):
    """Longueurs permises : pas de phrase unique pour une façon de répondre en deux temps (« puis »)."""
    return [x["id"] for x in longueurs
            if x["id"] not in type_.get("sans_longueurs", [])
            and not (x.get("sans_puis") and "puis" in mots(reponse))]


def heurte(interdit, texte):
    return any(fragment in texte.lower() for fragment in interdit.get("heurte", []))


def contradictions(interdit, cote, texte):
    """Lignes de la table CONTRADICTIONS que ce couple (interdit, texte de fiche du même côté) enfreint."""
    return [(a, b) for a, c, b in CONTRADICTIONS
            if c == cote and re.search(a, interdit, re.IGNORECASE) and re.search(b, texte, re.IGNORECASE)]


def controle_axes(axes):
    brut = json.dumps({k: v for k, v in axes.items()}, ensure_ascii=False)
    evoque = A_NE_PAS_EVOQUER.search(brut)
    if evoque:
        erreur(f"les axes évoquent une pièce tenue à l'écart pour la mesure : « {evoque.group(0)} »")
    groupes = groupes_de(axes)
    maxi = axes["fiche"]["max_par_axe"]
    types, formes = valeurs(axes, "types_de_message"), valeurs(axes, "formes_du_premier_message")
    deroules, longueurs = valeurs(axes, "deroules"), valeurs(axes, "longueurs_des_reponses")
    entrees = valeurs(axes, "entrees_de_la_premiere_reponse")
    listes = {"type": types, "forme": formes, "deroule": deroules, "entree": entrees, "longueur": longueurs,
              "interdit côté réponses": axes["interdits"]["reponses"],
              "interdit côté messages": axes["interdits"]["messages"], "voix": valeurs(axes, "voix")}
    tous_les_ids = []
    for nom, liste in listes.items():
        ids = [v["id"] for v in liste]
        if len(set(ids)) != len(ids):
            erreur(f"axe « {nom} » : id en double")
        tous_les_ids += [] if nom == "voix" else ids
        for v in liste:
            if "#" in v["id"] or "-" in v["id"]:
                erreur(f"axe « {nom} » : l'id « {v['id']} » contient « # » ou « - »")
            if nom != "voix":
                controle_texte(v["texte"], nom, maxi[nom.split(" ")[0]])
    tous_les_ids += [axes["detail"]["id"], SANS_SUPPLEMENT]
    if len(set(tous_les_ids)) != len(tous_les_ids):
        erreur("un même id sert à deux axes (l'id d'une fiche deviendrait ambigu)")
    controle_texte(axes["detail"]["texte"], "détail", maxi["detail"])
    id_deroules, id_longueurs = {d["id"] for d in deroules}, {x["id"] for x in longueurs}
    for t in types:
        if set(t["groupes"]) - set(groupes):
            erreur(f"type « {t['id']} » : groupes inconnus {sorted(set(t['groupes']) - set(groupes))}")
        if len({bool(groupes[g].get("essentiel")) for g in t["groupes"]}) != 1:
            erreur(f"type « {t['id']} » : mêle des groupes essentiels et d'autres groupes")
        if set(t.get("deroules", [])) - id_deroules or set(t.get("sans_longueurs", [])) - id_longueurs:
            erreur(f"type « {t['id']} » : déroulé ou longueur inconnus")
        if not t["reponses"]:
            erreur(f"type « {t['id']} » : aucune façon de répondre appariée")
        permises = formes_permises(t, formes)
        if not permises:
            erreur(f"type « {t['id']} » : aucune forme de premier message compatible")
        for r in t["reponses"]:
            controle_texte(r, f"façon de répondre du type « {t['id']} »", maxi["reponse"])
            if not longueurs_compatibles(t, r, longueurs):
                erreur(f"type « {t['id']} » : aucune longueur compatible avec « {r} »")
        for g in t["groupes"]:
            for f in permises:
                if not deroules_compatibles(t, g, f, deroules):
                    erreur(f"type « {t['id']} », groupe « {g} », forme « {f['id']} » : aucun déroulé compatible")
    sans_type = sorted(g for g in groupes if not any(g in t["groupes"] for t in types))
    if sans_type:
        erreur(f"groupes de thèmes sans aucun type de message compatible : {sans_type}")
    for d in deroules:
        if set(d.get("groupes", [])) - set(groupes) or set(d.get("types", [])) - {t["id"] for t in types}:
            erreur(f"déroulé « {d['id']} » : groupes ou types inconnus")
    for v in valeurs(axes, "voix"):
        if not v["id"].startswith(("tu", "vous")):
            erreur(f"voix « {v['id']} » : l'id doit commencer par « tu » ou « vous » (filtre « registre »)")
    ess = axes["essentiels"]
    if not 0 <= ess["part"] < 1 or not 0 <= ess["part_des_deroules_preferes"] <= 1:
        erreur("essentiels : parts hors de [0, 1]")
    if set(ess["deroules_preferes"]) - id_deroules:
        erreur("essentiels : déroulé préféré inconnu")
    for sorte in ("essentiels", "vie courante", "autres"):
        parts = axes["supplements"].get(sorte)
        if not parts or set(parts) - set(SUPPLEMENTS) or "aucun" not in parts or min(parts.values()) <= 0:
            erreur(f"supplements.{sorte} : clés permises {SUPPLEMENTS}, « aucun » obligatoire, parts positives")
    if "interdit_message" in axes["supplements"]["essentiels"] or "detail" in axes["supplements"]["essentiels"]:
        erreur("supplements.essentiels : ni détail imposé ni interdit côté messages pour le bloc « essentiels »")
    gab = axes["gabarit"]
    if "#" in gab["id"]:
        erreur("l'id du gabarit ne doit pas contenir « # » (reprise : prompt_id.split('#'))")
    if any("{demonstrations}" in ligne for ligne in gab["lignes"]):
        erreur("gabarit : l'emplacement « demonstrations » n'est plus employé (liste vide)")
    try:  # accolade littérale ou nom inconnu -> KeyError / ValueError / IndexError au .format
        "\n".join(gab["lignes"]).format(**NOMS_DU_GABARIT)
        gab["ligne_sujet"].format(i=1, sujet="F")
        gab["ligne_exemple"].format(texte="E")
        gab["ligne_demonstration"].format(a="M", b="R")
        "\n".join(axes["juge"]["lignes"]).format(contexte="C", conversation="V", question="Q")
        axes["juge"]["contexte_moliere"].format(persona="P")
    except (KeyError, ValueError, IndexError) as e:
        erreur(f"gabarit : emplacement inconnu ou accolade littérale ({e!r})")
    if "\n".join(axes["juge"]["lignes"]).startswith(gab["lignes"][0]) or gab["lignes"][0].startswith(
            axes["juge"]["lignes"][0]):
        erreur("le prompt de génération et celui du juge commencent pareil (est_prompt_juge les confondrait)")


# ------------------------------------------------------------------ tirage équilibré
class Tourniquet:
    """Rend, parmi les candidats, celui qui a été le moins servi au regard de son poids.

    Égalité -> tirage au sort (graine fixe). Les compteurs sont communs à tous les appels : une
    valeur écartée par une table de compatibilité est rattrapée dès qu'elle redevient permise.
    `choisit` ne compte rien ; `sert` compte ; `tire` fait les deux.
    """

    def __init__(self, rng, poids=None):
        self.rng, self.poids, self.servis = rng, poids or {}, Counter()

    def charge(self, c):  # « + 1 » : une valeur de poids double passe avant, pas seulement plus souvent
        return (self.servis[c] + 1) / self.poids.get(c, 1.0)

    def choisit(self, candidats):
        plus_bas = min(self.charge(c) for c in candidats)
        return self.rng.choice([c for c in candidats if self.charge(c) == plus_bas])

    def sert(self, c):
        self.servis[c] += 1

    def tire(self, candidats):
        choix = self.choisit(candidats)
        self.sert(choix)
        return choix


def tours_melanges(rng, elements, n):
    """n éléments pris par tours complets mélangés : chacun revient autant que les autres, à un près."""
    ordre = []
    while len(ordre) < n:
        tour = list(elements)
        rng.shuffle(tour)
        ordre += tour
    return ordre[:n]


def composer_fiches(axes, n, graine):
    """n fiches {id, theme, texte} + le décompte du tirage par axe."""
    rng = random.Random(f"consignes-v4:{graine}")
    themes = themes_a_plat(axes)
    groupes = groupes_de(axes)
    types = {t["id"]: t for t in valeurs(axes, "types_de_message")}
    formes = valeurs(axes, "formes_du_premier_message")
    deroules = valeurs(axes, "deroules")
    entrees = valeurs(axes, "entrees_de_la_premiere_reponse")
    longueurs = valeurs(axes, "longueurs_des_reponses")
    inter = {"interdit_reponse": axes["interdits"]["reponses"], "interdit_message": axes["interdits"]["messages"]}
    detail = axes["detail"]
    par_id = {v["id"]: v for axe in (formes, deroules, entrees, longueurs, *inter.values(), [detail]) for v in axe}
    seg, ess = axes["fiche"], axes["essentiels"]
    preferes = ess["deroules_preferes"]

    def essentiel(groupe):
        return bool(groupes[groupe].get("essentiel"))

    types_du_groupe = {g: [t["id"] for t in types.values() if g in t["groupes"]] for g in groupes}
    formes_du_type = {t["id"]: [f["id"] for f in formes_permises(t, formes)] for t in types.values()}
    courtes = [f["id"] for f in formes if f.get("courte")]
    part = axes["formes_du_premier_message"]["part_des_formes_courtes"]
    poids_formes = {f["id"]: (part / len(courtes) if f.get("courte") else (1 - part) / (len(formes) - len(courtes)))
                    for f in formes}
    parts_ar = {int(k): v for k, v in axes["deroules"]["poids_des_allers_retours"].items()}
    nb_par_ar = Counter(d["allers_retours"] for d in deroules)
    poids_deroules = {d["id"]: parts_ar[d["allers_retours"]] / nb_par_ar[d["allers_retours"]] for d in deroules}
    poids_longueurs = {x["id"]: x.get("poids", 1) for x in longueurs}
    poids_types = {t["id"]: int(t.get("poids", 1)) for t in types.values()}

    # Tourniquets. Forme, déroulé, longueur et sorte de supplément sont équilibrés PAR TYPE de message (la
    # longueur par façon de répondre) : avec un compteur commun, une valeur que peu de types permettent
    # serait « rattrapée » sur ces seuls types et leur serait servie presque à chaque fois (une demande
    # de nouvelles n'aurait par exemple jamais reçu la forme la plus courante, deux à quatre mots). Les
    # valeurs des suppléments (début imposé, interdits) tournent sur un compteur par population.
    def par_cle(poids=None):
        return collections.defaultdict(lambda: Tourniquet(rng, poids))
    t_formes, t_deroules, t_preferes = par_cle(poids_formes), par_cle(poids_deroules), par_cle(preferes)
    t_longueurs = par_cle(poids_longueurs)
    t_supplements = {sorte: par_cle(axes["supplements"][sorte])
                     for sorte in ("essentiels", "vie courante", "autres")}
    jeux = {pop: {"entree": Tourniquet(rng), "interdit_reponse": Tourniquet(rng),
                  "interdit_message": Tourniquet(rng)} for pop in (True, False)}
    t_types = {g: Tourniquet(rng, poids_types) for g in groupes}
    t_reponses = {t: Tourniquet(rng) for t in types}

    # Ordre des thèmes : une part fixée de fiches « essentiels », réparties régulièrement parmi les autres.
    les_essentiels = [t for t in themes if essentiel(t[2])]
    les_autres = [t for t in themes if not essentiel(t[2])]
    n_ess = min(n, round(ess["part"] * n)) if les_essentiels else 0
    if not les_autres:
        n_ess = n
    file_ess = tours_melanges(rng, les_essentiels, n_ess)
    file_autres = tours_melanges(rng, les_autres, n - n_ess)
    ordre = [file_ess.pop(0) if (i + 1) * n_ess // n > i * n_ess // n else file_autres.pop(0) for i in range(n)]

    def sans_supplement(note, type_, forme, sid, reponse, lid):
        """(texte de la fiche sans supplément, position où s'insère le détail)."""
        debut = seg["theme"].format(note=note) + seg["message"].format(type=type_["texte"], forme=forme["texte"])
        return (debut + seg["deroule"].format(deroule=par_id[sid]["texte"])
                + seg["reponse"].format(reponse=reponse, longueur=par_id[lid]["texte"])), len(debut)

    def ajout(genre, val):
        """Texte qu'un supplément ajoute à la fiche."""
        if genre == "aucun":
            return ""
        return seg[genre].format(**{genre.split("_")[0]: par_id[val]["texte"]})

    def supplements_possibles(note, type_, groupe, forme, sid, reponse, lid):
        """{sorte de supplément: [id des valeurs que cette fiche peut recevoir]}.

        Un supplément qui contredirait la fiche, ou qui lui ferait dépasser la longueur maximale, n'est
        pas proposé.
        """
        cote_reponse = reponse + " " + par_id[lid]["texte"]
        cote_message = " ".join((type_["texte"], forme["texte"], par_id[sid]["texte"]))
        place = seg["max_caracteres"] - len(sans_supplement(note, type_, forme, sid, reponse, lid)[0])
        possibles = {"aucun": [SANS_SUPPLEMENT]}
        if (groupes[groupe].get("detail") and not forme.get("courte")
                and forme["id"] not in detail.get("sauf_formes", [])):
            possibles["detail"] = [detail["id"]]
        if not type_.get("sans_entree"):
            possibles["entree"] = [e["id"] for e in entrees
                                   if not (e.get("sauf_formes_courtes") and forme.get("courte"))]
        possibles["interdit_reponse"] = [i["id"] for i in inter["interdit_reponse"] if not heurte(i, cote_reponse)]
        if not essentiel(groupe):
            possibles["interdit_message"] = [
                i["id"] for i in inter["interdit_message"]
                if par_id[sid]["allers_retours"] >= i.get("min_allers_retours", 1) and not heurte(i, cote_message)]
        possibles = {k: [v for v in vals if len(ajout(k, v)) <= place] for k, vals in possibles.items()}
        return {k: v for k, v in possibles.items() if v}

    fiches, vues, quotas = [], set(), {}
    compte = {k: Counter() for k in ("theme", "groupe", "type", "forme", "deroule", "allers_retours", "longueur",
                                     "supplement", "valeur_du_supplement")}
    nb_ess, nb_preferes = Counter(), Counter()  # par type de message essentiel
    tailles, tailles_sans, nb_mots = [], [], []
    for numero, (code, famille, groupe, note) in enumerate(ordre, 1):
        est_ess = essentiel(groupe)
        jeu_ = jeux[est_ess]
        sorte = "essentiels" if est_ess else "vie courante" if groupes[groupe].get("detail") else "autres"
        # Type de message : compatible avec le groupe du thème. Chaque note a un quota par cycle (le
        # poids du type) : elle reçoit des messages de nature différente avant d'en reprendre un.
        quota = quotas.setdefault(code, Counter())
        candidats = [t for t in types_du_groupe[groupe] if quota[t] > 0]
        if not candidats:
            quota.update({t: poids_types[t] for t in types_du_groupe[groupe]})
            candidats = types_du_groupe[groupe]
        tid = t_types[groupe].tire(candidats)
        quota[tid] -= 1
        type_ = types[tid]
        rep = t_reponses[tid].tire(list(range(len(type_["reponses"]))))
        reponse = type_["reponses"][rep]
        fid = t_formes[tid].tire(formes_du_type[tid])
        forme = par_id[fid]

        # Déroulé, longueur, supplément : choisis d'abord sans rien compter, pour pouvoir écarter une
        # fiche déjà composée (même thème, même type, mêmes valeurs) avant de servir les tourniquets.
        compat = deroules_compatibles(type_, groupe, forme, deroules)
        des_preferes = [d for d in compat if d in preferes] if est_ess else []
        des_autres = [d for d in compat if d not in des_preferes]
        if des_preferes and (not des_autres
                             or nb_preferes[tid] < ess["part_des_deroules_preferes"] * (nb_ess[tid] + 1)):
            sid = t_preferes[tid].choisit(des_preferes)
        else:
            sid = t_deroules[tid].choisit(des_autres)
        lcand = longueurs_compatibles(type_, reponse, longueurs)
        lid = t_longueurs[tid, rep].choisit(lcand)
        possibles = supplements_possibles(note, type_, groupe, forme, sid, reponse, lid)
        parts = axes["supplements"][sorte]
        genre = t_supplements[sorte][tid].choisit([k for k in parts if k in possibles])
        val = possibles[genre][0] if genre in ("aucun", "detail") else jeu_[genre].choisit(possibles[genre])
        cle = (code, tid, rep, fid, sid, lid, val)
        if cle in vues:  # fiche déjà composée : une autre combinaison, tirée parmi celles qui restent
            libres = []
            for sid2 in compat:
                for lid2 in lcand:
                    for genre2, vals in supplements_possibles(note, type_, groupe, forme, sid2, reponse,
                                                              lid2).items():
                        if genre2 in parts:
                            libres += [(sid2, lid2, genre2, v) for v in vals
                                       if (code, tid, rep, fid, sid2, lid2, v) not in vues]
            if not libres:
                erreur(f"impossible de rendre unique la fiche n° {numero} : baisser --fiches")
            # ... en changeant le moins de choses possible : d'abord la valeur du supplément, puis sa sorte,
            # puis la longueur, et le déroulé en dernier (il porte les parts voulues d'allers-retours).
            ecart = min((s != sid, x != lid, g != genre) for s, x, g, _ in libres)
            sid, lid, genre, val = rng.choice(
                [c for c in libres if (c[0] != sid, c[1] != lid, c[2] != genre) == ecart])
            cle = (code, tid, rep, fid, sid, lid, val)
        vues.add(cle)
        if est_ess:
            nb_ess[tid] += 1
            nb_preferes[tid] += sid in des_preferes
        (t_preferes if sid in des_preferes else t_deroules)[tid].sert(sid)
        t_longueurs[tid, rep].sert(lid)
        t_supplements[sorte][tid].sert(genre)
        if genre not in ("aucun", "detail"):
            jeu_[genre].sert(val)
        ar = par_id[sid]["allers_retours"]

        sans, avant_deroule = sans_supplement(note, type_, forme, sid, reponse, lid)
        if genre == "detail":  # le détail suit la description du premier message
            texte = sans[:avant_deroule] + ajout(genre, val) + sans[avant_deroule:]
        else:
            texte = sans + ajout(genre, val)
        if genre in inter:
            cote = "reponse" if genre == "interdit_reponse" else "message"
            du_cote = (reponse + " " + par_id[lid]["texte"] if cote == "reponse"
                       else " ".join((type_["texte"], forme["texte"], par_id[sid]["texte"])))
            if contradictions(par_id[val]["texte"], cote, du_cote):
                erreur(f"fiche n° {numero} : l'interdit « {par_id[val]['texte']} » contredit « {du_cote} » "
                       f"(table CONTRADICTIONS) : compléter la clé « heurte » de cet interdit")
        controle_texte(texte, f"fiche n° {numero}")
        if "\n" in texte:
            erreur(f"fiche n° {numero} : une fiche tient sur une seule ligne")
        if len(texte) > seg["max_caracteres"]:
            erreur(f"fiche n° {numero} : {len(texte)} caractères, plus que le maximum de "
                   f"{seg['max_caracteres']} : raccourcir un texte d'axe ({texte})")
        tailles.append(len(texte))
        tailles_sans.append(len(sans))
        nb_mots.append(len(texte.split()))
        fiches.append({"id": f"v4-{numero:05d}-t{code:03d}-{tid}{rep + 1}-{fid}-{sid}-{lid}-{val}",
                       "theme": famille, "texte": texte})
        for k, v in (("theme", famille), ("groupe", groupe), ("type", tid), ("forme", fid), ("deroule", sid),
                     ("allers_retours", str(ar)), ("longueur", lid), ("supplement", genre),
                     ("valeur_du_supplement", val)):
            compte[k][v] += 1
    par_note = Counter(code for code, *_ in ordre)
    notes_ess = [par_note[t[0]] for t in les_essentiels]
    notes_autres = [par_note[t[0]] for t in les_autres]
    bilan = {k: dict(sorted(v.items())) for k, v in compte.items()}
    bilan["essentiels"] = {
        "fiches": n_ess, "part": round(n_ess / n, 4), "notes": len(les_essentiels),
        "fiches_par_note": {"min": min(notes_ess, default=0), "max": max(notes_ess, default=0)},
        "deroules_preferes": sum(nb_preferes.values()),
        "part_des_deroules_preferes": round(sum(nb_preferes.values()) / max(n_ess, 1), 4),
        "allers_retours": dict(sorted(Counter(
            str(par_id[f["id"].split("-")[5]]["allers_retours"]) for f in fiches
            if f["id"].split("-")[3].startswith("x")).items()))}
    bilan["part_des_formes_courtes"] = {
        "essentiels": round(sum(f["id"].split("-")[4] in courtes for f in fiches
                                if f["id"].split("-")[3].startswith("x")) / max(n_ess, 1), 4),
        "autres": round(sum(f["id"].split("-")[4] in courtes for f in fiches
                            if not f["id"].split("-")[3].startswith("x")) / max(n - n_ess, 1), 4)}
    bilan["fiches_par_note_hors_essentiels"] = {"min": min(notes_autres, default=0),
                                                "max": max(notes_autres, default=0)}
    bilan["caracteres_par_fiche"] = {
        "mediane": int(statistics.median(tailles)), "max": max(tailles),
        "mediane_sans_supplement": int(statistics.median(tailles_sans)), "max_sans_supplement": max(tailles_sans),
        "objectif": seg["objectif_caracteres"],
        "part_dans_l_objectif": round(sum(t <= seg["objectif_caracteres"] for t in tailles) / n, 4),
        "mots_mediane": int(statistics.median(nb_mots))}
    return fiches, bilan, themes


# ------------------------------------------------------------------ le fichier de consignes
def empreinte_des_consignes(cfg):
    """8 caractères d'empreinte de tout le fichier composé, hors l'empreinte elle-même.

    Calculée avec l'id de gabarit de base ; le compositeur l'ajoute ensuite à cet id. Le lanceur la
    recalcule pour refuser un fichier de consignes retouché à la main.
    """
    c = copy.deepcopy(cfg)
    c["composition"].pop("empreinte", None)
    c["gabarits"]["moliere"]["id"] = c["composition"]["gabarit_de_base"]
    return hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def composer(modele, licence, fiches_voulues, graine, axes_path=AXES, origine_path=ORIGINE):
    axes = json.loads(Path(axes_path).read_text(encoding="utf-8"))
    base = json.loads(Path(origine_path).read_text(encoding="utf-8"))
    controle_axes(axes)
    fiches, bilan, themes = composer_fiches(axes, fiches_voulues, graine)

    voix = valeurs(axes, "voix")
    clause = axes["voix"]["clause_des_voix_vous"]
    moderne = {p: next(r["moderne"] for r in base["registres"] if r["id"].startswith(p)) for p in ("tu", "vous")}
    registres = [{"id": v["id"],
                  "moderne": moderne["vous" if v["id"].startswith("vous") else "tu"],
                  "moliere": v["texte"] + (" ; " + clause if v["id"].startswith("vous") else "")}
                 for v in voix]
    exemples, exemples_ecartes = exemples_propres(base["exemples_moliere"])
    nb_exemples = 4
    if len(exemples) < 4 * nb_exemples:
        erreur(f"trop peu de vraies répliques propres ({len(exemples)}) : chacune serait montrée trop souvent")

    filtres = copy.deepcopy(base["filtres"])
    if not filtres.get("mots_modernes_reponse_moliere"):
        erreur("mots_modernes_reponse_moliere vide dans le fichier d'origine : tout serait rejeté")
    filtres["tours_moliere"] = [2, 8]
    filtres["echo_max"] = 0.7
    filtres["echo_max_utilisateur"] = 0.5
    filtres["max_conversations_par_5gramme"] = base["filtres"]["max_conversations_par_5gramme"]
    filtres["formules_pour_mille"] = 6
    tics = list(filtres.get("tics_plafonnes", {}).get("moliere", []))
    tics += [t for t in axes["tics_plafonnes_en_plus"]["valeurs"] if t not in tics]
    filtres["tics_plafonnes"] = dict(filtres.get("tics_plafonnes", {}), moliere=tics)
    for cle, valeur in axes.get("filtres_en_plus", {}).get("valeurs", {}).items():  # axes v5 : seuils remplacés
        filtres[cle] = dict(filtres.get(cle) or {}, **valeur) if isinstance(valeur, dict) else valeur
    filtres["tours_moliere_justification"] = (
        "De 2 à 8 tours (un à quatre allers-retours) au lieu de 4 à 6 : les fiches demandent aussi le message "
        "seul suivi de sa réponse et la conversation suivie, comme un vrai utilisateur. Le code exige toujours "
        "un nombre pair de tours : la personne commence, le personnage termine. Clé commune : le style "
        "« moderne » garde tours_moderne.")
    filtres["max_mots_par_tour_justification"] = (
        f"Hérité du fichier d'origine ({base['filtres']['max_mots_par_tour']} mots par tour). Les longueurs "
        "demandées par les fiches sont des classes sans compte de mots (une phrase courte, deux phrases, trois "
        "phrases brèves) : le plafond ne sert qu'à écarter une réponse démesurée. Le filtre compte l'apostrophe "
        "et le trait d'union comme des coupures (« n'est » = 2 mots) : un plafond serré rejetterait des réponses "
        "obéissantes.")
    filtres["echo_v4_justification"] = (
        "ASSOUPLISSEMENT DÉCLARÉ, à confirmer par l'étudiant. echo_max passe de "
        f"{base['filtres']['echo_max']} à 0.7 et echo_max_utilisateur de {base['filtres']['echo_max_utilisateur']} "
        "à 0.5 ; remettre les valeurs d'origine pour revenir à la définition d'origine. Pourquoi : le filtre "
        "« echo » mesure la part des mots distincts du tour précédent repris dans le tour suivant (dès 3 mots "
        "distincts), et ces seuils avaient été réglés sur des messages longs (dans la génération précédente, "
        "91 % des messages gardés avaient 6 mots distincts ou plus). Les fiches demandent maintenant des "
        "messages et des réponses courts : après un message de 3 mots distincts, deux mots repris suffisaient "
        "au rejet (0,67). Mesuré en lecture seule sur les paires réelles de moliere_train (réponse de 6 à 45 "
        "mots) : rejet « echo » de 9,0 % après un message de 3 mots distincts au seuil 0,6 et de 3,5 % au seuil "
        "0,7 ; sur toutes les paires, 1,5 % puis 0,7 %. Côté personne (tour qui suit une réponse) : 5,1 % au "
        "seuil 0,4, 2,9 % au seuil 0,5. Ce que le nouveau seuil laisse passer en plus : une réponse qui reprend "
        "2 mots distincts sur 3, 3 sur 5, 4 sur 6 ou 5 sur 8 du message ; 3 sur 4 et au-delà restent rejetés, "
        "comme la reprise des 3 premiers mots de la réponse par la personne. À mesurer sur le lot pilote : les "
        "rejets « echo » par forme et par longueur (tools/campagne_synthese.py bilan).")
    filtres["formules_justification"] = (
        "ASSOUPLISSEMENT DÉCLARÉ, à confirmer par l'étudiant. Une même suite de 5 mots peut apparaître dans au "
        "plus max(3, 6 pour 1 000 conversations gardées) conversations : le plancher de 3 est inchangé "
        "(max_conversations_par_5gramme), le seuil proportionnel passe de 3 à 6 pour mille. Pourquoi : "
        "l'ancienne génération tournait de fait à 8,5 pour mille par fichier (3 sur 353 gardées) et c'est le "
        "refiltrage à 3 pour mille du fichier fusionné qui a fait passer 3 594 conversations gardées à 1 305 ; "
        "les tournures classiques banales du personnage et les débuts de phrase banals de la personne "
        "(« je n'ai pas de », compté 5 mots parce que l'apostrophe coupe) dépassent 3 pour mille. À 6 pour "
        "mille, une suite peut figurer dans 30 conversations sur 5 000. Cela change la définition de "
        "« distinct » : remettre 3 pour revenir à la définition d'origine. Restent stricts : doublon exact, "
        "aucun tour de 6 mots ou plus présent deux fois dans tout le fichier, plafond de 15 % par premier "
        "mot de réponse et par tic.")
    filtres["tics_v4_justification"] = (
        "Tics ajoutés à tics_plafonnes.moliere (" + ", ".join(f"« {t} »" for t in axes["tics_plafonnes_en_plus"]["valeurs"])
        + ") : formules relevées dans les réponses gardées de la génération précédente. Chacune est permise "
        "dans 15 % au plus des réponses du personnage ; le plafond est mécanique, ces formules n'ont donc pas "
        "besoin d'être citées dans chaque prompt.")

    gab, jug = axes["gabarit"], axes["juge"]
    gabarits = copy.deepcopy(base["gabarits"])
    gabarits["moliere"] = {k: gab[k] for k in ("id", "lignes", "ligne_sujet", "ligne_exemple", "ligne_demonstration")}
    gabarits["juge"] = dict(base["gabarits"]["juge"], id=jug["id"], lignes=jug["lignes"],
                            contexte_moliere=jug["contexte_moliere"])

    options = {"num_ctx": 4096, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.1, "stop": ["###"]}
    nb_types = len(valeurs(axes, "types_de_message"))
    nb_interdits = {k: len(v) for k, v in axes["interdits"].items() if not k.startswith("_")}
    car = bilan["caracteres_par_fiche"]
    cfg = {
        "version": 4,
        "description": [
            "Consignes « v4 fiches » de la campagne de données de synthèse du chatbot Molière, lues par "
            "chatbot/synthetic.py (même schéma que chatbot/prompts_synthetiques.json).",
            f"Générateur : {modele} (poids ouverts, licence : {licence}), servi en local par Ollama.",
            "Fichier COMPOSÉ, à ne pas éditer à la main : tools/composer_consignes_v4.py croise les axes de "
            "chatbot/axes_consignes_v4.json (rédigés par des agents IA comme consignes : aucun texte de "
            "conversation) et hérite du reste de chatbot/prompts_synthetiques.json. Voir « composition ».",
            "Une conversation par appel, pilotée par une fiche d'échange ; les conversations sont écrites par "
            "le modèle ouvert seul, puis triées par les filtres publiés. Règles suivies : modèle à poids "
            "ouverts, prompts publiés, part de synthèse bornée et déclarée.",
            "Ce fichier sert au style « moliere » seulement, par tools/campagne_synthese.py.",
        ],
        "composition": {
            "script": "tools/composer_consignes_v4.py",
            "date_des_axes": axes["date"],
            "axes": relatif(axes_path), "axes_sha256": sha256(axes_path),
            "origine": relatif(origine_path), "origine_sha256": sha256(origine_path),
            "graine": graine, "fiches": len(fiches), "gabarit_de_base": gab["id"],
            "notes_de_theme": len(themes), "familles_de_themes": len({t[1] for t in themes}),
            "types_de_message": nb_types, "voix": len(registres), "interdits": nb_interdits,
            "vraies_repliques_gardees": len(exemples), "vraies_repliques_ecartees": exemples_ecartes,
            "allers_retours_par_deroule": {d["id"]: d["allers_retours"] for d in valeurs(axes, "deroules")},
            "tirage": bilan,
        },
        "generateur": {"model": modele, "url": URL_OLLAMA, "temperature": 0.8, "think": False},
        "generateur_justification": (
            "Le modèle et sa licence sont les arguments --modele et --licence du compositeur. Température 0,9 "
            "(0,8 dans le fichier d'origine) : la variété doit venir des fiches, pas d'une température qui "
            "multiplierait les fautes de français. think = false : le juge n'a que 20 tokens, un modèle qui "
            "réfléchirait les consommerait (à vérifier sur un appel d'essai avec le modèle retenu)."),
        "licence_generateur": licence,
        "licence_generateur_justification": (
            "Licence des poids du générateur, recopiée par tools/campagne_synthese.py dans le champ "
            "« license » de chaque conversation (à la place de la mention figée dans chatbot/synthetic.py, "
            "qui nomme un autre modèle)."),
        "options_ollama": options,
        "options_ollama_justification": (
            "Options d'échantillonnage ajoutées à chaque appel de GÉNÉRATION par tools/campagne_synthese.py "
            "(paramètre « generateur » de chatbot.synthetic.generer : synthetic.py n'envoie de lui-même que "
            "temperature, num_predict et seed). num_ctx 4096 : le prompt et les 420 tokens générés tiennent "
            "largement, et un contexte fixé empêche le serveur d'en prendre un plus grand qui déborderait la "
            "carte ; top_p 0,95, top_k 64 et repeat_penalty 1,1 : contre les redites ; stop « ### » : le "
            "gabarit demande une ligne ### après la dernière réponse, la génération s'arrête donc là et tout "
            "second échange est coupé (un second bloc n'aurait pas de fiche). Le juge ne reçoit que num_ctx "
            "(même contexte, pour que le serveur ne recharge pas le modèle) : il reste tel que configuré. Ces "
            "options ne sont écrites ni dans les conversations ni dans les rapports de chatbot/synthetic.py : "
            "cette clé en est la trace publiée, et le lanceur la recopie dans le fichier « .campagne.json »."),
        "par_appel": dict(base["par_appel"], moliere=1),
        "par_appel_justification": (
            "Une seule conversation par appel (3 dans le fichier d'origine) : une fiche, un personnage, une "
            "voix, quatre vraies répliques et une graine par conversation. Cela supprime le moule partagé par "
            "les échanges d'un même appel et la fusion de plusieurs sujets en un seul échange (2 007 rejets "
            "« mal_formee » dans la génération précédente)."),
        "num_predict": dict(base["num_predict"], moliere=420),
        "num_predict_justification": (
            "420 tokens générés au plus : le déroulé le plus long (8 tours) fait environ 250 mots."),
        "nb_exemples_par_prompt": nb_exemples,
        "nb_exemples_justification": (
            "4 vraies répliques par prompt au lieu de 6 : elles donnent la couleur de la langue, et chacune "
            "interdit ses suites de 6 mots à tout le fichier (filtre copie_exemple)."),
        "nb_demonstrations_par_prompt": 0,
        "registres": registres,
        "registres_justification": (
            f"Les {len(registres)} voix de la personne (axe « voix »), une par appel, tirée par le code. L'id "
            "commence par « tu » ou « vous » (filtre « registre ») ; chaque voix « vous » reçoit la même clause "
            "finale, car ce filtre rejette tout tutoiement et tout impératif singulier dans un message. La voix "
            "règle la manière, jamais la forme ni la longueur : le gabarit dit que la fiche prime. Le champ "
            "« moderne » est hérité du fichier d'origine (le code le lit pour l'autre style)."),
        "gabarits": gabarits,
        "gabarits_justification": (
            f"gabarits.moliere : prompt à fiche (id de base « {gab['id']} », suivi d'un tiret et des 8 caractères "
            "de composition.empreinte) : règles fixes peu nombreuses, plus aucun échange de démonstration ni "
            "bloc d'interdits, « la personne » et « le personnage » au lieu des lettres d'étiquette. L'id sans "
            "« # » porte l'identité de ces consignes dans les rapports et dans le prompt_id de chaque "
            f"conversation : un fichier recomposé autrement a un autre id. gabarits.juge = « {jug['id']} » : "
            "questions 1 et 2 réécrites pour ces fiches (voir juge_v4_justification). gabarits.moderne est "
            "hérité tel quel."),
        "filtres": filtres,
        "demonstrations_moliere": [],
        "demonstrations_justification": (
            "Liste VIDE et nb_demonstrations_par_prompt = 0 : le prompt ne montre plus aucun échange de "
            "démonstration. Les démonstrations écrites à la main étaient la première source de copies (2 675 "
            "rejets « copie_exemple ») et de moules repris ; les supprimer respecte aussi la règle « les données "
            "sont écrites par le modèle ouvert ». Les interdits tournants ne sont plus tirés à part par le code : "
            "une fiche en porte au plus un, choisi par le compositeur parmi ceux qui ne la contredisent pas "
            f"({nb_interdits['reponses']} côté réponses, {nb_interdits['messages']} côté messages, jamais côté "
            "messages pour le bloc « essentiels »). Le filtre copie_exemple ne compare donc plus les tours "
            "qu'aux vraies répliques de Molière. Aucun filtre ne vérifie un interdit : ce sont des incitations, "
            "à compter après coup par l'id de fiche."),
        "exemples_moliere": exemples,
        "exemples_moliere_justification": (
            f"Vraies répliques de Molière héritées du fichier d'origine, dont on ne garde que les propres : "
            f"{len(exemples)} sur {len(base['exemples_moliere'])}. Écartées (première raison seulement) : "
            + (", ".join(f"{k} {v}" for k, v in exemples_ecartes.items()) or "aucune")
            + ". Règles dans tools/composer_consignes_v4.py : pièce tenue à l'écart pour la mesure, lettre "
            "majuscule isolée sans accent (recopiée, elle donne « meta »), didascalie, graphie ancienne, "
            "patois, ponctuation fautive, vers reconnaissable à sa majuscule interne (le personnage doit "
            "parler en prose), bienséance, tournure banale d'aujourd'hui, et toute réplique dont une suite de "
            "six mots n'est faite que de mots très courants (liste fermée MOTS_COURANTS) : le filtre "
            "copie_exemple interdit en effet chaque suite de six mots de chaque réplique gardée à toutes les "
            "conversations, messages de la personne compris."),
        "personas_exclus": base["personas_exclus"],
        "personas_exclus_justification": base.get("personas_exclus_justification"),
        "interlocuteurs": base["interlocuteurs"],
        "genres": base["genres"],
        "interlocuteurs_justification": base.get("interlocuteurs_justification"),
        "juge": base["juge"],
        "juge_justification": base.get("juge_justification"),
        "juge_mesure": base.get("juge_mesure"),
        "juge_v4_justification": (
            f"Configuration du juge héritée (actif pour le style moliere, température 0, 20 tokens). Gabarit "
            f"« {jug['id']} » propre à ces consignes : la question 1 du juge d'origine exigeait des « phrases "
            "complètes » partout, la question 2 que chaque tour réponde « vraiment » au précédent « sans rien "
            "d'absurde » ; elles condamnaient ce que les fiches demandent à la personne (messages sans verbe, "
            "télégraphiques ou vagues, changement de sujet, français parlé). Ici la question 1 ne juge que le "
            "français des réponses du personnage et la question 2 laisse la personne libre de changer de sujet "
            "ou d'être vague. La question 3 est celle d'origine. Le juge est joué par le modèle générateur "
            "lui-même : « juge_justification » et « juge_mesure » décrivent des essais faits avec le modèle et "
            "le gabarit du fichier d'origine, pas avec ceux-ci. CE JUGE N'A PAS ENCORE ÉTÉ MESURÉ : à vérifier "
            "sur le lot pilote avant la campagne. Il voit mal les fautes de français : une relecture "
            "d'échantillon reste nécessaire."),
        "moderne_justification": (
            "Les clés du style « moderne » (gabarits.moderne, par_appel.moderne, num_predict.moderne, "
            "filtres.tours_moderne, champ « moderne » des registres) sont héritées telles quelles pour que le "
            "code ne casse pas. Mais « sujets » ne contient plus que des fiches du style « moliere », et les "
            "seuils « echo » assouplis valent pour les deux styles : ne pas générer de style « moderne » avec "
            "ce fichier."),
        "sujets_justification": (
            f"{len(fiches)} fiches d'échange composées par tirage équilibré (graine {graine}) à partir de "
            f"{len(themes)} notes de thème et {nb_types} types de message. "
            f"{bilan['essentiels']['fiches']} fiches ({100 * bilan['essentiels']['part']:.0f} %) vont au bloc "
            f"« essentiels » ({bilan['essentiels']['notes']} notes : saluer, demander des nouvelles, demander au "
            f"personnage qui il est, dire comment on va, parler du temps, remercier, prendre congé), soit "
            f"{bilan['essentiels']['fiches_par_note']['min']} fiches ou plus par note, sans détail imposé ni "
            "interdit côté messages. Table de compatibilité thème / type de message : un type n'est croisé "
            "qu'avec les groupes de thèmes qu'il nomme ; les formes et les déroulés ont aussi leurs exclusions "
            "par type. Appariement message -> réponse : la fiche tire UNE façon de répondre dans la liste propre "
            "au type de message. Une fiche porte au plus un supplément. Longueur d'une fiche : médiane "
            f"{car['mediane']} caractères, maximum {car['max']} ({100 * car['part_dans_l_objectif']:.0f} % des "
            f"fiches en {car['objectif']} caractères au plus). L'id code les valeurs tirées (voir « composition » et les id de "
            "chatbot/axes_consignes_v4.json), ce qui permet de compter après coup ce que les filtres ont gardé "
            "par axe (tools/campagne_synthese.py bilan). Le champ « theme » est la famille du thème. Aucune "
            "fiche ne contient de mot de troisième personne : le rejet « recopie_sujet » est sans prise. Limite "
            "connue : le code tire la fiche de chaque appel AVEC remise ; sur 13 500 appels, près de la moitié "
            "reprennent une fiche déjà sortie, avec un autre personnage, une autre voix et une autre graine."),
        "sujets": fiches,
    }
    # Dernières vérifications : le prompt se construit avec ces clés comme le fait construire_appel.
    "\n".join(cfg["gabarits"]["moliere"]["lignes"]).format(**NOMS_DU_GABARIT)
    if len({f["id"] for f in fiches}) != len(fiches) or len({f["texte"] for f in fiches}) != len(fiches):
        erreur("fiches en double")
    # Identité des consignes : l'empreinte de tout le fichier entre dans l'id du gabarit.
    marque = empreinte_des_consignes(cfg)
    cfg["composition"]["empreinte"] = marque
    cfg["gabarits"]["moliere"]["id"] = f"{gab['id']}-{marque}"
    return cfg


def ecrire(cfg, sortie):
    """Écriture atomique, en UTF-8, fins de ligne « \\n » : mêmes octets sur toute machine."""
    sortie = Path(sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    octets = (json.dumps(cfg, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    tmp = sortie.with_name(sortie.name + ".tmp")
    tmp.write_bytes(octets)
    os.replace(tmp, sortie)
    return len(octets)


def main(argv=None):
    for flux in (sys.stdout, sys.stderr):  # consoles Windows : éviter une erreur d'encodage à l'affichage
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modele", required=True, help="étiquette Ollama du générateur à poids ouverts")
    ap.add_argument("--licence", required=True, help="licence des poids du générateur (texte)")
    ap.add_argument("--sortie", default=str(SORTIE), help="fichier de consignes à écrire")
    ap.add_argument("--fiches", type=int, default=9000, help="nombre de fiches d'échange")
    ap.add_argument("--graine", type=int, default=GRAINE)
    ap.add_argument("--axes", default=str(AXES), help="axes de la campagne (lecture seule)")
    ap.add_argument("--origine", default=str(ORIGINE), help="fichier de consignes d'origine (lecture seule)")
    a = ap.parse_args(argv)
    if a.fiches < 1:
        erreur("--fiches doit être positif")
    if not a.modele.strip() or not a.licence.strip():
        erreur("--modele et --licence ne peuvent pas être vides")
    sortie = Path(a.sortie).resolve()
    if sortie in (Path(a.axes).resolve(), Path(a.origine).resolve(), ORIGINE.resolve(), AXES.resolve()):
        erreur("--sortie écraserait un fichier d'entrée")
    if sortie.suffix != ".json":
        erreur("--sortie doit finir par .json")
    cfg = composer(a.modele.strip(), a.licence.strip(), a.fiches, a.graine, a.axes, a.origine)
    taille = ecrire(cfg, sortie)
    c = cfg["composition"]
    t = c["tirage"]
    print(f"écrit : {sortie} ({taille} octets, sha256 {sha256(sortie)[:16]})")
    print(f"gabarit {cfg['gabarits']['moliere']['id']} ; juge {cfg['gabarits']['juge']['id']} ; "
          f"générateur {a.modele} ({a.licence}) ; graine {a.graine}")
    print(f"{c['fiches']} fiches ; {c['notes_de_theme']} notes de thème en {c['familles_de_themes']} familles ; "
          f"{c['types_de_message']} types de message ; {c['voix']} voix ; interdits {c['interdits']}")
    print("essentiels : " + json.dumps(t["essentiels"], ensure_ascii=False))
    print("hors essentiels, fiches par note : " + json.dumps(t["fiches_par_note_hors_essentiels"], ensure_ascii=False))
    print("caractères par fiche : " + json.dumps(t["caracteres_par_fiche"], ensure_ascii=False))
    print(f"vraies répliques gardées : {c['vraies_repliques_gardees']} ; écartées : {c['vraies_repliques_ecartees']}")
    print("fiches par nombre d'allers-retours : " + json.dumps(t["allers_retours"], ensure_ascii=False))
    print("fiches par supplément : " + json.dumps(t["supplement"], ensure_ascii=False))
    print("fiches par longueur : " + json.dumps(t["longueur"], ensure_ascii=False))
    print("fiches par type de message : " + json.dumps(t["type"], ensure_ascii=False))
    print("fiches par forme : " + json.dumps(t["forme"], ensure_ascii=False))


if __name__ == "__main__":
    main()
