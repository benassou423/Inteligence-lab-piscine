"""Nettoie le théâtre d'affinage (theatre_train, moliere_train, moliere_val) et écrit de NOUVEAUX fichiers.

RÉSULTAT DE L'ESSAI (14/11/2026) : NÉGATIF. Le modèle publié est affiné sur le théâtre D'ORIGINE, pas
sur ces fichiers nettoyés. Deux modèles affinés sur le théâtre nettoyé ont été jugés à l'aveugle sur
72 réponses : 34 et 37 sans défaut, contre 44 pour la référence, et leur BPB sur la validation Molière
est moins bon (1,0628 contre 1,0560). Explication probable : avec nettement moins de texte de théâtre,
le modèle sur-apprend plus tôt. L'outil est gardé pour que l'essai puisse être refait.

Les fichiers d'origine (data/stages/chatbot_20261112/theatre) ne sont jamais modifiés, et aucun mot
n'est ajouté ni réécrit : on ne fait que RETIRER (indications de scène, appels de note, marques de
mise en page, répliques chantées, scènes trop courtes, scènes qui ne sont pas en français, scènes
en double, œuvres écartées par la liste --exclusions). L'orthographe d'époque est gardée.

Une parenthèse n'est retirée que si elle ressemble à une indication de scène. Dans le doute elle est
GARDÉE : Molière met des parenthèses dans la bouche de ses personnages (« (car je sors de chez lui) »).

moliere_val (validation, jamais entraînée) reçoit le même nettoyage des répliques, sans qu'aucune
scène soit écartée : la mesure reste faite sur les mêmes scènes, dans la même forme que l'entraînement.

  python tools/nettoyer_affinage.py --exclusions chatbot/exclusions_theatre_v2.json \
      --dst data/stages/chatbot_20261114/theatre_propre [--controle 150]
"""
import argparse
import collections
import hashlib
import io
import json
import random
import re
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/stages/chatbot_20261112/theatre"
FICHIERS = ("theatre_train", "moliere_train", "moliere_val")

MOT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ]+")
# Mots-outils propres au français (pas « de », « la », « que », « est »… communs au latin, à l'italien, à l'espagnol).
FRANCAIS = frozenset("les des du dans pour pas vous nous je ne qui avec cette être au aux sur ce il elle ils on mais où à ai "
                     "suis êtes avez mon votre bien point cela une j n qu et un le vos mes ses son leur dont".split())
PARENTHESE = re.compile(r"\(([^()]{0,1500})\)")
CROCHET = re.compile(r"\[([^\[\]]{0,1500})\]")
# 1re ou 2e personne : le personnage parle. (« ton » : pas dans « d'un ton », « le ton »…)
PERSONNE = re.compile(r"\b(?:je|me|moi|mon|ma|mes|tu|te|toi|ta|tes|vous|votre|vos|vôtre|nous|notre|nos|nôtre)\b|\b[jmt]['’]"
                      r"|(?<!un )(?<!le )(?<!du )(?<!de )(?<!ce )\bton\b", re.I)
PAS_PARTICIPE = r"(?!pourtant|cependant|maintenant|avant|quant|tant|autant|devant|pendant|durant|suivant|néanmoins)"
DEBUT = re.compile(
    r"^(?:(?:à part|bas|haut|seule?s?|silence|pause|même jeu|un temps|un silence|une pause)(?:$|[ .,;:])"
    r"|[àa] (?:(?-i:[A-ZÀ-Ý])|lui-même|elle-même|soi-même|voix basse|demi-voix|mi-voix)"
    r"|au (?:public|fond|parterre)\b|aux [a-zà-ÿ]+s\b|avec |d['’]une? (?:ton|air|voix)"
    r"|en " + PAS_PARTICIPE + r"[a-zà-ÿ]+ant\b|(?:se |s['’]|lui |leur )?" + PAS_PARTICIPE + r"[a-zà-ÿ]{3,}ant\b"
    r"|(?:il|elle|ils|elles) (?:sort|sortent|entre|entrent|rit|rient|pleure|se |s['’]|lui |va |vont |prend|donne|montre"
    r"|regarde|tire|lit|écrit|tombe|salue|baise|embrasse|frappe|ôte|met |remonte|descend|passe|gagne|vide|reste)"
    r"|[a-zà-ÿ]+ment(?:$|[.,;])|[a-zà-ÿ]+(?:é|ée|és|ées)(?:$|[.,;]))", re.I)
# Formule courte DITE, même avec une majuscule : « (Allons donc.) », « (Oui.) ».
FORMULE = re.compile(r"^(?:Allons|Oui|Non|Bien|Bon|Ah|Oh|Eh|Hé|Hélas|Quoi|Voyons|Tiens|Soit|Parbleu|Morbleu|Diable|Ma foi|Peste|Fi)\b[^.]{0,20}\.?$")
CHIFFRES = re.compile(r"^[\d\s.,;:—–-]*$")
# Appel ou note d'éditeur entre crochets : [A], [12], [1601-a], [loc4], [sic], [Note …], [1. …].
NOTE = re.compile(r"^(?:[A-Za-z]|\d+(?:-[A-Za-z])?|[a-z]{1,3}\d+|sic|Note\b.*|Var\b.*|\d+\..*)$", re.S)
NOMS_PROPRES = re.compile(r"^(?:[A-ZÀ-Ý][\w’'-]*(?:, | et | |$|\.$)){2,}$")
TROISIEME = re.compile(r"^[A-ZÀ-Ý][\w-]+ (?:se |s['’])[a-zà-ÿ]+")
QUEUE = r"(?! d['’]ici| de chez| de là| du | de | au couvent)[^.!?…;,\n]{0,60}\.?"
SORTIE = r"(?:Il|Elle|Ils|Elles|Tous|Toutes) (?:sort|sortent|entre|entrent|rentre|rentrent|s[’']en va|s[’']en vont|se retire|se retirent)\b" + QUEUE
SORTIE_LIGNE = re.compile(r"\n[ \t]*(" + SORTIE + r")[ \t]*$")
SORTIE_FIN = re.compile(r"(?<=[.!?…»]) (" + SORTIE + r")[ \t]*$")
MARQUES_AVANT = re.compile(r"'{2,}|^[ \t]*:+[ \t]*|(?<![A-Za-zÀ-ÿ])_|_(?![A-Za-zÀ-ÿ])|(?:=[ \t]*){2,}|\{\{[A-Za-zé]+\||\{\{[A-Za-zé]+\}\}?|\}\}", re.M)
SAUT_HTML = re.compile(r"[ \t]*<\s*br\s*/?\s*>[ \t]*")
ORPHELINS = re.compile(r"[\[\]{}]")
SUSPENSION = re.compile(r"(…|\.\.\.)[ \t]*\x00[ \t]*(?:…|\.\.\.)")
POINT_ORPHELIN = re.compile(r"(^|[.!?…,;:—»])([ \t]*)\x00[ \t]*\.(?!\.)", re.M)
COLLE = re.compile(r"[ \t]*\x00[ \t]*(?=[,.](?!\.))")
ESPACES = re.compile(r"[ \t]+")
VIDE = re.compile(r"^[\W_]*$")
CHANTE = re.compile(r"CH[ŒO]E?URS?\b|MUSICIEN|CHANTEU", re.I)
RESIDU = re.compile(r"[()\[\]]")


def lire(p):
    with open(p, "rb") as f:
        for l in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            if l.strip():
                yield json.loads(l)


def ecrire(p, docs):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as f, zstandard.ZstdCompressor(level=10).stream_writer(f) as z:
        for d in docs:
            z.write((json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8"))
    tmp.replace(p)


def est_didascalie(c, crochet=False):
    """Vrai si le contenu d'une parenthèse (ou d'un crochet) est une indication de scène. Dans le doute : faux."""
    c = c.strip()
    if not c or CHIFFRES.match(c):
        return True
    if "!" in c or "?" in c or PERSONNE.search(c):
        return False
    if DEBUT.match(c):
        return True
    if crochet:  # crochets d'éditeur : souvent une variante DITE, même avec une majuscule
        return bool(NOMS_PROPRES.match(c) or TROISIEME.match(c))
    return c[0].isupper() and not FORMULE.match(c)


def crochets_didascalies(scenes):
    """Vrai si, dans cette œuvre, les crochets servent d'indications de scène (et non de variantes d'éditeur)."""
    oui = total = 0
    for d in scenes:
        for t in d["turns"]:
            for c in CROCHET.findall(t["text"]):
                if not NOTE.match(c.strip()):
                    total += 1
                    oui += est_didascalie(c)
    return total >= 5 and oui / total >= 0.6


def nettoyer_replique(t, compte, crochets_scene=False):
    """Retire les indications de scène et les marques de mise en page d'une réplique. Ne réécrit rien."""
    def parenthese(m):
        if est_didascalie(m.group(1)):
            compte["parenthese_retiree"] += 1
            return "\x00"
        compte["parenthese_dite_gardee"] += 1
        return "\x01" + m.group(1) + "\x02"  # gardée : protégée du second passage, rétablie plus bas

    def crochet(m):
        c = m.group(1).strip()
        if NOTE.match(c):
            compte["appel_de_note_retire"] += 1
            return "\x00"
        if est_didascalie(c, crochet=not crochets_scene):
            compte["crochet_retire"] += 1
            return "\x00"
        compte["crochet_ouvert"] += 1  # variante d'éditeur : on garde le texte, sans les crochets
        return m.group(1)
    t, n = MARQUES_AVANT.subn("", t)
    t, n2 = SAUT_HTML.subn("\n", t)
    compte["marque_de_mise_en_page"] += n + n2
    for _ in range(2):  # deux niveaux d'imbrication
        t = PARENTHESE.sub(parenthese, t)
    t = CROCHET.sub(crochet, t)
    t = "\n".join(l.rstrip() for l in t.split("\n")).rstrip()
    for motif, nom in ((SORTIE_LIGNE, "sortie_en_ligne_finale"), (SORTIE_FIN, "sortie_en_phrase_finale")):
        m = motif.search(t)
        if m and not PERSONNE.search(m.group(1)):
            t = t[:m.start()]
            compte[nom] += 1
    t = SUSPENSION.sub(r"\1 ", t)
    t = COLLE.sub("", POINT_ORPHELIN.sub(r"\1\2", t)).replace("\x00", " ")
    t, n = ORPHELINS.subn("", t)
    compte["marque_de_mise_en_page"] += n
    t = t.replace("\x01", "(").replace("\x02", ")")
    return "\n".join(l.strip() for l in ESPACES.sub(" ", t).split("\n") if l.strip())


def part_francais(turns):
    w = [x.lower() for t in turns for x in MOT.findall(t["text"])]
    return (sum(1 for x in w if x in FRANCAIS) / len(w), len(w)) if w else (0.0, 0)


def nettoyer_scene(d, seuil_fr, min_repliques, crochets_scene, validation=False):
    """(scène nettoyée ou None, raison de l'écart, compteurs, exemples avant/après)."""
    compte, exemples = collections.Counter(), []
    chantees = 0 if validation else sum(1 for t in d["turns"] if CHANTE.search(t["speaker"]))
    if chantees and chantees > 0.3 * len(d["turns"]):
        return None, "intermede_chante", compte, exemples
    turns = []
    retraits = lambda: sum(v for k, v in compte.items() if k != "parenthese_dite_gardee")
    for t in d["turns"]:
        if chantees and CHANTE.search(t["speaker"]):
            compte["repliques_chantees_retirees"] += 1
            continue
        avant = retraits()
        texte = nettoyer_replique(t["text"], compte, crochets_scene)
        if retraits() > avant:
            compte["repliques_modifiees"] += 1
            exemples.append({"avant": t["text"], "apres": texte})
        if VIDE.match(texte):
            compte["repliques_vides_retirees"] += 1
            continue
        if turns and turns[-1]["speaker"] == t["speaker"]:
            turns[-1]["text"] += "\n" + texte
            compte["repliques_fusionnees_meme_personnage"] += 1
        else:
            turns.append({"speaker": t["speaker"], "text": texte})
    if not validation:
        if len(turns) < min_repliques or len({t["speaker"] for t in turns}) < 2:
            return None, "scene_trop_courte", compte, exemples
        part, n = part_francais(turns)
        if n >= 80 and part < seuil_fr:
            return None, "pas_en_francais", compte, exemples
    compte["repliques_avec_parenthese_ou_crochet_restant"] = sum(1 for t in turns if RESIDU.search(t["text"]))
    return dict(d, turns=turns), None, compte, exemples


def scenes_en_double(scenes, seuil=0.3):
    """Indices des scènes à écarter : dans une œuvre dont au moins `seuil` des répliques longues figurent dans une
    œuvre plus grosse, les scènes dont la moitié au moins des répliques longues y figurent. (indices, paires)."""
    norm = lambda s: "".join(MOT.findall(s.lower()))
    longues = collections.defaultdict(set)
    for d in scenes:
        for t in d["turns"]:
            if len(t["text"]) >= 120:
                longues[d["doc_id"]].add(norm(t["text"]))
    ou = collections.defaultdict(set)
    for doc, s in longues.items():
        for x in s:
            ou[x].add(doc)
    commun = collections.Counter()
    for ids in ou.values():
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                commun[(a, b)] += 1
    paires = {}
    for (a, b), n in commun.items():
        petit, gros = (a, b) if len(longues[a]) <= len(longues[b]) else (b, a)
        if len(longues[petit]) >= 10 and n / len(longues[petit]) >= seuil:
            paires[petit] = gros
    retirer = set()
    for i, d in enumerate(scenes):
        gros = paires.get(d["doc_id"])
        if gros:
            l = [norm(t["text"]) for t in d["turns"] if len(t["text"]) >= 120]
            if l and sum(x in longues[gros] for x in l) >= 0.5 * len(l):
                retirer.add(i)
    return retirer, paires


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exclusions", default=None, help="JSON {documents: {doc_id: raison}} (œuvres écartées de theatre_train)")
    p.add_argument("--dst", required=True)
    p.add_argument("--seuil-francais", type=float, default=0.16, help="part minimale de mots-outils propres au français")
    p.add_argument("--min-theatre", type=int, default=4, help="répliques minimales d'une scène de theatre_train")
    p.add_argument("--min-moliere", type=int, default=2, help="répliques minimales d'une scène de Molière (texte rare)")
    p.add_argument("--controle", type=int, default=0, help="nombre de répliques modifiées à tirer pour relecture")
    p.add_argument("--graine", type=int, default=20261114)
    p.add_argument("--sans-ecrire", action="store_true", help="afficher les comptes, ne rien écrire")
    a = p.parse_args()
    dst = Path(a.dst)
    if dst.resolve() == SRC.resolve():
        p.error("--dst doit être un autre dossier que l'origine")
    exclus = json.loads(Path(a.exclusions).read_text(encoding="utf-8"))["documents"] if a.exclusions else {}
    rng = random.Random(a.graine)
    rapport = {"exclusions": a.exclusions, "seuil_francais": a.seuil_francais, "graine": a.graine,
               "exclusions_sha256": hashlib.sha256(Path(a.exclusions).read_bytes()).hexdigest() if a.exclusions else None,
               "min_repliques": {"theatre_train": a.min_theatre, "moliere_train": a.min_moliere}, "fichiers": {}}
    controle = {}
    for nom in FICHIERS:
        theatre, validation = nom == "theatre_train", nom == "moliere_val"
        source = SRC / f"{nom}.jsonl.zst"
        docs = list(lire(source))
        par_oeuvre = collections.defaultdict(list)
        for d in docs:
            par_oeuvre[d["doc_id"]].append(d)
        mode_crochets = {k: crochets_didascalies(v) for k, v in par_oeuvre.items()}
        ecart, vus, gardes, ecartes_ex = collections.Counter(), set(), [], collections.defaultdict(list)
        for d in docs:
            if theatre and d["doc_id"] in exclus:
                ecart["oeuvre_ecartee:" + exclus[d["doc_id"]]] += 1
                continue
            propre, raison, c, ex = nettoyer_scene(d, a.seuil_francais, a.min_theatre if theatre else a.min_moliere,
                                                   mode_crochets[d["doc_id"]], validation)
            if propre is None:
                ecart[raison] += 1
                if len(ecartes_ex[raison]) < 400:
                    ecartes_ex[raison].append(d)
                continue
            cle = hashlib.sha1("\x1f".join(t["speaker"] + "\x1e" + t["text"] for t in propre["turns"]).encode()).digest()
            if cle in vus and not validation:
                ecart["scene_en_double"] += 1
                continue
            vus.add(cle)
            gardes.append((propre, c, ex))
        paires = {}
        if theatre:
            retirer, paires = scenes_en_double([g[0] for g in gardes])
            ecart["scene_d_une_oeuvre_en_double"] = len(retirer)
            gardes = [g for i, g in enumerate(gardes) if i not in retirer]
        compte, exemples = collections.Counter(), []
        for _, c, ex in gardes:  # seulement ce qui est écrit
            compte.update(c)
            exemples.extend(ex)
        scenes = [g[0] for g in gardes]
        rapport["fichiers"][nom] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scenes_lues": len(docs), "scenes_gardees": len(scenes), "scenes_ecartees": dict(ecart),
            "repliques_lues": sum(len(d["turns"]) for d in docs), "repliques_gardees": sum(len(d["turns"]) for d in scenes),
            "mots_lus": sum(len(t["text"].split()) for d in docs for t in d["turns"]),
            "mots_gardes": sum(len(t["text"].split()) for d in scenes for t in d["turns"]),
            "retraits_dans_les_repliques": dict(compte),
            "oeuvres_a_crochets_didascalies": sorted(k for k, v in mode_crochets.items() if v), "oeuvres_en_double": paires}
        print(nom, json.dumps({k: v for k, v in rapport["fichiers"][nom].items() if not k.startswith("oeuvres_")}, ensure_ascii=False))
        if not a.sans_ecrire:
            dst.mkdir(parents=True, exist_ok=True)
            ecrire(dst / f"{nom}.jsonl.zst", scenes)
            rapport["fichiers"][nom]["sha256"] = hashlib.sha256((dst / f"{nom}.jsonl.zst").read_bytes()).hexdigest()
        controle[nom] = {"repliques_modifiees": rng.sample(exemples, min(a.controle, len(exemples))),
                         "scenes_ecartees": {r: rng.sample(v, min(12, len(v))) for r, v in ecartes_ex.items()},
                         "scenes_propres": rng.sample(scenes, min(40, len(scenes)))}
    if a.sans_ecrire:
        return
    (dst / "rapport_nettoyage.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.controle:
        (dst / "controle_nettoyage.json").write_text(json.dumps(controle, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
