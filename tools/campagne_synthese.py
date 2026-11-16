"""Lanceur de la campagne de données de synthèse « fiches » du chatbot Molière (campagnes v4 et v5).

Les conversations retenues ont été écrites par gemma4:31b-it-q4_K_M, servi par Ollama. Les deux
campagnes passent par ce lanceur ; seul le fichier de consignes change (--consignes) :
chatbot/prompts_synthetiques_v4.json pour la v4, chatbot/prompts_synthetiques_v5.json pour la v5
(axes : chatbot/axes_consignes_v5.json). qwen3:8b n'a servi qu'à un premier essai, non retenu.

À lancer depuis la racine du clone, avec .venv/Scripts/python.exe et PYTHONUTF8=1 :

    python tools/campagne_synthese.py apercu    --consignes F.json [--seed 201] [--n 3]
    python tools/campagne_synthese.py generer   --consignes F.json --n N --seed 201 --out DOSSIER/moliere_s201.jsonl.zst
                                                [--max-appels M] [--port 11434]
    python tools/campagne_synthese.py bilan     --consignes F.json --motif 'DOSSIER/moliere_s*.jsonl.zst'
    python tools/campagne_synthese.py fusion    --consignes F.json --motif 'DOSSIER/moliere_s*.jsonl.zst' --dst NOUVEAU_DOSSIER
                                                [--ecart FICHIER_TENU_A_L_ECART.jsonl.zst]
    python tools/campagne_synthese.py refiltrer --consignes F.json --fichier NOUVEAU_DOSSIER/moliere.jsonl.zst

(option générale, avant la sous-commande : --sans-torch, ou la variable NUIT_SANS_TORCH=1, si
« import torch » est refusé par Windows ; c'est tools/nuit_moliere.py qui s'en charge.)

LA CHAÎNE
1. tools/composer_consignes_v4.py croise les axes de chatbot/axes_consignes_v4.json (v4) ou de
   chatbot/axes_consignes_v5.json (v5, option --axes) et écrit un
   fichier de consignes complet (F.json) : modèle à poids ouverts et licence, gabarit du prompt,
   fiches d'échange (« sujets »), voix (« registres »), filtres, juge.
2. « apercu » construit et affiche des prompts, sans rien appeler : à relire avant de générer.
3. « generer » fait écrire les conversations par le modèle ouvert servi par Ollama (une fiche, donc
   une conversation, par appel), avec chatbot.synthetic.generer tel quel : mêmes filtres, même
   anti-doublon, même juge, même reprise si on relance avec le même --out. Un fichier par graine ;
   plusieurs serveurs Ollama = plusieurs ports = plusieurs processus.
4. « bilan » compte, pour des fichiers par graine, les appels et les conversations gardées par
   valeur d'axe (bloc essentiels, type de message, forme, déroulé, longueur, supplément, voix,
   personnage), le respect du nombre d'allers-retours demandé, et les rejets des filtres par forme
   et par longueur : c'est la lecture du lot pilote (quelles fiches les filtres défavorisent).
5. « fusion » réunit les fichiers par graine avec la fusion de tools/nuit_moliere.py (fuites vers
   les pièces de validation, personnages de ces pièces, conversations sans fiche, réponses sans
   fin), dans un dossier NEUF.
6. « refiltrer » réapplique filtres et anti-doublon au fichier fusionné, avec les MÊMES consignes
   (le plafond des formules se recalcule sur l'ensemble). Le plafonnement de la part de synthèse
   reste : python tools/nuit_moliere.py plafonner --fichier ... --max ...

CE QUE CE LANCEUR BRANCHE, EN MÉMOIRE SEULEMENT (aucun fichier existant du dépôt n'est modifié)
- le fichier de consignes : chargé par chatbot.synthetic.load_config(chemin) et passé partout par
  le paramètre cfg ; pour la fusion, qui appelle load_config() sans argument, la fonction est
  remplacée le temps de l'appel (changer synthetic.PROMPTS_PATH ne servirait à rien : la valeur par
  défaut est figée à la définition) ;
- l'adresse du serveur : cfg["generateur"]["url"] reçoit le port demandé ;
- synthetic.SOURCE et synthetic.LICENSE, qui nomment en dur un autre modèle : remplacés d'après le
  modèle (generateur.model) et la licence (licence_generateur) du fichier de consignes ; ils
  entrent dans « id », « source » et « license » de chaque conversation ;
- les options d'Ollama du fichier (options_ollama : num_ctx, top_p, top_k, repeat_penalty, stop),
  que synthetic.py n'envoie pas de lui-même : ajoutées à chaque appel de génération par le
  paramètre « generateur » de synthetic.generer. Le juge reste tel que configuré : il ne reçoit en
  plus que num_ctx, pour que le serveur ne recharge pas le modèle entre un appel et sa relecture ;
- synthetic.raison_rejet, le temps de « generer » : la fonction d'origine est appelée telle quelle
  et son verdict rendu sans changement ; le lanceur note seulement chaque rejet (raison, fiche,
  voix, personnage, numéro d'appel) dans « <sortie>.rejets.jsonl », parce que le rapport de
  synthetic.py ne garde que des totaux et cinq exemples par raison.

GARDE-FOUS
- rien n'est écrit dans data/stages/chatbot_20261113/ (sorties de la campagne précédente) ;
- IDENTITÉ DES CONSIGNES : « generer » pose à côté de la sortie un fichier « .campagne.json » (chemin
  et empreinte SHA-256 du fichier de consignes, modèle, licence, graine, options envoyées). Il
  n'est écrit qu'après tous les contrôles, par fichier temporaire. Reprendre un --out existant
  exige ce fichier, la même empreinte et la même graine : des consignes recomposées (autres fiches,
  autre graine, autre température, autres filtres, autres options) sont refusées, même si le modèle
  est le même. L'id du gabarit porte en plus 8 caractères d'empreinte des consignes composées,
  recalculés ici : un fichier de consignes retouché à la main est refusé ;
- « fusion » refuse un --dst existant non vide (elle y écrit toujours moliere.jsonl.zst,
  moliere.rapport.json et moliere.journal.log) et tout fichier dont le « .campagne.json » manque ou
  porte une autre empreinte ; elle écrit dans --dst un « moliere.campagne.json » (consignes,
  empreinte, options, fichiers réunis) ;
- « refiltrer » exige ce « .campagne.json » et la même empreinte ; il n'emploie jamais
  « python -m chatbot.synthetic --refiltrer », qui relirait toujours le fichier de consignes
  d'origine ; il écarte d'abord les conversations sans fiche (second bloc écrit par le modèle), qui
  feraient planter le refiltrage, et celles qui citent le titre d'une des trois pièces tenues à
  l'écart pour la mesure (la fusion n'écarte que les noms de rôles).

Ce script n'écrit aucune conversation : toutes viennent du modèle ouvert.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))  # racine du clone : chatbot, pipeline, tools

STYLE = "moliere"
PROTEGE = RACINE / "data" / "stages" / "chatbot_20261113"
GRAINES_PRISES = set(range(101, 125)) | {900}  # campagne précédente (synthetique_parts)
OPTIONS_DU_JUGE = ("num_ctx",)                  # seule option ajoutée aux appels du juge
# Titres des trois pièces tenues à l'écart pour la mesure, écrits avec leur majuscule (« l'avare » ou
# « misanthrope » en nom commun ne sont pas des titres). La fusion de tools/nuit_moliere.py n'écarte
# que les noms de rôles.
TITRES_A_L_ECART = re.compile(
    r"\b(?:[Ll]e|[Dd]u|[Aa]u)\s+(?:Misanthrope|Sicilien)\b|\b[Ll]['’]\s?Avare\b|\bAmour\s+peintre\b")
# id d'une fiche : v4-<n°>-t<thème>-<type><n° de la façon de répondre>-<forme>-<déroulé>-<longueur>-<supplément>
ID_DE_FICHE = re.compile(r"^v4-\d+-t(\d+)-([a-z]+)(\d+)-(\w+)-(\w+)-(\w+)-(\w+)$")
SORTES_DE_SUPPLEMENT = {"d": "detail", "e": "entree", "r": "interdit_reponse", "m": "interdit_message", "x": "aucun"}


def erreur(message):
    sys.exit("ERREUR : " + message)


def dit(message):
    print(message, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ garde-fous sur les chemins
def dans(chemin, dossier):
    a = os.path.normcase(os.path.realpath(chemin))
    b = os.path.normcase(os.path.realpath(dossier))
    return a == b or a.startswith(b + os.sep)


def refuse_protege(chemin, option):
    if dans(chemin, PROTEGE):
        erreur(f"{option} = {chemin} : ce lanceur n'écrit rien dans {PROTEGE} "
               "(sorties de la campagne précédente) ; choisir un autre dossier")


def exige_zst(chemin, option):
    if not str(chemin).endswith(".jsonl.zst"):
        erreur(f"{option} doit finir par .jsonl.zst")


def a_cote(out, suffixe):
    """Fichier voisin d'une sortie : DOSSIER/x.jsonl.zst -> DOSSIER/x<suffixe>."""
    out = Path(out)
    return out.with_name(out.name[:-len(".jsonl.zst")] + suffixe)


# ------------------------------------------------------------------ identité des consignes
def empreinte_fichier(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def lire_trace(out):
    chemin = a_cote(out, ".campagne.json")
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None


def ecrire_trace(out, trace):
    """Écriture atomique : fichier temporaire puis os.replace."""
    chemin = a_cote(out, ".campagne.json")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.name + ".tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, chemin)


def chemin_publiable(chemin):
    """Chemin écrit dans les traces « .campagne.json », qui accompagnent les données publiées : relatif
    à la racine du dépôt (jamais le dossier personnel de la machine) ; hors du dépôt, le nom du fichier
    seul. L'identité des consignes tient à leur empreinte SHA-256, pas à ce chemin."""
    p = Path(chemin).resolve()
    try:
        return p.relative_to(RACINE).as_posix()
    except ValueError:
        return p.name


def exige_memes_consignes(out, empreinte, consignes):
    """Le « .campagne.json » d'une sortie existe et porte l'empreinte de ces consignes. Rend la trace."""
    trace = lire_trace(out)
    if trace is None:
        erreur(f"{out} existe sans son fichier « .campagne.json » : impossible de vérifier avec quelles "
               "consignes il a été écrit ; prendre un autre nom de fichier")
    if trace.get("consignes_sha256") != empreinte:
        erreur(f"{out} a été écrit avec d'autres consignes ({trace.get('consignes')}, empreinte "
               f"{str(trace.get('consignes_sha256'))[:12]}) que {consignes} (empreinte {empreinte[:12]}) : "
               "ne pas mélanger (prendre un autre nom de fichier ou les consignes d'origine)")
    return trace


def meme_modele_et_gabarit(rapport, cfg, fichier):
    """Second contrôle, sur le rapport de synthetic.py : même modèle et même gabarit (l'id du gabarit
    porte l'empreinte des consignes composées)."""
    attendu = (cfg["generateur"]["model"], cfg["gabarits"][STYLE]["id"])
    trouve = (rapport.get("model"), rapport.get("gabarit"))
    if trouve != attendu:
        erreur(f"{fichier} a été écrit avec modèle/gabarit {trouve}, ces consignes disent {attendu} : "
               "ne pas mélanger (prendre un autre nom de fichier ou les bonnes consignes)")


# ------------------------------------------------------------------ branchement en mémoire
def charger_modules(sans_torch):
    """(chatbot.synthetic, tools.nuit_moliere). PyTorch n'est utile ici que par ricochet."""
    import tools.nuit_moliere as nm
    if sans_torch:
        nm.sans_torch()  # AVANT d'importer chatbot.synthetic
    try:
        from chatbot import synthetic as s
    except OSError as e:  # par exemple WinError 4551 : chargement de torch_cpu.dll refusé par le système
        erreur(f"« import torch » a échoué ({e}) : relancer avec --sans-torch avant la sous-commande")
    return s, nm


def charger_consignes(s, chemin):
    chemin = Path(chemin)
    if not chemin.is_file():
        erreur(f"--consignes : fichier introuvable : {chemin}")
    cfg = s.load_config(chemin)
    verifier(s, cfg, chemin)
    return cfg


def verifier(s, cfg, chemin):
    """Ce que load_config ne valide pas, et qui ferait planter ou tout rejeter en cours de route."""
    def exige(condition, message):
        if not condition:
            erreur(f"{chemin} : {message}")

    for cle in ("generateur", "par_appel", "num_predict", "registres", "gabarits", "sujets", "exemples_moliere",
                "nb_exemples_par_prompt", "demonstrations_moliere", "nb_demonstrations_par_prompt",
                "interlocuteurs", "genres", "filtres", "licence_generateur"):
        exige(cle in cfg, f"clé « {cle} » absente")
    for cle in ("model", "url", "temperature", "think"):
        exige(cle in cfg["generateur"], f"generateur.{cle} absent")
    exige(str(cfg["licence_generateur"]).strip(), "licence_generateur vide")
    for cle in ("tours_moliere", "mots_anglais", "max_mots_par_tour", "echo_max", "echo_max_utilisateur",
                "tour_repete_min_mots", "mots_modernes_reponse_moliere", "copie_exemple_ngramme",
                "max_conversations_par_5gramme"):
        exige(cle in cfg["filtres"], f"filtres.{cle} absent")
    exige(cfg["filtres"]["mots_modernes_reponse_moliere"],
          "filtres.mots_modernes_reponse_moliere est vide : toutes les conversations seraient rejetées")
    gab = cfg["gabarits"].get(STYLE) or {}
    for cle in ("id", "lignes", "ligne_sujet", "ligne_exemple", "ligne_demonstration"):
        exige(cle in gab, f"gabarits.{STYLE}.{cle} absent")
    exige("#" not in gab["id"], "l'id du gabarit contient « # » (la reprise coupe le prompt_id sur ce signe)")
    exige("juge" in cfg["gabarits"], "gabarits.juge absent")
    k = cfg["par_appel"][STYLE]
    exige(len(cfg["sujets"]) >= k, "moins de fiches que par_appel")
    exige(len({x["id"] for x in cfg["sujets"]}) == len(cfg["sujets"]), "id de fiche en double")
    exige(len({x["texte"] for x in cfg["sujets"]}) == len(cfg["sujets"]), "texte de fiche en double")
    exige(cfg["nb_demonstrations_par_prompt"] <= len(cfg["demonstrations_moliere"]),
          "nb_demonstrations_par_prompt dépasse la liste (rng.sample planterait)")
    exige(all(r["id"].startswith(("tu", "vous")) and "moliere" in r for r in cfg["registres"]),
          "un registre sans champ « moliere » ou dont l'id ne commence ni par « tu » ni par « vous »")
    exclus = set(cfg.get("personas_exclus", []))
    manquants = sorted({p["interlocutor"] for p in s.PERSONAS if p["name"] not in exclus}
                       - set(cfg["interlocuteurs"]))
    exige(not manquants, f"interlocuteurs sans genre : {manquants}")
    marque = (cfg.get("composition") or {}).get("empreinte")
    if marque:  # fichier écrit par tools/composer_consignes_v4.py : rien n'y a été retouché à la main
        from tools.composer_consignes_v4 import empreinte_des_consignes
        exige(empreinte_des_consignes(cfg) == marque and gab["id"].endswith("-" + marque),
              "l'empreinte des consignes ne correspond plus à leur contenu (fichier retouché à la main ?) : "
              "le recomposer avec tools/composer_consignes_v4.py")
    try:
        prompt, _ = s.construire_appel(STYLE, 0, 0, cfg)
    except (KeyError, ValueError, IndexError) as e:
        erreur(f"{chemin} : le prompt ne se construit pas ({e!r})")
    exige(not s.est_prompt_juge(prompt, cfg), "le prompt de génération est pris pour un prompt du juge")


def etiquettes(cfg):
    """(SOURCE, LICENSE) des conversations, d'après le modèle et la licence du fichier de consignes."""
    modele, licence = cfg["generateur"]["model"], str(cfg["licence_generateur"]).strip()
    source = "synthetique_" + re.sub(r"[^a-z0-9]+", "_", modele.lower()).strip("_")
    return source, f"synthétique : généré par {modele} (poids ouverts, {licence}), prompts publiés"


def fabrique_generateur(s, cfg, appel=None, avant_generation=None):
    """Fonction (prompt, options) -> réponse, pour le paramètre « generateur » de synthetic.generer.

    Les options d'Ollama du fichier s'ajoutent aux trois que synthetic.py fixe lui-même (temperature,
    num_predict, seed), qui gardent la priorité. Un prompt du juge ne reçoit que num_ctx.
    `avant_generation()` est appelée avant chaque appel de génération (pas avant ceux du juge).
    """
    appel = appel or s.appel_ollama
    extra = dict(cfg.get("options_ollama") or {})
    pour_le_juge = {k: extra[k] for k in OPTIONS_DU_JUGE if k in extra}

    def generateur(prompt, options):
        if s.est_prompt_juge(prompt, cfg):
            return appel(prompt, {**pour_le_juge, **options}, cfg)
        if avant_generation:
            avant_generation()
        return appel(prompt, {**extra, **options}, cfg)

    return generateur


def axes_de_la_fiche(fiche_id, cfg):
    """Valeurs d'axe codées dans l'id d'une fiche (None si ce n'est pas un id de ces consignes)."""
    m = ID_DE_FICHE.match(fiche_id or "")
    if not m:
        return None
    _, type_, rep, forme, deroule, longueur, supplement = m.groups()
    ar = ((cfg.get("composition") or {}).get("allers_retours_par_deroule") or {}).get(deroule)
    return {"bloc": "essentiels" if type_.startswith("x") else "autres", "type": type_,
            "facon_de_repondre": f"{type_}{rep}", "forme": forme, "deroule": deroule,
            "allers_retours": str(ar) if ar else "?", "longueur": longueur,
            "supplement": SORTES_DE_SUPPLEMENT.get(supplement[:1], "?"), "valeur_du_supplement": supplement}


# ------------------------------------------------------------------ sous-commandes
def generer(a, appel=None):
    """`appel` (facultatif) remplace synthetic.appel_ollama : sert aux essais sans serveur."""
    from pipeline.io import read_docs
    exige_zst(a.out, "--out")
    refuse_protege(a.out, "--out")
    if a.n < 1:
        erreur("--n doit être positif")
    s, _ = charger_modules(a.sans_torch)
    cfg = charger_consignes(s, a.consignes)
    cfg["generateur"]["url"] = f"http://127.0.0.1:{a.port}/api/generate"
    empreinte = empreinte_fichier(a.consignes)
    out, rapport_path, _ = s.chemins(a.out)
    journal_rejets = a_cote(out, ".rejets.jsonl")
    debut = 0  # numéro du premier appel de cette exécution (même calcul que la reprise de synthetic.generer)
    if out.exists():
        if not rapport_path.exists():
            erreur(f"{out} existe sans son rapport : impossible de vérifier d'où il vient ; prendre un autre --out")
        ancien = json.loads(rapport_path.read_text(encoding="utf-8"))
        meme_modele_et_gabarit(ancien, cfg, out)
        if (ancien.get("style"), ancien.get("seed")) != (STYLE, a.seed):
            erreur(f"{out} a été écrit avec style={ancien.get('style')} seed={ancien.get('seed')} : "
                   f"donner --seed {ancien.get('seed')} pour le reprendre, ou un autre --out")
        exige_memes_consignes(out, empreinte, a.consignes)
        dernier = max((int(d["generator"]["prompt_id"].split("#")[1]) for d in read_docs(out)), default=-1)
        debut = max(ancien.get("next_call", 0), dernier + 1)
    if a.seed in GRAINES_PRISES:
        dit(f"attention : la graine {a.seed} a déjà servi à la campagne précédente (101 à 124 et 900)")
    s.SOURCE, s.LICENSE = etiquettes(cfg)
    # Tous les contrôles sont passés : la trace peut être écrite (jamais avant, pour qu'un lancement
    # refusé ne change pas la trace d'un fichier existant).
    trace = {"consignes": chemin_publiable(a.consignes), "consignes_sha256": empreinte,
             "gabarit": cfg["gabarits"][STYLE]["id"], "juge": cfg["gabarits"]["juge"]["id"],
             "model": cfg["generateur"]["model"], "licence_generateur": cfg["licence_generateur"],
             "source": s.SOURCE, "license": s.LICENSE, "url": cfg["generateur"]["url"], "seed": a.seed,
             "temperature": cfg["generateur"]["temperature"], "num_predict": cfg["num_predict"][STYLE],
             "options_ollama": cfg.get("options_ollama") or {}, "options_du_juge": list(OPTIONS_DU_JUGE)}
    ecrire_trace(out, trace)
    # Journal des rejets : on ne garde que ceux des appels qui ne seront pas refaits.
    lignes = []
    if journal_rejets.exists() and debut:
        lignes = [x for x in journal_rejets.read_text(encoding="utf-8").splitlines()
                  if x.strip() and json.loads(x).get("call", debut) < debut]
    journal_rejets.write_text("".join(x + "\n" for x in lignes), encoding="utf-8")
    dit(f"générateur : {trace['model']} sur {trace['url']} ; gabarit {trace['gabarit']} ; "
        f"source « {s.SOURCE} » ; options en plus : {trace['options_ollama']}")

    etat = {"call": debut - 1}
    id_de_fiche = {x["texte"]: x["id"] for x in cfg["sujets"]}
    d_origine = s.raison_rejet

    def raison_rejet_notee(tours, parasites, style, cfg_, ctx=None):
        raison = d_origine(tours, parasites, style, cfg_, ctx)  # verdict inchangé
        if raison:
            try:
                c = ctx or {}
                with open(journal_rejets, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"call": etat["call"], "raison": raison, "fiche": id_de_fiche.get(c.get("sujet")),
                                        "voix": c.get("registre"), "persona": c.get("persona")},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
        return raison

    def avant_generation():
        etat["call"] += 1

    s.raison_rejet = raison_rejet_notee
    try:
        _, rapport = s.generer(STYLE, a.n, out, a.seed,
                               generateur=fabrique_generateur(s, cfg, appel, avant_generation), cfg=cfg,
                               max_appels=a.max_appels, log=dit, juge=None)
    finally:
        s.raison_rejet = d_origine
    print(json.dumps(rapport, ensure_ascii=False, indent=1))
    return rapport


def bilan(a):
    """Appels et conversations gardées par valeur d'axe, pour des fichiers par graine (lot pilote)."""
    from pipeline.io import read_docs
    s, _ = charger_modules(a.sans_torch)
    cfg = charger_consignes(s, a.consignes)
    empreinte = empreinte_fichier(a.consignes)
    fichiers = sorted(glob.glob(a.motif))
    if not fichiers:
        erreur(f"aucun fichier pour --motif {a.motif}")
    axes = ("bloc", "type", "forme", "deroule", "allers_retours", "longueur", "supplement",
            "valeur_du_supplement", "voix", "persona")
    appels = {k: Counter() for k in axes}
    gardes = {k: Counter() for k in axes}
    rejets = {k: Counter() for k in ("raison", "raison_par_bloc", "raison_par_forme", "raison_par_longueur")}
    tours_demandes = Counter()
    total = {"appels": 0, "gardees": 0, "candidats": 0}
    for f in fichiers:
        exige_zst(f, "chaque fichier de --motif")
        out, rapport_path, _ = s.chemins(f)
        if not rapport_path.exists():
            erreur(f"rapport absent : {rapport_path}")
        rapport = json.loads(rapport_path.read_text(encoding="utf-8"))
        if not isinstance(rapport.get("seed"), int):
            erreur(f"{f} : fichier fusionné (plusieurs graines) ; donner les fichiers par graine")
        meme_modele_et_gabarit(rapport, cfg, f)
        exige_memes_consignes(f, empreinte, a.consignes)
        total["appels"] += rapport["next_call"]
        total["candidats"] += rapport["candidats"]
        for call in range(rapport["next_call"]):
            _, meta = s.construire_appel(STYLE, rapport["seed"], call, cfg)
            v = dict(axes_de_la_fiche(meta["sujets"][0]["id"], cfg) or {}, voix=meta["registres"][0],
                     persona=meta["persona"])
            for k in axes:
                appels[k][v.get(k, "?")] += 1
        for d in read_docs(f):
            sujet = d.get("sujet") or {}
            v = dict(axes_de_la_fiche(sujet.get("id"), cfg) or {}, voix=sujet.get("registre"),
                     persona=d.get("persona"))
            for k in axes:
                gardes[k][v.get(k, "?")] += 1
            total["gardees"] += 1
            if v.get("allers_retours", "?") != "?":
                tours_demandes["respecte" if len(d["turns"]) == 2 * int(v["allers_retours"]) else "autre"] += 1
        journal = a_cote(f, ".rejets.jsonl")
        if journal.exists():
            for ligne in journal.read_text(encoding="utf-8").splitlines():
                if not ligne.strip():
                    continue
                r = json.loads(ligne)
                if r.get("call", 0) >= rapport["next_call"]:
                    continue
                v = axes_de_la_fiche(r.get("fiche"), cfg) or {}
                rejets["raison"][r["raison"]] += 1
                for k, axe in (("raison_par_bloc", "bloc"), ("raison_par_forme", "forme"),
                               ("raison_par_longueur", "longueur")):
                    rejets[k][f"{r['raison']} / {v.get(axe, '?')}"] += 1
    sortie = {"fichiers": [Path(f).name for f in fichiers], "consignes_sha256": empreinte, **total,
              "taux_de_garde_par_appel": round(total["gardees"] / max(total["appels"], 1), 4),
              "nombre_de_tours_demande": dict(tours_demandes),
              "par_axe": {k: {v: {"appels": n, "gardees": gardes[k][v], "taux": round(gardes[k][v] / n, 3)}
                              for v, n in sorted(appels[k].items())} for k in axes if k != "persona"},
              "appels_par_personnage": dict(sorted(appels["persona"].items())),
              "gardees_par_personnage": dict(sorted(gardes["persona"].items(), key=lambda x: str(x[0]))),
              "rejets_des_filtres": {k: dict(sorted(v.items(), key=lambda x: (-x[1], x[0]))) for k, v in rejets.items()},
              "note": ("« rejets_des_filtres » ne compte que les rejets de synthetic.raison_rejet (forme, echo, "
                       "trop_long, registre…), notés par « generer » ; les rejets de l'anti-doublon et du juge "
                       "sont dans le rapport de chaque fichier. Les appels sont reconstruits par construire_appel.")}
    print(json.dumps(sortie, ensure_ascii=False, indent=1))
    return sortie


def fusion(a):
    refuse_protege(a.dst, "--dst")
    dst = Path(a.dst)
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        erreur(f"--dst = {dst} existe déjà et n'est pas vide : la fusion y écraserait moliere.jsonl.zst ; "
               "donner un dossier neuf")
    s, nm = charger_modules(a.sans_torch)
    cfg = charger_consignes(s, a.consignes)
    empreinte = empreinte_fichier(a.consignes)
    fichiers = sorted(glob.glob(a.motif))
    if not fichiers:
        erreur(f"aucun fichier pour --motif {a.motif}")
    traces = []
    for f in fichiers:
        exige_zst(f, "chaque fichier de --motif")
        rapport = Path(f[:-len(".jsonl.zst")] + ".rapport.json")
        if not rapport.exists():
            erreur(f"rapport absent : {rapport} (génération pas finie ?)")
        meme_modele_et_gabarit(json.loads(rapport.read_text(encoding="utf-8")), cfg, f)
        traces.append(exige_memes_consignes(f, empreinte, a.consignes))
    if not Path(nm.VALIDATION).exists():
        erreur(f"fichier de validation introuvable : {nm.VALIDATION}")
    if a.ecart and not Path(a.ecart).exists():
        erreur(f"--ecart : fichier introuvable : {a.ecart}")
    # nm.fusion appelle load_config() sans argument (pour personas_exclus) : on lui fait lire CES consignes.
    d_origine, consignes = s.load_config, str(Path(a.consignes).resolve())
    s.load_config = lambda path=consignes, _f=d_origine: _f(path)
    try:
        nm.fusion(argparse.Namespace(motif=a.motif, dst=str(dst), validation=nm.VALIDATION, ecart=a.ecart,
                                     n=6, n_ecart=12))
    finally:
        s.load_config = d_origine
    source, licence = etiquettes(cfg)
    ecrire_trace(dst / "moliere.jsonl.zst", {
        "consignes": chemin_publiable(a.consignes), "consignes_sha256": empreinte, "gabarit": cfg["gabarits"][STYLE]["id"],
        "juge": cfg["gabarits"]["juge"]["id"], "model": cfg["generateur"]["model"],
        "licence_generateur": cfg["licence_generateur"], "source": source, "license": licence,
        "temperature": cfg["generateur"]["temperature"], "num_predict": cfg["num_predict"][STYLE],
        "options_ollama": cfg.get("options_ollama") or {}, "options_du_juge": list(OPTIONS_DU_JUGE),
        "seed": [t.get("seed") for t in traces], "ecart": a.ecart,
        "fichiers": [{"fichier": Path(f).name, "seed": t.get("seed"), "url": t.get("url")}
                     for f, t in zip(fichiers, traces)]})
    print(f"à faire ensuite : refiltrer --consignes {a.consignes} --fichier {dst / 'moliere.jsonl.zst'}")


def refiltrer(a):
    from pipeline.io import read_docs
    exige_zst(a.fichier, "--fichier")
    refuse_protege(a.fichier, "--fichier")
    s, _ = charger_modules(a.sans_torch)
    cfg = charger_consignes(s, a.consignes)
    out, rapport_path, _ = s.chemins(a.fichier)
    if not out.exists() or not rapport_path.exists():
        erreur(f"fichier ou rapport introuvable : {out}, {rapport_path}")
    rapport = json.loads(rapport_path.read_text(encoding="utf-8"))
    meme_modele_et_gabarit(rapport, cfg, out)
    exige_memes_consignes(out, empreinte_fichier(a.consignes), a.consignes)
    if not rapport.get("candidats"):
        erreur(f"{out} : aucune conversation candidate dans le rapport, rien à refiltrer")
    # Deux écarts faits ici, avant synthetic.refiltrer, comptés comme des rejets :
    # - « sans_sujet » : une conversation sans fiche (second bloc écrit malgré la consigne) ferait planter
    #   synthetic.refiltrer ; même raison que la fusion ;
    # - « titre_validation » : une conversation qui cite le titre d'une des trois pièces tenues à l'écart.
    docs = list(read_docs(out))
    ecartes, gardes = Counter(), []
    for d in docs:
        if not (d.get("sujet") or {}).get("registre"):
            ecartes["sans_sujet"] += 1
        elif any(TITRES_A_L_ECART.search(t["text"]) for t in d["turns"]):
            ecartes["titre_validation"] += 1
        else:
            gardes.append(d)
    if ecartes:
        for k, v in ecartes.items():
            rapport["rejets"][k] = rapport["rejets"].get(k, 0) + v
            rapport.setdefault("refiltrage", {})[k] = rapport.get("refiltrage", {}).get(k, 0) + v
        rapport["gardes"] = len(gardes)
        s.ecrit_atomique(out, rapport_path, gardes, rapport)
    gardes, rapport = s.refiltrer(STYLE, out, cfg=cfg)
    print(json.dumps({"fichier": str(out), "avant": len(docs), **dict(ecartes), "gardes": len(gardes),
                      "refiltrage": rapport.get("refiltrage", {}), "taux_rejet": rapport.get("taux_rejet")},
                     ensure_ascii=False, indent=1))


def apercu(a):
    s, _ = charger_modules(a.sans_torch)
    cfg = charger_consignes(s, a.consignes)
    source, licence = etiquettes(cfg)
    print(f"consignes : {a.consignes} (empreinte {empreinte_fichier(a.consignes)[:12]})")
    print(f"gabarit {cfg['gabarits'][STYLE]['id']} ; juge {cfg['gabarits']['juge']['id']} ; générateur "
          f"{cfg['generateur']['model']} ; température {cfg['generateur']['temperature']} ; num_predict "
          f"{cfg['num_predict'][STYLE]} ; options en plus : {json.dumps(cfg.get('options_ollama') or {}, ensure_ascii=False)}")
    print(f"source « {source} » ; licence « {licence} »")
    print(f"{len(cfg['sujets'])} fiches, {len(cfg['registres'])} voix, {len(cfg['exemples_moliere'])} vraies "
          f"répliques, {len(cfg['demonstrations_moliere'])} démonstration ; aucun appel n'est fait.")
    for i in range(a.n):
        prompt, meta = s.construire_appel(STYLE, a.seed, i, cfg)
        print(f"\n===== {meta['prompt_id']} | graine {a.seed} | personnage {meta['persona']} | voix "
              f"{meta['registres'][0]} | fiche {meta['sujets'][0]['id']} | graine Ollama {meta['ollama_seed']} | "
              f"{len(prompt)} caractères =====")
        print(prompt)


def main(argv=None):
    for flux in (sys.stdout, sys.stderr):  # consoles Windows : éviter une erreur d'encodage à l'affichage
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sans-torch", action="store_true", default=os.environ.get("NUIT_SANS_TORCH") == "1",
                   help="ne charge pas PyTorch (tools/nuit_moliere.py sans_torch ; aussi NUIT_SANS_TORCH=1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generer", help="fait écrire les conversations par le modèle ouvert (Ollama)")
    g.add_argument("--consignes", required=True, help="fichier écrit par tools/composer_consignes_v4.py")
    g.add_argument("--n", type=int, required=True, help="nombre de conversations gardées visé")
    g.add_argument("--seed", type=int, required=True, help="graine du fichier (201 et suivantes)")
    g.add_argument("--max-appels", type=int, default=None, help="plafond d'appels pour cette exécution")
    g.add_argument("--port", type=int, default=11434, help="port du serveur Ollama local")
    g.add_argument("--out", required=True, help="fichier .jsonl.zst (le même --out reprend la génération)")
    b = sub.add_parser("bilan", help="appels et conversations gardées par valeur d'axe (fichiers par graine)")
    b.add_argument("--consignes", required=True)
    b.add_argument("--motif", required=True, help="entre apostrophes : c'est Python qui développe l'étoile")
    f = sub.add_parser("fusion", help="réunit les fichiers par graine dans un dossier neuf")
    f.add_argument("--consignes", required=True)
    f.add_argument("--motif", required=True, help="entre apostrophes : c'est Python qui développe l'étoile")
    f.add_argument("--dst", required=True, help="dossier NEUF")
    f.add_argument("--ecart", default=None, help="fichier tenu à l'écart (jamais entraîné), facultatif")
    r = sub.add_parser("refiltrer", help="réapplique filtres et anti-doublon avec ces consignes")
    r.add_argument("--consignes", required=True)
    r.add_argument("--fichier", required=True, help="fichier .jsonl.zst, réécrit sur place")
    ap = sub.add_parser("apercu", help="affiche des prompts construits, sans rien appeler")
    ap.add_argument("--consignes", required=True)
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--n", type=int, default=3)
    a = p.parse_args(argv)
    {"generer": generer, "bilan": bilan, "fusion": fusion, "refiltrer": refiltrer, "apercu": apercu}[a.cmd](a)


if __name__ == "__main__":
    main()
