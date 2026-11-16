"""Outils des campagnes de données de synthèse du chatbot Molière (v4 et v5), hors chatbot/ (n'entre pas dans code_sha256).

Ce module sert aussi de bibliothèque : tools/campagne_synthese.py en importe sans_torch, fusion et VALIDATION.

Sous-commandes (python tools/nuit_moliere.py [--sans-torch] SOUS-COMMANDE ...) :

refiltrer : lance « python -m chatbot.synthetic --refiltrer » (--style, --out), d'ici pour profiter de
            --sans-torch ; réapplique filtres et anti-doublon à un fichier de conversations.
generer   : chatbot.synthetic.generer tel quel, sur le serveur Ollama d'un autre port (--port) ;
            plusieurs serveurs = plusieurs ports = plusieurs processus.
fusion    : réunit les fichiers par graine ; écarte les fuites vers la validation (suites de mots et
            personnages propres aux pièces de validation), les recouvrements avec le fichier tenu à
            l'écart, les conversations sans sujet et les réponses sans fin ; écrit fichier, rapport et
            journal fusionnés.
plafonner : garde au plus N conversations (tirage à graine fixe) pour tenir le plafond de synthèse.
controle  : comptes par thème / persona / registre / tours + N conversations tirées au hasard.
protocole : joue un jeu FIXE de messages basiques (--jeu), 5 graines, critères de forme (SEUILS).
            Les quatre jeux sont dans JEUX : « reglage » et « test » (ont servi à choisir le
            décodage et à cinq comparaisons), « neuf » (écrit le 14/11 avant toute réponse des
            modèles de la campagne de synthèse, a servi une fois) et « neuf2 » (écrit le 15/11 avant
            la campagne v5 et tout entraînement dessus).
bpb       : BPB d'un modèle sur plusieurs fichiers de scènes (float32, comme chatbot.evaluate).

Tout s'appuie sur les fonctions du dépôt (chatbot.synthetic, chatbot.evaluate, chatbot.finetune,
chatbot.generate, pipeline.io) sans en modifier aucune. Les règles de retrait sont publiées ici et
chaque retrait est compté dans le rapport fusionné : aucune conversation n'est écrite ni corrigée
à la main. À lancer depuis la racine du clone, avec .venv/Scripts/python.exe et PYTHONUTF8=1.
"""
import argparse
import glob
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))  # racine du clone : chatbot, pipeline, ftb

TOKENIZER = str(RACINE / "submissions/2026-11-11_153839/tokenizer.json")
VALIDATION = str(RACINE / "data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst")
ENTRAINEMENT = [str(RACINE / "data/stages/chatbot_20261112/theatre/theatre_train.jsonl.zst"),
                str(RACINE / "data/stages/chatbot_20261112/theatre/moliere_train.jsonl.zst")]
# Personnages des 3 pièces de validation (Le Misanthrope, Le Sicilien, L'Avare) dont le nom de rôle
# existe aussi ailleurs dans l'entraînement (homonymes de Corneille, Labiche...), donc que le calcul
# « rôles absents de l'entraînement » ne trouve pas. Écartés quand même : dans une conversation
# d'aujourd'hui écrite par un modèle de langue, ces noms renvoient presque sûrement à Molière.
NOMS_EN_PLUS = ["PHILINTE", "ÉLIANTE", "ARSINOÉ", "ADRASTE", "ISIDORE", "MAITRE SIMON", "DON PÈDRE"]
SEEDS = [11, 22, 33, 44, 55]  # critères automatiques sur les 5 ; lecture F/R/S sur 11, 22, 33
JEUX = {
    "reglage": (
        ["Bonjour !", "Salut, ça va ?", "Comment allez-vous ?", "Qui êtes-vous ?",
         "Quel temps fait-il aujourd'hui ?", "Qu'avez-vous mangé ce midi ?", "Je suis fatigué ce soir.",
         "J'ai beaucoup de travail en ce moment.", "Comment va votre famille ?", "Aimez-vous le théâtre ?",
         "Merci beaucoup pour votre aide.", "Au revoir, et bonne soirée !"],
        [["Bonjour, cher ami.", "Comment vous portez-vous ?", "Quel temps fait-il chez vous ?",
          "Merci, bonne journée !"],
         ["Salut, tu vas bien ?", "Moi, j'ai passé une mauvaise journée.", "Mon travail m'ennuie.",
          "Que feriez-vous à ma place ?"],
         ["Bonsoir, qui êtes-vous donc ?", "Que faites-vous dans la vie ?", "Vous aimez votre métier ?",
          "Au revoir, à demain !"]]),
    "test": (
        ["Bonjour, comment ça va ?", "Bonsoir !", "Comment vous appelez-vous ?", "D'où venez-vous ?",
         "Il pleut depuis ce matin.", "Avez-vous bien dormi ?", "Que faites-vous ce week-end ?",
         "Je suis content de vous voir.", "J'ai faim, que me conseillez-vous ?", "Pouvez-vous m'aider ?",
         "Je vous remercie de tout cœur.", "Bonne nuit, à bientôt !"],
        [["Bien le bonjour !", "Je m'appelle Camille.", "Et vous, quel est votre métier ?",
          "Cela vous plaît-il ?"],
         ["Bonsoir, monsieur.", "J'ai une question à vous poser.", "Faut-il se lever tôt le matin ?",
          "Je vous remercie, adieu."],
         ["Salut !", "Tu veux venir dîner ce soir ?", "On mangera du poulet et du fromage.",
          "À quelle heure viens-tu ?"]]),
    # Jeu NEUF, écrit le 14/11 à 18:45 AVANT toute réponse des modèles de la campagne de synthèse : les jeux
    # « reglage » et « test » ont servi à choisir le décodage et à cinq comparaisons. Deux fois plus grand
    # (24 messages isolés, 6 conversations suivies de 4 tours = 48 tours par graine) pour une mesure moins bruitée.
    "neuf": (
        ["Bonjour à vous !", "Coucou, quoi de neuf ?", "Comment se passe votre journée ?", "Quel est votre nom ?",
         "Quel âge avez-vous ?", "Où habitez-vous ?", "Il fait très chaud aujourd'hui.", "Il neige dehors, c'est beau.",
         "Avez-vous faim ?", "Que buvez-vous le matin ?", "Je me suis levé tard ce matin.", "J'ai mal à la tête.",
         "Je suis triste aujourd'hui.", "J'ai réussi mon examen !", "Mon chat est malade.", "Aimez-vous la musique ?",
         "Quel est votre plat préféré ?", "Avez-vous des enfants ?", "Que pensez-vous des médecins ?",
         "Dois-je épouser quelqu'un de riche ?", "Racontez-moi une histoire courte.", "Pardon de vous déranger.",
         "Vous êtes bien aimable.", "Bonne journée, à la prochaine !"],
        [["Bonjour, je suis nouveau ici.", "Je viens de Lyon.", "Connaissez-vous cette ville ?",
          "Merci pour l'accueil."],
         ["Bonsoir, je n'arrive pas à dormir.", "Je pense trop à mon travail.", "Avez-vous un conseil pour moi ?",
          "Je vais essayer, bonne nuit."],
         ["Salut, tu fais quoi ?", "Moi, je prépare le dîner.", "Tu aimes la soupe ?",
          "Alors viens manger avec nous !"],
         ["Bonjour, j'ai une nouvelle à vous dire.", "Je vais me marier.", "Qu'en pensez-vous ?",
          "Vous viendrez à la noce ?"],
         ["Excusez-moi, je cherche mon chemin.", "Je veux aller au marché.", "Est-ce loin d'ici ?",
          "Merci, vous êtes bien bon."],
         ["J'ai perdu de l'argent aujourd'hui.", "Cent euros, dans la rue.", "Que dois-je faire ?",
          "Vous avez raison, tant pis."]]),
    # Second jeu neuf, écrit le 15/11 à 16:00 AVANT toute génération de la campagne v5 et tout entraînement dessus :
    # le jeu « neuf » a servi une fois (choix de S03). Même taille (48 tours par graine).
    "neuf2": (
        ["Bien le bonjour, l'ami !", "Tout va bien chez vous ?", "Quelle heure est-il ?", "Quel est votre métier ?",
         "Aimez-vous lire ?", "Savez-vous danser ?", "Il y a du vent ce matin.", "Je n'ai pas le moral.",
         "J'ai soif.", "J'ai trop mangé ce midi.", "Mon voisin fait du bruit.", "J'ai raté mon train.",
         "Je pars en vacances demain.", "C'est mon anniversaire aujourd'hui.", "Que pensez-vous de l'amour ?",
         "Faut-il dire toujours la vérité ?", "Donnez-moi un conseil pour bien dormir.", "Quel est votre plus grand défaut ?",
         "Êtes-vous heureux ?", "Je ne vous comprends pas.", "Vous me faites rire.", "Excusez mon retard.",
         "Félicitations !", "À demain, portez-vous bien !"],
        [["Bonjour, puis-je vous poser une question ?", "Comment devient-on comédien ?", "Est-ce un métier difficile ?",
          "Merci de votre réponse."],
         ["Coucou, tu es là ?", "Je m'ennuie un peu.", "Tu as une idée pour m'occuper ?", "Bonne idée, j'y vais !"],
         ["Bonsoir, j'ai un souci.", "Mon ami ne me parle plus.", "Dois-je lui écrire ?", "D'accord, je vais le faire."],
         ["Bonjour, il fait beau aujourd'hui.", "Je vais me promener au parc.", "Voulez-vous venir avec moi ?",
          "Alors à tout à l'heure !"],
         ["J'ai acheté un chapeau.", "Il est rouge, avec une plume.", "Me va-t-il bien, à votre avis ?",
          "Vous êtes trop aimable."],
         ["Je dois parler devant cent personnes.", "J'ai très peur.", "Comment faire pour ne pas trembler ?",
          "Merci, je me sens mieux."]]),
}
# Seuils fixés AVANT les mesures (conventions du protocole, déclarées telles quelles). empty_rate
# n'y figure pas : ChatBot.reply rend « … » au lieu d'une réponse vide, c'est repli_rate qui le voit.
SEUILS = {"repli_rate": ("max", 0.01), "natural_stop_rate": ("min", 0.95),
          "clean_end_rate": ("min", 0.95), "echo_rate": ("max", 0.03), "role_name_rate": ("max", 0.02),
          "mean_words": ("entre", (5, 20)), "passe_partout_rate": ("max", 0.10),
          "self_copy_rate": ("max", 0.05), "history_gain": ("min", 0.001)}
CUMULS = ("appels", "next_call", "candidats", "blocs_sans_tour", "tokens_prompt", "tokens_generes",
          "duree_generation_s", "appels_juge", "duree_juge_s")
FIN = re.compile(r'[.!?…»"]$')


def plat(texte):
    """Minuscules, sans accents, tout ce qui n'est pas une lettre devient une espace.

    « Maître Jacques », « MAITRE JACQUES » et « maitre-jacques » donnent « maitre jacques ».
    """
    sans = "".join(c for c in unicodedata.normalize("NFKD", texte.casefold())
                   if unicodedata.category(c) != "Mn")
    return " ".join("".join(c if c.isalpha() else " " for c in sans).split())


def noms_propres_a_la_validation(validation, entrainement, min_lettres=4):
    """Rôles qui parlent dans les pièces de validation et dans AUCUNE scène d'entraînement.

    Règle du dépôt (chatbot/theatre.py) : ni les textes des 3 pièces tenues à l'écart, ni des scènes
    qui reprennent leurs personnages. On ne garde que les noms d'au moins `min_lettres` lettres : un
    rôle comme « UN » ou « LE » écarterait toutes les conversations.
    """
    from pipeline.io import read_docs

    def roles(path):
        return {t["speaker"].strip() for d in read_docs(path) for t in d["turns"] if t.get("speaker")}

    connus = set().union(*(roles(p) for p in entrainement))
    return sorted(n for n in roles(validation) - connus if sum(c.isalpha() for c in n) >= min_lettres)


def nombre_de_mots(docs):
    from chatbot.synthetic import mots
    return sum(len(mots(t["text"])) for d in docs for t in d["turns"])


def fusion(a):
    from chatbot.format import clean_text
    from chatbot.synthetic import load_config, ngrammes
    from pipeline.io import read_docs, write_docs

    def suites(path, n):
        s = set()
        for d in read_docs(path):
            for t in d["turns"]:
                s |= ngrammes(t["text"], n)
        return s

    def suites_doc(textes, n):
        return set().union(*(ngrammes(t, n) for t in textes)) if textes else set()

    fichiers = sorted(glob.glob(a.motif))
    if not fichiers:
        sys.exit(f"ERREUR : aucun fichier pour {a.motif}")
    bases = [f[:-len(".jsonl.zst")] for f in fichiers]
    manquants = [b + ext for b in bases for ext in (".rapport.json", ".journal.log")
                 if not Path(b + ext).exists()]
    if manquants:  # fichier sans rapport : génération pas finie ou plantée avant sa 1re sauvegarde
        sys.exit("ERREUR : rapport ou journal absent : " + ", ".join(manquants))
    val, val8 = suites(a.validation, a.n), suites(a.validation, 8)
    ecart = suites(a.ecart, a.n_ecart) if a.ecart else set()
    # Personnages des pièces de validation : calculés, + ceux que les consignes publiées excluent
    # (HARPAGON parle dans 5 scènes d'un pastiche de Guignol de theatre_train, le calcul le manque),
    # + les homonymes listés dans NOMS_EN_PLUS.
    noms = sorted(set(noms_propres_a_la_validation(a.validation, ENTRAINEMENT))
                  | set(load_config().get("personas_exclus", [])) | set(NOMS_EN_PLUS))
    motifs = [(n, re.compile(r"(?<![a-z])" + re.escape(plat(n)) + r"(?![a-z])")) for n in noms]
    messages_du_protocole = {clean_text(m).casefold() for isoles, suivies in JEUX.values()
                             for m in isoles + [x for c in suivies for x in c]}
    gardes, ecartes, par_nom = [], Counter(), Counter()
    fuite8 = identiques = 0
    for f in fichiers:
        for d in read_docs(f):
            textes = [t["text"] for t in d["turns"]]
            bas = plat("\n".join(textes))
            touche = [n for n, m in motifs if m.search(bas)]
            if d.get("sujet") is None:  # 4e bloc d'un appel : sans sujet, --refiltrer planterait
                ecartes["sans_sujet"] += 1
            elif any(not unicodedata.is_normalized("NFC", t) for t in textes):
                ecartes["non_nfc"] += 1  # accents décomposés : échapperaient aux suites de mots
            elif suites_doc(textes, a.n) & val:
                ecartes[f"fuite_validation_{a.n}_mots"] += 1
                fuite8 += bool(suites_doc(textes, 8) & val8)
            elif touche:
                ecartes["personnage_validation"] += 1
                par_nom.update(touche)
            elif ecart and suites_doc(textes, a.n_ecart) & ecart:
                ecartes[f"commun_avec_ecart_{a.n_ecart}_mots"] += 1
            elif not all(FIN.search(t.strip()) for t in textes[1::2]):
                ecartes["reponse_sans_fin"] += 1
            else:
                identiques += sum(t.casefold() in messages_du_protocole for t in textes[0::2])
                gardes.append(d)
    dst = Path(a.dst)
    n = write_docs(dst / "moliere.jsonl.zst", gardes)
    rs = [json.loads(Path(b + ".rapport.json").read_text(encoding="utf-8")) for b in bases]
    rejets = sum((Counter(r["rejets"]) for r in rs), Counter()) + ecartes
    tot = dict(rs[0], seed=[r["seed"] for r in rs], gardes=n, rejets=dict(rejets), rejets_exemples={},
               juge_rejets_exemples=[], fichiers=[Path(f).name for f in fichiers],
               ecartes_a_la_fusion=dict(ecartes),
               fusion={"suites_de_mots_validation": a.n, "dont_fuite_a_8_mots": fuite8,
                       "suites_de_mots_ecart": a.n_ecart if a.ecart else None,
                       "noms_ecartes": noms, "rejets_par_nom": dict(par_nom),
                       "messages_identiques_au_protocole": identiques,
                       "gardes_par_fichier": {Path(f).name: r["gardes"] for f, r in zip(fichiers, rs)}})
    for k in CUMULS:
        tot[k] = sum(r[k] for r in rs)
    tot["taux_rejet"] = round(1 - n / tot["candidats"], 4) if tot["candidats"] else None
    tot["mots_gardes"] = nombre_de_mots(gardes)
    (dst / "moliere.rapport.json").write_text(json.dumps(tot, ensure_ascii=False, indent=1), encoding="utf-8")
    (dst / "moliere.journal.log").write_text(
        "".join(Path(b + ".journal.log").read_text(encoding="utf-8") for b in bases), encoding="utf-8")
    print(f"{n} conversations fusionnées depuis {len(fichiers)} fichiers ; écartées : {dict(ecartes)}")
    print("gardées par fichier avant fusion : " + ", ".join(f"{Path(f).name} {r['gardes']}"
                                                              for f, r in zip(fichiers, rs)))
    print(f"noms écartés ({len(noms)}) : {', '.join(noms)}")
    print(f"rejets par nom : {dict(par_nom)} ; messages de A identiques au protocole (gardés, comptés) : {identiques}")


def generer(a):
    """chatbot.synthetic.generer tel quel, sur un serveur Ollama d'un autre PORT.

    llama-server traite ses flux parallèles dans une seule boucle (1 cœur de processeur) : pour
    remplir la carte il faut plusieurs serveurs, donc plusieurs ports. Seule l'adresse change, en
    mémoire ; chatbot/prompts_synthetiques.json (consignes, filtres, juge) n'est pas modifié.
    """
    from chatbot import synthetic as s
    cfg = s.load_config()
    cfg["generateur"]["url"] = f"http://127.0.0.1:{a.port}/api/generate"
    _, rapport = s.generer(a.style, a.n, a.out, a.seed, cfg=cfg, max_appels=a.max_appels,
                           log=lambda x: print(x, file=sys.stderr, flush=True), juge=None)
    print(json.dumps(rapport, ensure_ascii=False, indent=1))


def plafonner(a):
    """Garde au plus --max conversations, tirées avec une graine fixe, dans l'ordre d'origine."""
    from chatbot.accel import replace_file
    from pipeline.io import read_docs, write_docs
    src = Path(a.fichier)
    rapport_path = Path(str(src)[:-len(".jsonl.zst")] + ".rapport.json")
    rapport = json.loads(rapport_path.read_text(encoding="utf-8"))  # lu AVANT toute écriture
    docs = list(read_docs(src))
    avant = len(docs)
    if avant > a.max:
        gardes = set(random.Random(a.graine).sample(range(avant), a.max))
        docs = [d for i, d in enumerate(docs) if i in gardes]
    if avant > a.max or "plafonnement" not in rapport:  # une 2e exécution ne perd pas le vrai « avant »
        rapport["plafonnement"] = {"avant": avant, "max": a.max, "apres": len(docs), "graine": a.graine,
                                   "pourquoi": a.pourquoi}
    rapport["gardes"] = len(docs)
    rapport["mots_gardes"] = nombre_de_mots(docs)
    tmp = src.with_name(src.name + ".tmp")
    write_docs(tmp, docs)
    tmp_r = rapport_path.with_name(rapport_path.name + ".tmp")
    tmp_r.write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    replace_file(tmp, src)
    replace_file(tmp_r, rapport_path)
    print(f"{avant} -> {len(docs)} conversations (plafond {a.max}, graine {a.graine})")


def controle(a):
    from chatbot.synthetic import rendu
    from pipeline.io import read_docs
    docs = list(read_docs(a.fichier))
    champs = {"theme": lambda d: (d.get("sujet") or {}).get("theme"),
              "persona": lambda d: d.get("persona"),
              "registre": lambda d: (d.get("sujet") or {}).get("registre"),
              "tours": lambda d: len(d["turns"])}
    lignes = [f"{len(docs)} conversations dans {a.fichier}"]
    for nom, f in champs.items():
        c = Counter(f(d) for d in docs)
        lignes.append(nom + " : " + ", ".join(f"{k} {v} ({100 * v / len(docs):.1f} %)"
                                              for k, v in c.most_common()))
    resume = list(lignes)
    for i, d in enumerate(random.Random(a.graine).sample(docs, min(a.n, len(docs))), 1):
        s = d.get("sujet") or {}
        lignes.append(f"\n### {i}. {d['id']} | {s.get('theme')} | {s.get('registre')}\n{rendu(d).rstrip()}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print("\n".join(resume))


def charger(model, device):
    from chatbot.generate import DEFAULT_CHAT_CONFIG, ChatBot
    p = Path(model)
    if p.is_dir():
        return ChatBot.load(p, device=device)
    # comme chatbot.evaluate avec un .pt : réglages par défaut, format appris par le checkpoint
    return ChatBot.from_checkpoint(p, TOKENIZER, dict(DEFAULT_CHAT_CONFIG, format=None), device)


def lire_reglages(paires):
    """--regler CLE=VALEUR : seulement les clés de décodage relues à chaque réponse, valeur en JSON."""
    from chatbot.generate import DEFAULT_CHAT_CONFIG
    # le format est fixé au chargement du modèle ; ancre_mu = décodage ancré sur le dernier message (chatbot/ancrage.py)
    reglables = (set(DEFAULT_CHAT_CONFIG) - {"format"}) | {"ancre_mu"}
    reglages = {}
    for x in paires:
        k, sep, v = x.partition("=")
        if not sep or k not in reglables:
            sys.exit(f"ERREUR : réglage inconnu « {x} » ; clés réglables : {sorted(reglables)}")
        try:
            reglages[k] = json.loads(v)
        except json.JSONDecodeError:
            sys.exit(f"ERREUR : « {x} » : la valeur doit être du JSON (0.7, 8, true, null, '\"SCAPIN\"')")
    return reglages


def protocole(a):
    from chatbot.accel import setup_console
    from chatbot.evaluate import (panel_persona, play, summarize, summarize_discussion, words,
                                  write_transcript)
    setup_console()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)  # avant le calcul : un mauvais --out échoue tout de suite
    reglages = lire_reglages(a.regler)
    bot = charger(a.model, a.device)
    # « ancre_mu » (nom d'origine de cet outil) et « anchor_mu » (clé du chat_config.json) règlent la même
    # chose. La clé seule ne suffit pas : ChatBot ne la lit qu'à sa création, il faut BRANCHER l'ancrage ici,
    # sinon la mesure serait faite sans ancrage tout en l'annonçant dans mesures.json.
    ancre_mu = reglages.pop("ancre_mu", None)
    if "anchor_mu" in reglages:
        if ancre_mu is not None and float(ancre_mu) != float(reglages["anchor_mu"]):
            sys.exit("ERREUR : --regler ancre_mu et anchor_mu donnent deux valeurs différentes")
        ancre_mu = reglages.pop("anchor_mu")
    bot.config.update(reglages)
    if ancre_mu is not None:  # 0 coupe l'ancrage d'un modèle dont le chat_config le demande
        from chatbot.ancrage import installer  # dans le dépôt : l'outil marche après un simple clone
        installer(bot, ancre_mu)
        bot.config["anchor_mu"] = float(ancre_mu)
        reglages["ancre_mu"] = ancre_mu
    qui = panel_persona(bot, None)
    isoles, suivies = JEUX[a.jeu]
    jobs = [("basique", qui, "isole", [m]) for m in isoles] + [("basique", qui, "suivi", c) for c in suivies]
    # play() donne la graine « graine x 100 + tour » sans le numéro du message : deux messages
    # différents recevraient les mêmes nombres aléatoires. On joue donc message par message avec une
    # graine propre (graine x 1000 + numéro du message), puis on remet job et graine de base.
    rows, total = [], sum(len(j[3]) for j in jobs) * len(SEEDS)
    for i, job in enumerate(jobs):
        part = play(bot, [job], False, [g * 1000 + i for g in SEEDS], False, "")
        for r in part:
            r.update(job=i, base_seed=r["base_seed"] // 1000)
        rows += part
        print(f"\r{a.jeu}/{a.nom} : {len(rows)}/{total} réponses", end="", file=sys.stderr)
    print(file=sys.stderr)
    n = len(rows)
    s = summarize(rows)
    d = summarize_discussion([r for r in rows if r["theme"] == "suivi"])

    def cle(texte):
        return " ".join(words(texte))

    messages = defaultdict(set)  # texte de réponse -> messages DIFFÉRENTS qui l'ont reçu
    for r in rows:
        messages[cle(r["reply"])].add(cle(r["message"]))
    debut, nb = Counter((words(r["reply"]) or ["(vide)"])[0] for r in rows).most_common(1)[0]
    m = {k: s[k] for k in ("n", "empty_rate", "natural_stop_rate", "clean_end_rate", "echo_rate",
                           "role_name_rate", "mean_words", "overlap_non_echo_rate", "overlap_eligible",
                           "seconds_per_reply", "finish")}
    m.update(repli_rate=round(sum(r["score"] is None for r in rows) / n, 4),
             # (une réponse sans lettre, comme le repli « … », a une clé vide : comptée par repli_rate)
             passe_partout_rate=round(sum(bool(cle(r["reply"])) and len(messages[cle(r["reply"])]) >= 2
                                          for r in rows) / n, 4),
             debut_le_plus_frequent=[debut, round(nb / n, 4)],
             self_copy_rate=d["self_copy_rate"], history_gain=d["history_gain_turns_2_6"],
             conversations_tronquees=d["truncated_conversation_rate"],
             par_graine={k: [summarize([r for r in rows if r["base_seed"] == g])[k] for g in SEEDS]
                         for k in ("natural_stop_rate", "echo_rate", "mean_words", "overlap_non_echo_rate")})
    verdict = {}
    for k, (sens, seuil) in SEUILS.items():
        x = m.get(k)
        verdict[k] = None if x is None else (x <= seuil if sens == "max" else
                                             x >= seuil if sens == "min" else seuil[0] <= x <= seuil[1])
    m.update(verdict=verdict, criteres_reussis=sum(v is True for v in verdict.values()),
             criteres=len(SEUILS), forme_ok=all(v is True for v in verdict.values()))
    (out / "mesures.json").write_text(json.dumps(
        dict(m, jeu=a.jeu, nom=a.nom, model=str(a.model), reglages=reglages, generation=bot.config,
             seeds=SEEDS), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                                            for r in rows), encoding="utf-8")
    write_transcript(out / "transcript.md", rows, f"Conversation basique : {a.jeu}, {a.nom}, {a.model}")
    print(json.dumps({k: m[k] for k in ("criteres_reussis", "criteres", "forme_ok", "verdict", "mean_words",
                                        "passe_partout_rate", "debut_le_plus_frequent",
                                        "overlap_non_echo_rate", "seconds_per_reply")},
                     ensure_ascii=False, indent=1))


def bpb(a):
    from chatbot.finetune import evaluate_windows, load_val_windows, token_byte_lengths
    out = Path(a.out)
    out = out / "bpb.json" if out.is_dir() else out
    out.parent.mkdir(parents=True, exist_ok=True)
    bot = charger(a.model, a.device)
    res = {}
    for f in a.fichiers:
        w = load_val_windows(f, bot.tok, 512, None, fmt_name=bot.config["format"])
        r = evaluate_windows(bot.model, w, token_byte_lengths(bot.tok), a.device, 16)
        res[f] = {"val_bpb": round(r["val_bpb"], 5), "val_loss": round(r["val_loss"], 5), "fenetres": len(w)}
        print(f"{f} : BPB {r['val_bpb']:.4f} ({len(w)} fenêtres)")
    out.write_text(json.dumps({"model": str(a.model), "fichiers": res}, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")


CRUDE_SHA256_ATTENDU = "cd083f939257897b"  # « crude_regex_sha256 » du rapport de la base (lancement v1)
SANS_TORCH = {"chatbot.generate": ("chatbot/generate.py", {"WORD", "echo_ratio"}),
              "chatbot.pretrain": ("chatbot/pretrain.py", {"CRUDE"}),
              "chatbot.accel": ("chatbot/accel.py", {"replace_file"})}


def sans_torch():
    """Rend chatbot.generate, chatbot.pretrain et chatbot.accel importables SANS charger PyTorch.

    Pourquoi : sur certains postes Windows, le chargement de torch_cpu.dll peut être refusé par le
    système (WinError 4551), et charger PyTorch ne sert à rien pour écrire du texte.
    Le générateur de synthèse n'a besoin de PyTorch que par ricochet : chatbot/synthetic.py prend
    WORD et echo_ratio dans chatbot/generate.py et CRUDE dans chatbot/pretrain.py, deux fichiers
    qui font « import torch » en tête. Ici on lit ces fichiers SOURCE tels quels et on n'en
    exécute que : leurs imports de la bibliothèque standard, et les définitions demandées. Les
    objets obtenus viennent donc des mêmes lignes que d'habitude ; aucun fichier du dépôt n'est
    modifié et la DLL refusée n'est jamais chargée. Ne sert qu'aux sous-commandes de TEXTE
    (generer, fusion, refiltrer, plafonner, controle) ; protocole et bpb exigent le vrai PyTorch.
    """
    import ast
    import hashlib
    import types
    if "torch" in sys.modules:
        sys.exit("--sans-torch : PyTorch est déjà chargé, l'option n'a pas de sens ici")
    for nom, (fichier, voulus) in SANS_TORCH.items():
        chemin = RACINE / fichier
        corps, trouves = [], set()
        for n in ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin)).body:
            if isinstance(n, ast.Import) and all(x.name.split(".")[0] in sys.stdlib_module_names for x in n.names):
                corps.append(n)
            elif (isinstance(n, ast.ImportFrom) and n.level == 0
                  and (n.module or "").split(".")[0] in sys.stdlib_module_names):
                corps.append(n)
            elif isinstance(n, ast.FunctionDef) and n.name in voulus:
                corps.append(n)
                trouves.add(n.name)
            elif isinstance(n, ast.Assign):
                cibles = {t.id for t in n.targets if isinstance(t, ast.Name)} & voulus
                if cibles:
                    corps.append(n)
                    trouves |= cibles
        if trouves != voulus:
            sys.exit(f"--sans-torch : {sorted(voulus - trouves)} introuvable(s) dans {fichier}")
        mod = types.ModuleType(nom)
        mod.__file__ = str(chemin)
        exec(compile(ast.Module(body=corps, type_ignores=[]), str(chemin), "exec"), mod.__dict__)
        sys.modules[nom] = mod
    sha = hashlib.sha256(sys.modules["chatbot.pretrain"].CRUDE.pattern.encode()).hexdigest()[:16]
    if sha != CRUDE_SHA256_ATTENDU:
        sys.exit(f"--sans-torch : CRUDE a changé ({sha} au lieu de {CRUDE_SHA256_ATTENDU})")
    print("sans torch : WORD, echo_ratio, CRUDE et replace_file extraits des sources du dépôt "
          f"(CRUDE sha256 {sha}) ; PyTorch non chargé", file=sys.stderr, flush=True)


def refiltrer(a):
    """« python -m chatbot.synthetic --refiltrer » tel quel, lancé d'ici pour profiter de --sans-torch."""
    import runpy
    sys.argv = ["chatbot.synthetic", "--refiltrer", "--style", a.style, "--out", a.out]
    runpy.run_module("chatbot.synthetic", run_name="__main__")


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sans-torch", action="store_true", default=os.environ.get("NUIT_SANS_TORCH") == "1",
                   help="texte seulement : ne charge pas PyTorch (aussi : variable NUIT_SANS_TORCH=1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    rf = sub.add_parser("refiltrer")
    rf.add_argument("--style", choices=["moderne", "moliere"], default="moliere")
    rf.add_argument("--out", required=True)
    f = sub.add_parser("fusion")
    f.add_argument("--motif", required=True, help="entre apostrophes : c'est Python qui développe l'étoile")
    f.add_argument("--dst", required=True)
    f.add_argument("--validation", default=VALIDATION)
    f.add_argument("--ecart", default=None, help="fichier tenu à l'écart (jamais entraîné)")
    f.add_argument("--n", type=int, default=6, help="suites de mots comparées aux pièces de validation")
    f.add_argument("--n-ecart", type=int, default=12,
                   help="suites de mots comparées au fichier à l'écart (les messages de A reprennent "
                        "les sujets du prompt : à 6 mots, plus de la moitié des conversations partiraient)")
    g = sub.add_parser("generer")
    g.add_argument("--style", choices=["moderne", "moliere"], default="moliere")
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--max-appels", type=int, required=True)
    g.add_argument("--port", type=int, default=11434)
    pl = sub.add_parser("plafonner")
    pl.add_argument("--fichier", required=True)
    pl.add_argument("--max", type=int, required=True)
    pl.add_argument("--graine", type=int, default=2026)
    pl.add_argument("--pourquoi", default="plafond de la part synthétique")
    c = sub.add_parser("controle")
    c.add_argument("--fichier", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--graine", type=int, default=7)
    pr = sub.add_parser("protocole")
    pr.add_argument("--model", required=True, help="dossier exporté ou checkpoint .pt")
    pr.add_argument("--jeu", choices=sorted(JEUX), required=True)
    pr.add_argument("--out", required=True, help="DOSSIER (mesures.json, rows.jsonl, transcript.md)")
    pr.add_argument("--nom", default="ref")
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--regler", nargs="*", default=[], metavar="CLE=VALEUR")
    b = sub.add_parser("bpb")
    b.add_argument("--model", required=True)
    b.add_argument("--fichiers", nargs="+", required=True)
    b.add_argument("--out", required=True, help="FICHIER .json (ou dossier existant : bpb.json dedans)")
    b.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.sans_torch:
        if a.cmd in ("protocole", "bpb"):
            sys.exit(f"--sans-torch : « {a.cmd} » fait tourner un modèle et exige le vrai PyTorch")
        sans_torch()
    {"fusion": fusion, "generer": generer, "plafonner": plafonner, "controle": controle, "protocole": protocole,
     "bpb": bpb, "refiltrer": refiltrer}[a.cmd](a)


if __name__ == "__main__":
    main()
