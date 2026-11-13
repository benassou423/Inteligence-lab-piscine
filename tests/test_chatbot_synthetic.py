"""Tests du générateur synthétique (chatbot/synthetic.py), SANS Ollama : un faux générateur est injecté."""

import json
import random

import pytest

from chatbot import synthetic as syn
from chatbot.format import PERSONAS
from pipeline.io import read_docs, write_docs

CFG = syn.load_config()


VOCAB = ("maison jardin pain soleil chemin rivière lettre voisin marché fenêtre livre chanson "
         "village route cheval table chapeau porte lampe fromage pomme hiver printemps forêt "
         "montagne nuage horloge bougie miroir panier manteau valise bateau ruisseau colline "
         "prairie moulin grenier cuisine théâtre musique peinture dimanche matin soirée").split()


def phrase(rng):
    """Phrase de 8 mots tirés au hasard : jamais deux fois le même 5-gramme (anti-doublon)."""
    return " ".join(rng.choice(VOCAB) for _ in range(8)).capitalize() + "."


def bloc(n):
    """Une conversation de 4 tours valide pour les deux styles, unique grâce au numéro n."""
    rng = random.Random(n)
    return "\n".join(f"{'AB'[i % 2]}: {phrase(rng)}" for i in range(4))


class FauxGenerateur:
    """Rend k conversations valides par appel et garde les prompts reçus.

    Les prompts du juge (relecture OUI/NON) sont rangés à part et reçoivent `verdict`."""

    def __init__(self, style, k=3, verdict="1: OUI\n2: OUI\n3: OUI"):
        self.style, self.k, self.n, self.prompts = style, k, 0, []
        self.verdict, self.jugements = verdict, []

    def __call__(self, prompt, options):
        if syn.est_prompt_juge(prompt, CFG):
            self.jugements.append((prompt, options))
            return {"response": self.verdict, "eval_count": 6, "eval_duration": 3e8}
        self.prompts.append((prompt, options))
        blocs = []
        for _ in range(self.k):
            self.n += 1
            blocs.append(bloc(self.n))
        return {"response": "\n###\n".join(blocs), "eval_count": 100, "eval_duration": 5e9,
                "prompt_eval_count": 400}


def tours(*textes):
    return [("AB"[i % 2], t) for i, t in enumerate(textes)]


OK4 = ["Tu viens ce soir au cinéma ?", "Je ne peux pas, je travaille tard.",
       "Dommage, une autre fois alors.", "Promis, samedi je suis libre."]
OKM = ["Salut, tu vas bien ?", "Fort bien, je vous remercie, et vous-même ?",
       "Ça va, je suis un peu fatigué.", "Prenez du repos, la fatigue ne vaut rien à personne."]


# ---------------------------------------------------------------- configuration publiée
def test_prompts_publies_complets():
    assert len(CFG["sujets"]) >= 120
    assert len({s["id"] for s in CFG["sujets"]}) == len(CFG["sujets"])
    assert len(CFG["exemples_moliere"]) >= 60
    for e in CFG["exemples_moliere"]:
        assert 8 <= len(syn.mots(e["texte"])) <= 25 and e["texte"][-1] in ".!?"
    regs = {r["id"] for r in CFG["registres"]}
    assert {"tu_court", "tu_long", "vous_court", "vous_long"} <= regs


# ---------------------------------------------------------------- 1. prompts reproductibles
@pytest.mark.parametrize("style", ["moderne", "moliere"])
def test_prompt_deterministe_par_graine(style):
    p1, m1 = syn.construire_appel(style, 1, 7, CFG)
    p2, m2 = syn.construire_appel(style, 1, 7, CFG)
    p3, _ = syn.construire_appel(style, 2, 7, CFG)
    p4, _ = syn.construire_appel(style, 1, 8, CFG)
    assert (p1, m1) == (p2, m2)
    assert p1 != p3 and p1 != p4
    assert "{" not in p1 and "}" not in p1  # tous les champs du gabarit sont remplis
    for s in m1["sujets"]:
        assert s["texte"] in p1


def test_moliere_une_fois_sur_deux():
    noms = [syn.construire_appel("moliere", 3, c, CFG)[1]["persona"] for c in range(400)]
    part = noms.count("MOLIÈRE") / len(noms)
    assert 0.4 < part < 0.6
    assert set(noms) <= {p["name"] for p in PERSONAS}
    assert len(set(noms)) > 5


# ---------------------------------------------------------------- 2. analyse de la sortie
def test_parse_sortie_preambule_structure_et_variantes():
    raw = ("<think>\n</think>\nBien sûr ! Voici trois conversations :\n###\n"
           "Conversation 1\nA: Salut !\n**B:** « Salut, ça va ? »\n\n###\n"
           "A : Tu viens ?\n- B: — Oui j'arrive.\n###\n")
    blocs, vides = syn.parse_sortie(raw)
    assert vides == 1  # le préambule n'est pas une conversation
    assert [b[0] for b in blocs] == [[("A", "Salut!"), ("B", "Salut, ça va?")],
                                     [("A", "Tu viens?"), ("B", "Oui j'arrive.")]]
    assert all(not b[1] for b in blocs)


def test_parse_sortie_ligne_parasite():
    blocs, _ = syn.parse_sortie("A: Bonjour.\nVoici un échange entre deux amis.\nB: Bonjour à toi.")
    assert blocs[0][1] == ["Voici un échange entre deux amis."]
    assert syn.raison_rejet(blocs[0][0], blocs[0][1], "moderne", CFG) == "meta"
    blocs, _ = syn.parse_sortie("A: Bonjour.\nil fait beau\nB: Oui.")
    assert syn.raison_rejet(blocs[0][0], blocs[0][1], "moderne", CFG) == "mal_formee"


# ---------------------------------------------------------------- 3. chaque filtre
def test_conversations_valides_gardees():
    assert syn.raison_rejet(tours(*OK4), [], "moderne", CFG) is None
    assert syn.raison_rejet(tours(*OKM), [], "moliere", CFG) is None
    # « Bien sûr » est du français naturel : pas du méta-texte dans une réplique.
    assert syn.raison_rejet(tours("Tu m'aides ?", "Bien sûr, j'arrive tout de suite.",
                                  "Merci beaucoup.", "Avec plaisir, à tout de suite."),
                            [], "moderne", CFG) is None


@pytest.mark.parametrize("conv,style,raison", [
    (tours(*OK4[:3]), "moderne", "mal_formee"),                       # 3 tours < 4
    (tours(*(OK4 * 3)), "moderne", "mal_formee"),                     # 12 tours > 8
    ([("A", OK4[0]), ("A", OK4[1]), ("B", OK4[2]), ("A", OK4[3])], "moderne", "mal_formee"),
    ([("B", OK4[0]), ("A", OK4[1]), ("B", OK4[2]), ("A", OK4[3])], "moderne", "mal_formee"),
    (tours(*OKM, "Et demain ?"), "moliere", "mal_formee"),            # le personnage ne finit pas
    (tours(OK4[0], "我很好，谢谢。", *OK4[2:]), "moderne", "non_latin"),
    (tours(OK4[0], "Ça va 😀 merci beaucoup.", *OK4[2:]), "moderne", "non_latin"),
    (tours(OK4[0], "Sure, I'm free, see you there with the others.", *OK4[2:]), "moderne", "anglais"),
    (tours(OK4[0], "*sourit* Oui, je viens.", *OK4[2:]), "moderne", "meta"),
    (tours(OK4[0], "Oui (il sourit) je viens.", *OK4[2:]), "moderne", "meta"),
    (tours(OK4[0], "Ce dialogue est amusant, je viens.", *OK4[2:]), "moderne", "meta"),
    (tours("Salut B, tu as reçu mon message ?", *OK4[1:]), "moderne", "meta"),
    (tours(OK4[0], "J'ai vraiment merdé, je ne peux pas.", *OK4[2:]), "moderne", "vocabulaire_cru"),
    (tours(OK4[0], "…", *OK4[2:]), "moderne", "tour_vide"),
    (tours(OK4[0], " ".join(["mot"] * 61), *OK4[2:]), "moderne", "trop_long"),
    (tours(OK4[0], "Tu viens ce soir au cinéma ? Oui.", *OK4[2:]), "moderne", "echo"),
    (tours(OK4[0], "Merde, je ne peux pas ce soir.", *OK4[2:]), "moderne", "vocabulaire_cru"),
    (tours(OKM[0], "Fort bien, et toi, mon ami ?", *OKM[2:]), "moliere", "tutoiement"),
    (tours(OKM[0], "Ma foi, t'es bien curieux.", *OKM[2:]), "moliere", "tutoiement"),
    (tours(OKM[0], "Ma foi, c'est super, je vous remercie.", *OKM[2:]), "moliere", "mot_moderne"),
    (tours(OKM[0], "Du coup je vous remercie fort.", *OKM[2:]), "moliere", "mot_moderne"),
    # tutoiement à l'impératif, sans pronom (cas relevés dans l'échantillon v1)
    (tours(OKM[0], "Alors attends, car l'amour ne se commande point.", *OKM[2:]), "moliere", "tutoiement"),
    (tours(OKM[0], "Ne me dis pas que c'est une médecine nouvelle!", *OKM[2:]), "moliere", "tutoiement"),
    (tours(OKM[0], "Ne les écoute pas, et demande à un homme de bonne foi.", *OKM[2:]), "moliere",
     "tutoiement"),
    # « je m'en fous » et « foutre » sont du vocabulaire cru
    (tours(OK4[0], "Bon, on fait comme tu veux, je m'en fous, c'est toi qui décides.", *OK4[2:]),
     "moderne", "vocabulaire_cru"),
    (tours(OK4[0], "Il se fout de nous depuis ce matin.", *OK4[2:]), "moderne", "vocabulaire_cru"),
    # le personnage recopie une réplique d'exemple ou une démonstration du prompt (6 mots de suite)
    (tours(OKM[0], CFG["exemples_moliere"][0]["texte"], *OKM[2:]), "moliere", "copie_exemple"),
    (tours(OKM[0], "Volontiers. Qui se tait sur son mérite le voit bientôt oublié.", *OKM[2:]), "moliere",
     "copie_exemple"),
    # l'utilisateur paraphrase la réplique précédente (40 % de ses mots) ou en reprend le début
    (tours("J'ai froid ce matin.", "Le froid est une maladie de l'âme, croyez-moi, et il passera vite.",
           "Le froid, une maladie de l'âme ? Je ne crois pas.", "Couvrez-vous donc, et ne songez plus à l'hiver."),
     "moliere", "echo"),
    (tours("Je me sens perdu.", "C'est comme si vous perdiez la tête, mais cela passera bientôt.",
           "C'est comme si je tombais du ciel.", "Asseyez-vous donc, et reprenez vos esprits."),
     "moliere", "echo"),
    # même réplique deux fois dans une conversation
    (tours("Tu viens ce soir ?", "Je ne sais pas encore, je dois voir avec ma sœur.",
           "D'accord, préviens-moi vite.", "Je ne sais pas encore, je dois voir avec ma sœur."),
     "moderne", "tour_repete"),
])
def test_filtres(conv, style, raison):
    assert syn.raison_rejet(conv, [], style, CFG) == raison


def test_a_verbe_avoir_n_est_pas_un_nom_de_role():
    conv = tours("Il a dit qu'il viendrait demain.", "Il a précisé l'heure ?",
                 "Vers midi je crois.", "Parfait, je serai là.")
    assert syn.raison_rejet(conv, [], "moderne", CFG) is None


def test_refiltrer_applique_les_filtres_actuels(tmp_path):
    out = tmp_path / "moderne.jsonl.zst"
    docs, _ = syn.generer("moderne", 6, out, seed=2, generateur=FauxGenerateur("moderne"), cfg=CFG,
                          log=lambda s: None)
    docs[1]["turns"][1]["text"] = "Salut B, quelle merde ce matin."  # passé avant un nouveau filtre
    write_docs(out, docs)
    gardes, rapport = syn.refiltrer("moderne", out, cfg=CFG)
    assert len(gardes) == 5 and docs[1] not in gardes
    assert rapport["refiltrage"] == {"meta": 1} and rapport["gardes"] == 5
    assert list(read_docs(out)) == gardes


def test_tutoiement_et_mots_modernes_permis_chez_l_utilisateur():
    conv = tours("Salut, t'es là ? C'est super cool ici.", "Je suis là, monsieur, que voulez-vous ?",
                 "Ok, tu peux m'aider ?", "Volontiers, dites-moi ce qu'il vous faut.")
    assert syn.raison_rejet(conv, [], "moliere", CFG) is None
    # ... mais le tutoiement n'est pas filtré en style moderne
    assert syn.raison_rejet(tours(*OK4), [], "moderne", CFG) is None


def test_tutoiement_sans_faux_positif():
    for t in ["Je vous dis que non.", "Il demande pardon.", "Je crois que cela va.",
              "Je le veux, vous dis-je.", "Croyez-moi, monsieur.", "Dites-moi tout.", "Il me regarde."]:
        assert not syn.tutoie(t), t
    for t in ["Dis-moi tout.", "Ôte-toi de là.", "Va, je ne te hais point.", "Mais regarde donc !"]:
        assert syn.tutoie(t), t


def test_filtres_qui_dependent_du_prompt():
    """ctx = ce que le prompt demandait : registre, genre de A, personnage, sujet."""
    vous = tours("Bonjour, pourriez-vous me dire le chemin de la gare ?", "Bien sûr, tu es où exactement ?",
                 "Je suis devant la poste.", "Alors c'est tout droit, à deux minutes.")
    assert syn.raison_rejet(vous, [], "moderne", CFG, {"registre": "vous_court"}) == "registre"
    assert syn.raison_rejet(vous, [], "moderne", CFG) is None  # sans contexte : pas de vérification
    imperatif = tours("Bonjour, vous avez l'heure ?", "Attends, je regarde mon téléphone.",
                      "Merci, c'est gentil.", "Il est midi pile, bonne journée.")
    assert syn.raison_rejet(imperatif, [], "moderne", CFG, {"registre": "vous_long"}) == "registre"
    tout_vous = tours("Bonjour, vous allez bien ?", "Très bien, merci, et vous ?",
                      "Ça va, merci beaucoup.", "Bonne journée à vous.")
    assert syn.raison_rejet(tout_vous, [], "moderne", CFG, {"registre": "tu_court"}) == "registre"
    assert syn.raison_rejet(tout_vous, [], "moderne", CFG, {"registre": "vous_court"}) is None

    monsieur = tours("Je me suis perdue en ville.", "Je suis là, monsieur, que voulez-vous ?",
                     "Je cherche mon chemin.", "Suivez donc cette rue jusqu'à l'église.")
    assert syn.raison_rejet(monsieur, [], "moliere", CFG,
                            {"interlocuteur": "ELMIRE", "persona": "TARTUFFE"}) == "genre"
    assert syn.raison_rejet(monsieur, [], "moliere", CFG,
                            {"interlocuteur": "LA GRANGE", "persona": "MOLIÈRE"}) is None
    madame = tours("Je me suis perdu en ville.", "Madame, que voulez-vous que j'y fasse ?",
                   "Je cherche mon chemin.", "Suivez donc cette rue jusqu'à l'église.")
    assert syn.raison_rejet(madame, [], "moliere", CFG,
                            {"interlocuteur": "LA GRANGE", "persona": "MOLIÈRE"}) == "genre"
    assert syn.raison_rejet(tours(*OKM), [], "moliere", CFG, {"persona": "HARPAGON"}) == "persona_exclu"

    sujet = "A a dû manger chez des amis un plat qu'il n'aime pas"
    copie = tours("J'ai mangé un plat qu'il n'aime pas chez eux.", OKM[1], OKM[2], OKM[3])
    assert syn.raison_rejet(copie, [], "moliere", CFG, {"sujet": sujet}) == "recopie_sujet"
    je = tours("J'ai dû manger un plat que je n'aime pas.", OKM[1], OKM[2], OKM[3])
    assert syn.raison_rejet(je, [], "moliere", CFG, {"sujet": sujet}) is None
    # registre de A en style Molière : « vous » demandé, A tutoie -> rejet
    assert syn.raison_rejet(tours(*OKM), [], "moliere", CFG, {"registre": "vous_court"}) == "registre"


def test_dedup_tour_repete_ouvertures_et_seuil_proportionnel():
    d = syn.Dedup(max_convs=3, ouverture_max=0.15, echauffement=20, style="moliere")
    reponse = "Je vous dirai ce que j'en pense demain matin."
    d.ajoute(["Salut.", reponse])
    assert d.raison(["Bonjour.", reponse]) == "tour_repete"  # même réplique dans 2 conversations
    # 15 % de 20 réponses = 3 réponses du personnage qui commencent par « Ah »
    for i in range(2):
        conv = ["Bonjour.", f"Ah, voilà une nouvelle que je n'attendais point, numéro {i}."]
        assert d.raison(conv) is None
        d.ajoute(conv)
    assert d.raison(["Re.", "Ah, la belle affaire que voilà, en vérité, monsieur."]) is None
    d.ajoute(["Re.", "Ah, la belle affaire que voilà, en vérité, monsieur."])
    assert d.raison(["Encore.", "Ah, que me dites-vous là, et pourquoi donc ?"]) == "ouverture_repetee"
    # la courte question en écho (« Fatigué ? ») compte comme une seule ouverture
    assert syn.ouverture("Fatigué ? C'est une faiblesse de l'esprit.") == "<question courte>"
    assert syn.ouverture("Que voulez-vous que je fasse de toutes ces belles choses ?") == "que"
    for i in range(3):
        d.ajoute(["Q.", f"Trop chaud {i} ? Mettez-vous donc à l'ombre, numéro {i}."])
    assert d.raison(["Q.", "Gelé ? Couvrez-vous donc de votre manteau."]) == "ouverture_repetee"
    # seuil des formules : 3, puis 3 pour 1 000 conversations gardées
    p = syn.Dedup(max_convs=3, pour_mille=3)
    p.ngrams[("il", "fait", "vraiment", "très", "beau")] = 3
    assert p.raison(["il fait vraiment très beau"]) == "quasi_doublon"
    p.exacts |= {f"conv {i}" for i in range(2000)}  # 2 000 gardées -> seuil 6
    assert p.raison(["il fait vraiment très beau"]) is None


def test_dedup_exact_et_formules():
    d = syn.Dedup(max_convs=3)
    base = ["Il fait vraiment très beau aujourd'hui", "Oui"]
    assert d.raison(base) is None
    d.ajoute(base)
    assert d.raison(["il fait vraiment, très beau aujourd'hui!", "oui."]) == "doublon_exact"
    for i in range(2):  # même formule de 5 mots dans 3 conversations au total : encore permis
        conv = [f"Je pense que tu as raison {i}", f"Il fait vraiment très beau aujourd'hui donc {i}"]
        assert d.raison(conv) is None
        d.ajoute(conv)
    assert d.raison(["Bref, il fait vraiment très beau aujourd'hui", "Ah"]) == "quasi_doublon"
    # un 5-gramme ne chevauche pas deux tours
    assert d.raison(["il fait vraiment", "très beau aujourd'hui"]) is None


# ---------------------------------------------------------------- 4. contrat et reprise
@pytest.mark.parametrize("style", ["moderne", "moliere"])
def test_contrat(tmp_path, style):
    gen = FauxGenerateur(style)
    out = tmp_path / f"{style}.jsonl.zst"
    docs, rapport = syn.generer(style, 5, out, seed=4, generateur=gen, cfg=CFG, log=lambda s: None)
    lus = list(read_docs(out))
    assert lus == docs and len(lus) == 5
    assert rapport["gardes"] == 5 and rapport["appels"] == 2 and rapport["rejets"] == {}
    rapport_disque = json.loads((tmp_path / f"{style}.rapport.json").read_text())
    assert rapport_disque["next_call"] == 2
    assert (tmp_path / f"{style}.journal.log").read_text().count("tokens/s") == 2
    assert len({d["id"] for d in lus}) == 5
    for d in lus:
        assert {"id", "source", "doc_id", "url", "license", "style", "turns", "generator"} <= set(d)
        assert d["source"] == "synthetique_qwen3" and d["style"] == style and d["url"] is None
        assert set(d["generator"]) == {"model", "prompt_id", "seed", "temperature"}
        assert d["generator"]["model"] == "qwen3:8b"
        assert all("\n" not in t["text"] and t["text"] for t in d["turns"])
        speakers = [t["speaker"] for t in d["turns"]]
        if style == "moderne":
            assert speakers == ["A", "B"] * (len(speakers) // 2) + ["A"] * (len(speakers) % 2)
        else:
            p = next(p for p in PERSONAS if p["name"] == d["persona"])
            assert speakers == [p["interlocutor"], p["name"]] * (len(speakers) // 2)
    # le prompt envoyé est exactement celui de construire_appel, avec sa graine Ollama
    prompt, meta = syn.construire_appel(style, 4, 0, CFG)
    assert gen.prompts[0] == (prompt, {"temperature": CFG["generateur"]["temperature"],
                                       "num_predict": CFG["num_predict"][style],
                                       "seed": meta["ollama_seed"]})


def test_reprise_continue_sans_refaire(tmp_path):
    out = tmp_path / "moderne.jsonl.zst"
    gen = FauxGenerateur("moderne")
    premiers, _ = syn.generer("moderne", 4, out, seed=9, generateur=gen, cfg=CFG, log=lambda s: None)
    gen2 = FauxGenerateur("moderne")
    gen2.n = 1000  # conversations nouvelles
    docs, rapport = syn.generer("moderne", 9, out, seed=9, generateur=gen2, cfg=CFG, log=lambda s: None)
    assert docs[:4] == premiers and len(docs) == 9
    # les appels déjà faits (0 et 1) ne sont pas refaits : la reprise commence à l'appel 2
    assert gen2.prompts[0][0] == syn.construire_appel("moderne", 9, 2, CFG)[0]
    assert rapport["appels"] == 4 and rapport["next_call"] == 4
    assert not list(tmp_path.glob("*.tmp"))  # écriture atomique : aucun fichier temporaire laissé
    with pytest.raises(SystemExit):  # autre graine sur le même fichier : refus
        syn.generer("moderne", 12, out, seed=10, generateur=gen2, cfg=CFG, log=lambda s: None)


def test_reprise_sans_rapport_se_cale_sur_les_documents(tmp_path):
    out = tmp_path / "moliere.jsonl.zst"
    syn.generer("moliere", 3, out, seed=1, generateur=FauxGenerateur("moliere"), cfg=CFG,
                log=lambda s: None)
    (tmp_path / "moliere.rapport.json").unlink()
    gen = FauxGenerateur("moliere")
    gen.n = 500
    syn.generer("moliere", 4, out, seed=1, generateur=gen, cfg=CFG, log=lambda s: None)
    assert gen.prompts[0][0] == syn.construire_appel("moliere", 1, 1, CFG)[0]


def test_rejets_comptes_par_raison_et_garde_fou(tmp_path):
    def mauvais(prompt, options):
        return {"response": "A: Hello, how are you with the kids?\nB: Fine, thank you and you?\n"
                            "A: Good.\nB: Great to hear that, see you.",
                "eval_count": 10, "eval_duration": 1e9}
    docs, rapport = syn.generer("moderne", 2, tmp_path / "x.jsonl.zst", seed=1, generateur=mauvais,
                                cfg=CFG, max_appels=3, log=lambda s: None)
    assert docs == [] and rapport["appels"] == 3
    assert rapport["rejets"]["anglais"] == 3 and rapport["rejets"]["nombre_de_blocs"] == 3
    assert rapport["taux_rejet"] == 1.0


# ---------------------------------------------------------------- 5. choix des exemples Molière
def test_choix_exemples_moliere(tmp_path):
    scenes = [{"id": "x", "play": "PIÈCE", "doc_id": "PIÈCE", "turns": [
        {"speaker": "SGANARELLE", "text": "Ma foi, monsieur, je ne sais point ce que vous voulez dire là."},
        {"speaker": "VALÈRE", "text": "Sganarelle, viens ici, que je te parle de cette affaire."},
        {"speaker": "VALÈRE", "text": "Trop court."},
        {"speaker": "VALÈRE", "text": "Il étoit une fois un homme qui aimoit fort son argent."},
        {"speaker": "VALÈRE", "text": "Je vous dis que je partirai demain pour la ville de Lyon."},
        {"speaker": "VALÈRE", "text": "Je vous en prie, monsieur, ne me parlez plus de cette affaire"},
        {"speaker": "VALÈRE", "text": "Le fait est, monsieu, que je venons vous prier de nous aider."},
        {"speaker": "VALÈRE", "text": "Allons, allons, je suis votre servante, et je n'ai pas envie de rire."},
        # écartées : tutoiement (pronom, impératif), « Ah » en tête, didascalie, patois
        {"speaker": "VALÈRE", "text": "Quoi! tu fais le surpris et demandes encore ton crime!"},
        {"speaker": "VALÈRE", "text": "Ne me dis pas que vous êtes encore fâché contre moi."},
        {"speaker": "VALÈRE", "text": "Ah! que je suis aise de vous voir enfin en ce logis."},
        {"speaker": "VALÈRE", "text": "Je ne parle pas. A part, après s'être donné des soufflets, voilà."},
        {"speaker": "VALÈRE", "text": "J'ai des gairans de tout cho que je di, et bien des gens."},
    ]}]
    path = tmp_path / "m.jsonl.zst"
    write_docs(path, scenes)
    textes = [e["texte"] for e in syn.choisir_exemples_moliere(path, n=10, max_par_piece=10)]
    assert sorted(textes) == sorted([
        "Ma foi, monsieur, je ne sais point ce que vous voulez dire là.",
        "Allons, allons, je suis votre servante, et je n'ai pas envie de rire."])


# ---------------------------------------------------------------- 6. prompts v2 et juge
def test_prompts_v2_registre_genre_personnages():
    regs = {r["id"]: r for r in CFG["registres"]}
    for style in ("moderne", "moliere"):
        for call in range(30):
            prompt, meta = syn.construire_appel(style, 1, call, CFG)
            assert len(set(meta["registres"])) == 1  # un seul registre par appel
            assert regs[meta["registres"][0]][style] in prompt
    vus = set()
    for call in range(400):
        prompt, meta = syn.construire_appel("moliere", 2, call, CFG)
        vus.add(meta["persona"])
        genre = CFG["genres"][CFG["interlocuteurs"][meta["interlocutor"]]]
        assert genre["texte"] in prompt and f"« {genre['titre']} »" in prompt
        assert sum(d["b"] in prompt for d in CFG["demonstrations_moliere"]) == CFG["nb_demonstrations_par_prompt"]
    assert not vus & set(CFG["personas_exclus"])  # Harpagon et Alceste : pièces de validation
    assert "TARTUFFE" in vus  # dont l'interlocutrice est ELMIRE (« madame »)


def test_exemples_et_demonstrations_sans_tutoiement():
    for e in CFG["exemples_moliere"]:
        assert not syn.tutoie(e["texte"]), e["texte"]
        assert syn.mots(e["texte"])[0] not in ("ah", "ha")
    for d in CFG["demonstrations_moliere"]:
        assert not syn.tutoie(d["b"]) and syn.mots(d["b"])[0] != "ah"


def test_juge_prompt_et_lecture_du_verdict():
    conv = tours(*OK4)
    ctx = {"registre": "vous_court"}
    prompt = syn.prompt_juge("moderne", conv, ctx, CFG)
    assert syn.est_prompt_juge(prompt, CFG) and "{" not in prompt
    assert "A: " + OK4[0] in prompt and CFG["gabarits"]["juge"]["question_moderne"] in prompt
    p2 = syn.prompt_juge("moliere", tours(*OKM), {"persona": "TARTUFFE", "persona_label": "Tartuffe (Le Tartuffe)"},
                         CFG)
    assert "Tartuffe (Le Tartuffe)" in p2
    for texte, attendu in [("1: OUI\n2: OUI\n3: OUI", "OUI"), ("1: OUI\n2: NON\n3: OUI", "NON"),
                           ("<think>\n</think>\n1. oui\n2. oui\n3. oui", "OUI"),
                           ("NON : aucun défaut", None), ("1: OUI\n2: OUI", None)]:
        recu = []
        gen = lambda p, o, recu=recu, texte=texte: recu.append(o) or {"response": texte}
        assert syn.juger("moderne", conv, ctx, CFG, gen)[0] == attendu, texte
        assert recu[0]["temperature"] == CFG["juge"]["temperature"]


@pytest.mark.parametrize("verdict,raison", [("1: OUI\n2: NON\n3: OUI", "juge"), ("Je ne sais pas.", "juge_illisible")])
def test_generer_rejette_ce_que_le_juge_refuse(tmp_path, verdict, raison):
    gen = FauxGenerateur("moderne", verdict=verdict)
    docs, rapport = syn.generer("moderne", 3, tmp_path / "j.jsonl.zst", seed=1, generateur=gen, cfg=CFG,
                                max_appels=2, log=lambda s: None, juge=True)
    assert docs == [] and rapport["rejets"] == {raison: 6}
    assert rapport["appels_juge"] == 6 == len(gen.jugements) and rapport["juge"] == CFG["gabarits"]["juge"]["id"]
    assert len(rapport["juge_rejets_exemples"]) == 6
    assert "juge " in (tmp_path / "j.journal.log").read_text()
    # sans juge, les mêmes conversations sont gardées et aucun prompt de juge n'est envoyé
    gen2 = FauxGenerateur("moderne", verdict=verdict)
    docs, _ = syn.generer("moderne", 3, tmp_path / "k.jsonl.zst", seed=1, generateur=gen2, cfg=CFG,
                          log=lambda s: None, juge=False)
    assert len(docs) == 3 and gen2.jugements == []


def test_refiltrer_relit_le_contexte_des_documents(tmp_path):
    out = tmp_path / "moliere.jsonl.zst"
    docs, _ = syn.generer("moliere", 4, out, seed=3, generateur=FauxGenerateur("moliere"), cfg=CFG,
                          log=lambda s: None)
    docs[0]["persona"] = "HARPAGON"  # comme un échange produit avant l'exclusion des personnages
    write_docs(out, docs)
    gardes, rapport = syn.refiltrer("moliere", out, cfg=CFG)
    assert len(gardes) == 3 and rapport["refiltrage"] == {"persona_exclu": 1}


def test_juge_actif_par_style(tmp_path):
    """Juge actif par défaut pour Molière seulement (mesure : il ne voyait rien en style moderne)."""
    assert CFG["juge"]["actif"] == {"moderne": False, "moliere": True}
    for style, attendu in (("moderne", 0), ("moliere", 4)):
        gen = FauxGenerateur(style)
        syn.generer(style, 4, tmp_path / f"{style}.jsonl.zst", seed=1, generateur=gen, cfg=CFG, log=lambda s: None)
        assert len(gen.jugements) == attendu


def test_tics_plafonnes_et_recopie_sujet_a_la_troisieme_personne():
    d = syn.Dedup(max_convs=3, ouverture_max=0.15, echauffement=20, style="moliere", tics=["c'est comme"])
    for i, debut in enumerate(["Partir", "Rester", "Chanter"]):  # ouvertures différentes, même tic
        conv = ["Bonjour.", f"{debut} ainsi, c'est comme un voyage sans fin, numéro {i}."]
        assert d.raison(conv) is None
        d.ajoute(conv)
    assert d.raison(["Re.", "Dormir, c'est comme mourir un peu chaque soir."]) == "tic_repete"
    assert d.nb_reponses == 3  # les tics ne comptent pas comme des réponses
    # « J'hésite à acheter une voiture d'occasion » : reprise du sujet sans 3e personne -> permise
    sujet = "A hésite à acheter une voiture d'occasion"
    conv = tours("J'hésite à acheter une voiture d'occasion.", OKM[1], OKM[2], OKM[3])
    assert syn.raison_rejet(conv, [], "moliere", CFG, {"sujet": sujet}) is None
