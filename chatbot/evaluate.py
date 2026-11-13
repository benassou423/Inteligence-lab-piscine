"""Banc d'évaluation du chatbot : c'est lui qui dit si une version progresse.

    uv run python -m chatbot.evaluate --model data/models/chat_v1 --out data/stages/chatbot_20261112/eval/v1
    uv run python -m chatbot.evaluate --model submissions/2026-11-11_153839/ckpt.pt \\
        --tokenizer submissions/2026-11-11_153839/tokenizer.json --baseline-prompt \\
        --out data/stages/chatbot_20261112/eval/baseline_demo_actuelle

Deux mesures complémentaires :
1. BPB « réplique suivante » sur moliere_val (pièces jamais vues), même calcul que l'affinage :
   le modèle prédit-il bien le texte de théâtre au format du chat ?
2. Un panel FIXE de 40 messages (graines fixes ; +10 messages joués par HARPAGON en mode format) :
   on génère et on mesure automatiquement des défauts visibles (réponses vides, coupées,
   répétitives, échos, hors sujet...). Les réponses sont aussi écrites dans transcript.md pour une
   relecture humaine : les chiffres ne suffisent pas.

--baseline-prompt : même panel avec l'ancien prompt de la démo (« Conversation en français... »,
température 0,8, top-k 40, 128 tokens) pour mesurer l'état « avant ». La baseline n'a pas de
personnage : on compare avant/après sur le groupe « commun_sans_harpagon » (mêmes 40 messages,
mêmes graines), jamais sur « tout ». Deux metrics.json ne se comparent que si leur code_sha256
est identique (même code de mesure et de génération).

--mmi-lambda, --candidates, --temperature, --top-p, --no-repeat-ngram, --persona : surchargent le
chat_config.json du modèle, pour comparer des réglages de décodage (p. ex. un témoin
--candidates 1 --no-repeat-ngram 0 sépare l'effet du format de celui du décodage).

Format « tirets » (chatbot en français moderne, lu dans chat_config.json) : même panel de 40
messages (groupe commun identique), sans personnage donc sans série HARPAGON ; la validation par
défaut devient romans_val (dialogues A/B), rendue au format tirets.

Deux mesures pour les vraies discussions (les deux formats) :
3. « gain de l'historique » : sur les tours 2 et 3 d'une conversation, moyenne par token de
   log P(réponse retenue | tout l'historique) − log P(réponse retenue | dernier message seul).
   Positif : le modèle se sert vraiment des tours précédents ; ~0 : il ne lit que le dernier.
4. une série de 6 discussions de 6 tours (français courant) : réponses qui recopient une réponse
   précédente du bot, conversations dont le contexte est tronqué, longueur (--no-discussion : sautée).
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from chatbot.format import DEFAULT_PERSONA, FORMATS, normalize_speaker, persona_info
from chatbot.generate import DEFAULT_CHAT_CONFIG, ChatBot, echo_ratio
from ftb.tokenizer import file_sha256

DEFAULT_VAL = "data/stages/chatbot_20261112/theatre/moliere_val.jsonl.zst"
# Validation par défaut selon le format du modèle (--val la remplace).
DEFAULT_VALS = {"repliques": DEFAULT_VAL,
                "tirets": "data/stages/chatbot_20261113/dialogues/romans_val.jsonl.zst"}
SEEDS = [11, 22, 33]

# 25 messages isolés (par thème) + 5 conversations de 3 tours = 40 messages utilisateur.
SINGLES = [
    ("salutation", "Salut, ça va ?"),
    ("salutation", "Bonjour, comment allez-vous aujourd'hui ?"),
    ("question", "Quel est votre nom ?"),
    ("question", "Quelle heure est-il ?"),
    ("question", "Où habitez-vous ?"),
    ("emotion", "Je suis très triste ce soir."),
    ("emotion", "Je suis tellement en colère contre mon frère !"),
    ("emotion", "J'ai peur de l'avenir."),
    ("argent", "Pouvez-vous me prêter dix écus ?"),
    ("argent", "Que pensez-vous de l'argent ?"),
    ("argent", "J'ai perdu ma bourse ce matin."),
    ("medecine", "Je suis malade, que dois-je faire ?"),
    ("medecine", "Faut-il croire les médecins ?"),
    ("amour", "Je suis amoureux de votre fille."),
    ("amour", "Voulez-vous m'épouser ?"),
    ("hypocrisie", "Mon voisin fait le dévot mais il ment à tout le monde."),
    ("hypocrisie", "Faut-il toujours dire la vérité, même quand elle blesse ?"),
    ("insulte", "Vous êtes un vieil avare !"),
    ("insulte", "Tais-toi, imbécile."),
    ("conseil", "Donnez-moi un conseil pour réussir ma vie."),
    ("conseil", "Mon père veut me marier contre mon gré, que faire ?"),
    ("moderne", "Que penses-tu d'Internet ?"),
    ("moderne", "As-tu un téléphone portable ?"),
    ("court", "chaise"),
    ("court", "Pourquoi ?"),
]
CONVERSATIONS = [
    ("conversation", ["Bonjour, monsieur.", "Comment se porte votre famille ?", "Et vos affaires ?"]),
    ("conversation", ["J'ai une mauvaise nouvelle.", "On vous a volé votre cassette.",
                      "Je ne sais pas qui est le voleur."]),
    ("conversation", ["Je me sens malade.", "J'ai mal à la tête depuis trois jours.",
                      "Faut-il appeler un médecin ?"]),
    ("conversation", ["Je veux me marier.", "Elle s'appelle Mariane.", "Mais elle n'a pas de dot."]),
    ("conversation", ["Que pensez-vous de la cour ?", "Les courtisans sont-ils sincères ?",
                      "Alors, faut-il fuir le monde ?"]),
]
# 6 discussions de 6 tours, en français courant, sur des sujets variés : tient-il une vraie
# conversation (ne pas se recopier, garder le fil) ? Rejouées avec les mêmes graines (SEEDS).
DISCUSSIONS = [
    ("vacances", ["Salut ! Tu as passé un bon week-end ?", "Moi, je suis allé à la mer avec ma sœur.",
                  "Il a fait très beau, mais l'eau était froide.", "Et toi, tu aimes nager ?",
                  "Où est-ce que tu voudrais partir cet été ?", "Pourquoi cet endroit-là ?"]),
    ("cuisine", ["J'ai envie de faire un gâteau ce soir.", "Je pensais à un gâteau au chocolat.",
                 "Mais je n'ai plus de beurre.", "Tu crois que je peux mettre de l'huile à la place ?",
                 "Combien de temps faut-il le laisser au four ?",
                 "Merci, je te dirai si c'est réussi."]),
    ("examen", ["Demain, j'ai un examen de mathématiques.", "J'ai peur de ne pas avoir assez révisé.",
                "Surtout les exercices sur les probabilités.", "Tu as un conseil pour bien dormir ?",
                "Et le matin, qu'est-ce que je dois manger ?", "D'accord, je vais essayer de rester calme."]),
    ("animal", ["Mon chien est malade depuis hier.", "Il ne veut plus manger ses croquettes.",
                "Le vétérinaire est fermé le dimanche.", "Est-ce que je dois m'inquiéter ?",
                "Je vais lui donner un peu d'eau en attendant.", "Tu as déjà eu un animal, toi ?"]),
    ("demenagement", ["Je viens de déménager à Lyon.", "Je ne connais encore personne ici.",
                      "Le quartier est joli, mais un peu bruyant.", "Tu prends souvent le métro ?",
                      "Qu'est-ce que je pourrais visiter ce week-end ?",
                      "Bonne idée, j'irai avec un collègue."]),
    ("cinema", ["Hier soir, j'ai vu un film au cinéma.", "C'était une histoire d'amour à Paris.",
                "La fin m'a fait pleurer.", "Tu préfères les films tristes ou les comédies ?",
                "Quel est le dernier livre que tu as aimé ?", "Tu me le prêteras un jour ?"]),
]
# HARPAGON répond aussi aux 10 premiers messages isolés. L'Avare est une pièce de validation : pour
# un modèle affiné sans elle, c'est un test de généralisation sur un rôle jamais vu.
HARPAGON_MESSAGES = 10
ECHO_THRESHOLD = 0.6  # écho : au moins 60 % des mots du message se retrouvent dans la réponse
CUT_WORDS = 15        # répétition et diversité aussi mesurées sur les 15 premiers mots (longueur égale)

# Mots vides de 5 lettres ou plus (le taux de reprise ne compte que les mots « pleins »).
STOPWORDS = set("""alors aussi autre autres avant avoir avait avais avez aviez beaucoup bien cela celle
celles celui cette ceux chaque chez comme comment contre dans depuis donc encore entre est-ce faire
faites fait jamais leurs mêmes moins notre nôtre parce pendant peut-être plusieurs pourquoi pouvez
quand quelle quelles quelque quelques quels rien sans selon serait seront sinon sommes soient toujours
toute toutes trop très vôtre votre vous-même voulez voudrais vraiment était étais étaient êtes être
monsieur madame point voici voilà ainsi assez certes après aujourd chose choses quoi
salut bonjour bonsoir tellement pensez penses donnez allez dites""".split())
# Mots d'adresse (salut, pensez, donnez...) ajoutés : les reprendre n'est pas répondre au sujet.
WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+")
# Nom de rôle parasite : une ligne entière en capitales (« CLÉANTE. », « M. LOYAL. ») ou un
# « Utilisateur : ».
ROLE_LINE = re.compile(r"^\s*[A-ZÀ-ÖØ-ÞŒ][A-ZÀ-ÖØ-ÞŒ'’ \-.]{2,40}\.?\s*$", re.M)
CHAT_ROLE = re.compile(r"^\s*(utilisateur|assistant|user|humain)\s*:", re.M | re.I)
CLEAN_END = re.compile(r"[.!?…][»\")\]]*\s*$")


# ---------------------------------------------------------------------------------------------
# Mesures automatiques


def words(text):
    return WORD.findall(text.lower().replace("’", "'"))


def content_words(text):
    return {w for w in words(text) if len(w) >= 5 and w not in STOPWORDS}


def repetition(text):
    """Part des 3-grammes de mots qui sont des répétitions dans la réponse (0 = aucune)."""
    w = words(text)
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def distinct_2(texts):
    """Diversité : bigrammes de mots distincts / bigrammes au total, sur toutes les réponses."""
    grams = [tuple(w[i:i + 2]) for t in texts for w in [words(t)] for i in range(len(w) - 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def first_words(text, n=CUT_WORDS):
    return " ".join(words(text)[:n])


def is_echo(reply, message):
    """Écho : la réponse recopie au moins 60 % des mots du message (même mesure que le chat)."""
    return echo_ratio(message, reply) >= ECHO_THRESHOLD


def has_role(text):
    return bool(ROLE_LINE.search(text) or CHAT_ROLE.search(text))


def invented_role(raw, interlocutor):
    """Rôle inventé dans le texte BRUT, hors passage normal de la parole.

    Passer la parole est l'arrêt prévu : à l'interlocuteur attendu (« LA GRANGE. ») dans le nouveau
    format, à « Utilisateur : » dans l'ancien prompt (interlocutor None). Le compter comme « rôle
    inventé » recopierait natural_stop_rate. Tout AUTRE nom compte : un tiers, ou le persona qui
    reprend la parole après une ligne vide.
    """
    if interlocutor is None:  # ancien prompt : son « Assistant : » de tête n'est pas inventé
        raw = re.sub(r"^\s*Assistant\s*:\s*", "", raw, count=1, flags=re.I)
        allowed_chat = {"utilisateur"}
    else:
        allowed_chat = set()
    if any(normalize_speaker(m) != interlocutor for m in ROLE_LINE.findall(raw)):
        return True
    return any(m.lower() not in allowed_chat for m in CHAT_ROLE.findall(raw))


def mean_history_gain(rows, turns):
    """Gain de l'historique moyen PAR TOKEN sur les tours `turns` (0 = premier tour) :
    somme des écarts de log-probabilité / nombre total de tokens notés. None si rien à mesurer."""
    sel = [r for r in rows if r.get("turn") in turns and r.get("history_gain") is not None]
    ntok = sum(r["history_gain_tokens"] for r in sel)
    if not ntok:
        return None
    return round(sum(r["history_gain"] * r["history_gain_tokens"] for r in sel) / ntok, 4)


def summarize(rows):
    replies = [r["reply"] for r in rows]
    nonempty = [t for t in replies if t.strip()]
    scores = [r["score"] for r in rows if r.get("score") is not None]
    secs = [r["seconds"] for r in rows]
    n = max(len(rows), 1)
    # Reprise du sujet : seulement sur les messages qui ont au moins un mot plein (« Pourquoi ? »
    # n'en a aucun : la réponse vaudrait 0 quoi qu'elle dise).
    eligible = [(t, r) for t, r in zip(replies, rows) if content_words(r["message"])]
    n_elig = max(len(eligible), 1)
    raws = [r.get("raw_reply") for r in rows]
    ratios = [len(t) / r["candidates_mean_chars"] for t, r in zip(replies, rows)
              if r.get("candidates_mean_chars")]
    return {
        "n": len(rows),
        "empty_rate": round(sum(not t.strip() for t in replies) / n, 4),
        "mean_chars": round(statistics.mean(len(t) for t in replies), 1) if replies else 0,
        "clean_end_rate": round(sum(bool(CLEAN_END.search(t)) for t in nonempty) / n, 4),
        "repetition_rate": round(statistics.mean(repetition(t) for t in replies), 4) if replies else 0,
        "distinct_2": round(distinct_2(replies), 4),
        "role_name_rate": round(sum(has_role(t) for t in replies) / n, 4),
        "overlap_rate": round(sum(bool(content_words(t) & content_words(r["message"]))
                                  for t, r in eligible) / n_elig, 4),
        "overlap_eligible": len(eligible),
        "overlap_non_echo_rate": round(sum(bool(content_words(t) & content_words(r["message"]))
                                           and not is_echo(t, r["message"])
                                           for t, r in eligible) / n_elig, 4),
        "echo_rate": round(sum(is_echo(t, r["message"]) for t, r in zip(replies, rows)) / n, 4),
        "mean_words": round(statistics.mean(len(words(t)) for t in replies), 1) if replies else 0,
        "repetition_rate_15w": (round(statistics.mean(repetition(first_words(t)) for t in replies), 4)
                                if replies else 0),
        "distinct_2_15w": round(distinct_2([first_words(t) for t in replies]), 4),
        "natural_stop_rate": round(sum(r.get("finish") in ("eot", "speaker") for r in rows) / n, 4),
        "role_leak_raw_rate": (round(sum(invented_role(x, r.get("interlocutor"))
                                         for x, r in zip(raws, rows) if x is not None)
                                     / max(sum(x is not None for x in raws), 1), 4)
                               if any(x is not None for x in raws) else None),
        "chosen_vs_candidates_length": round(statistics.mean(ratios), 4) if ratios else None,
        "mean_mmi_score": round(statistics.mean(scores), 4) if scores else None,
        "seconds_per_reply": round(statistics.mean(secs), 3) if secs else None,
        "seconds_median": round(statistics.median(secs), 3) if secs else None,
        "finish": dict(Counter(r.get("finish") for r in rows)),
        "context_truncated": sum(bool(r.get("context_truncated")) for r in rows),
        # Tours 2 et 3 (turn = 1, 2) des conversations ; None sans conversation ou en baseline.
        "history_gain_turns_2_3": mean_history_gain(rows, (1, 2)),
        "history_gain_n": sum(r.get("turn") in (1, 2) and r.get("history_gain") is not None
                              for r in rows),
    }


def summarize_discussion(rows):
    """Tenue d'une vraie discussion (série de 6 discussions de 6 tours).

    Une conversation = (job, base_seed). On y mesure : la recopie d'une réponse précédente du bot
    (>= 60 % de ses mots, même mesure que l'écho), les conversations dont le contexte a été
    tronqué (le modèle a perdu le début), la longueur, et le gain de l'historique.
    """
    convs = {}
    for r in rows:
        convs.setdefault((r["job"], r["base_seed"]), []).append(r)
    copies = eligible = 0
    last_ctx = []
    for conv in convs.values():
        conv.sort(key=lambda r: r["turn"])
        last_ctx.append(conv[-1].get("context_tokens") or 0)
        for i, r in enumerate(conv[1:], start=1):
            eligible += 1
            copies += any(echo_ratio(prev["reply"], r["reply"]) >= ECHO_THRESHOLD for prev in conv[:i])
    return {
        "n_conversations": len(convs), "n_replies": len(rows),
        "self_copy_rate": round(copies / eligible, 4) if eligible else None,
        "self_copy_eligible": eligible,
        "truncated_conversation_rate": (round(sum(any(r.get("context_truncated") for r in c)
                                                  for c in convs.values()) / len(convs), 4)
                                        if convs else None),
        "mean_context_tokens_last_turn": round(statistics.mean(last_ctx), 1) if last_ctx else None,
        "history_gain_turns_2_3": mean_history_gain(rows, (1, 2)),
        "history_gain_turns_2_6": mean_history_gain(rows, (1, 2, 3, 4, 5)),
        "summary": summarize(rows),
    }


# ---------------------------------------------------------------------------------------------
# Générateurs : nouveau format (ChatBot) ou ancien prompt de la démo


def baseline_reply(bot, messages, seed, max_new_tokens=128):
    """Reproduit demo/server.py::chat (état du 11/11) : même prompt, même tirage, même nettoyage."""
    from demo.server import clean_chat_reply, conversation_prompt
    t0 = time.monotonic()
    prompt = conversation_prompt(messages)
    all_ids = bot.tok.encode(prompt)
    ids = all_ids[-bot.context_limit:]
    with bot.lock, torch.inference_mode():
        torch.manual_seed(seed)
        out = bot.model.generate(torch.tensor([ids], device=bot.device), max_new_tokens,
                                 temperature=0.8, top_k=40, eot_id=bot.tok.eot_id)[0].tolist()
    new = out[len(ids):]
    finish = "eot" if new and new[-1] == bot.tok.eot_id else "length"
    visible = new[:-1] if finish == "eot" else new
    raw = bot.tok.decode(visible)
    # Le modèle a écrit un nouveau rôle (« Utilisateur : ») : clean_chat_reply coupe là. C'est un
    # arrêt naturel, comme finish="speaker" du nouveau format (le « Assistant : » de tête ne compte pas).
    if finish != "eot" and CHAT_ROLE.search(re.sub(r"^\s*Assistant\s*:\s*", "", raw, count=1, flags=re.I)):
        finish = "speaker"
    return {"reply": clean_chat_reply(raw), "raw_reply": raw, "seed": seed,
            "tokens_generated": len(new), "seconds": round(time.monotonic() - t0, 3),
            "context_tokens": len(ids), "context_truncated": len(all_ids) > len(ids),
            "finish": finish, "score": None, "persona": None, "interlocutor": None}


def history_gain(bot, history, reply, persona):
    """log P(réponse | tout l'historique) − log P(réponse | dernier message seul), par token.

    `history` finit par le message utilisateur auquel `reply` répond. Même mise en forme que le
    chat (ChatBot.reply_logprob). Renvoie (écart moyen par token, nombre de tokens notés).
    """
    full, n, _ = bot.reply_logprob(history, reply, persona=persona)
    last, _, _ = bot.reply_logprob(history[-1:], reply, persona=persona)
    return full - last, n


def play(bot, jobs, baseline, seeds, progress, label):
    """Joue des conversations (job = série, persona, thème, messages) avec des graines fixes."""
    rows = []
    total = sum(len(j[3]) for j in jobs) * len(seeds)
    for job_i, (serie, who, theme, msgs) in enumerate(jobs):
        for seed in seeds:
            history = []
            for turn, msg in enumerate(msgs):
                history.append({"role": "user", "content": msg})
                s = seed * 100 + turn  # graine fixe par tour : panel rejouable à l'identique
                r = baseline_reply(bot, history, s) if baseline else bot.reply(history, persona=who, seed=s)
                gain = (None, None)
                if not baseline and turn >= 1:  # à partir du 2e tour, il y a un historique
                    gain = history_gain(bot, history, r["reply"], who)
                history.append({"role": "assistant", "content": r["reply"]})
                rows.append(dict(r, job=job_i, serie=serie, theme=theme,
                                 persona=who if not baseline else None,
                                 turn=turn, message=msg, base_seed=seed,
                                 history_gain=None if gain[0] is None else round(gain[0], 5),
                                 history_gain_tokens=gain[1]))
                if progress:
                    print(f"\r{label} : {len(rows)}/{total} réponses", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    return rows


def panel_persona(bot, persona):
    """Persona du panel : celui demandé, sinon celui du modèle ; aucun en format tirets."""
    if not bot.personas:
        return None
    return persona or bot.config.get("default_persona") or DEFAULT_PERSONA


def run_discussions(bot, baseline=False, seeds=SEEDS, persona=None, progress=True):
    """Série « discussion » : 6 discussions de 6 tours, jouées par le persona du panel."""
    who = None if baseline else panel_persona(bot, persona)
    jobs = [("discussion", who, theme, msgs) for theme, msgs in DISCUSSIONS]
    return play(bot, jobs, baseline, seeds, progress, "discussions")


def run_panel(bot, baseline=False, seeds=SEEDS, persona=None, progress=True):
    persona = None if baseline else panel_persona(bot, persona)
    # Chaque ligne porte sa SÉRIE : « panel » (les 40 messages, comparables à la baseline) ou
    # « bonus_harpagon ». Les groupes filtrent sur la série, pas sur le nom du persona : avec
    # --persona HARPAGON, le panel principal est joué par HARPAGON et doit rester dans le groupe commun.
    jobs = [("panel", persona, theme, [msg]) for theme, msg in SINGLES]
    jobs += [("panel", persona, theme, msgs) for theme, msgs in CONVERSATIONS]
    # L'ancien prompt n'a pas de persona, le format tirets non plus (pas de HARPAGON à jouer), et
    # si le panel est déjà joué par HARPAGON la série bonus le rejouerait à l'identique : pas de
    # série bonus dans ces trois cas.
    if not baseline and bot.personas and persona != "HARPAGON":
        jobs += [("bonus_harpagon", "HARPAGON", theme, [msg]) for theme, msg in SINGLES[:HARPAGON_MESSAGES]]
    return play(bot, jobs, baseline, seeds, progress, "panel")


def write_transcript(path, rows, title, no_persona="ancien prompt de la démo"):
    lines = [f"# {title}", "",
             "Chaque bloc : message de l'utilisateur, puis la réponse pour chaque graine.", ""]
    by_job = {}
    for r in rows:
        by_job.setdefault(r["job"], []).append(r)
    for job, rs in by_job.items():
        head = rs[0]
        who = head["persona"] or no_persona
        lines.append(f"## {job + 1}. {head['theme']} — {who}")
        lines.append("")
        for seed in sorted({r["base_seed"] for r in rs}):
            if len({r["turn"] for r in rs}) > 1:
                lines.append(f"**Graine {seed}**")
                lines.append("")
            for r in (x for x in rs if x["base_seed"] == seed):
                reply = r["reply"].replace("\n", " / ") or "∅ (réponse vide)"
                tag = f"graine {r['seed']}" + (f", score {r['score']}" if r["score"] is not None else "")
                if r.get("history_gain") is not None:
                    tag += f", gain historique {r['history_gain']:+.3f}"
                lines.append(f"- **Vous** : {r['message']}")
                lines.append(f"  - **Réponse** ({tag}, fin={r['finish']}) : {reply}")
            lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m chatbot.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="dossier exporté, ou checkpoint .pt")
    p.add_argument("--tokenizer", default=None, help="obligatoire si --model est un .pt")
    p.add_argument("--out", required=True)
    p.add_argument("--val", default=None,
                   help="défaut : moliere_val (repliques) ou romans_val (tirets), voir DEFAULT_VALS")
    p.add_argument("--baseline-prompt", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=None,
                   help="au plus N lots de validation (défaut : toute la validation)")
    # Surcharges du décodage (sinon : chat_config.json du modèle). Sans effet en --baseline-prompt.
    p.add_argument("--mmi-lambda", type=float, default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--no-repeat-ngram", type=int, default=None)
    p.add_argument("--persona", default=None, help="persona du panel (défaut : default_persona)")
    p.add_argument("--format", choices=sorted(FORMATS), default=None,
                   help="avec un .pt seulement (défaut : format appris par le checkpoint) ; un "
                        "dossier exporté porte son format dans chat_config.json")
    p.add_argument("--no-discussion", action="store_true",
                   help="sauter la série de 6 discussions de 6 tours")
    args = p.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ("mmi_lambda", "candidates", "temperature", "top_p",
                                                "no_repeat_ngram") if getattr(args, k) is not None}

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if model_path.is_dir():
        bot = ChatBot.load(model_path, device=args.device)
        ckpt_path, tok_path = model_path / "ckpt.pt", model_path / "tokenizer.json"
        if args.format and args.format != bot.config["format"]:
            sys.exit(f"ERREUR : ce modèle est au format {bot.config['format']} (chat_config.json)")
    else:
        if not args.tokenizer:
            sys.exit("ERREUR : --tokenizer est obligatoire avec un checkpoint .pt")
        # Réglages par défaut ; format : --format, sinon celui appris par le checkpoint.
        bot = ChatBot.from_checkpoint(model_path, args.tokenizer,
                                      dict(DEFAULT_CHAT_CONFIG, format=args.format), args.device)
        ckpt_path, tok_path = model_path, Path(args.tokenizer)
    fmt_name = bot.config["format"]
    if args.persona is not None:
        if not bot.personas:
            sys.exit(f"ERREUR : le format {fmt_name} n'a pas de personnage : retirer --persona")
        overrides["default_persona"] = persona_info(args.persona)["name"]  # nom inconnu : refusé
    if args.baseline_prompt and fmt_name != "repliques":
        sys.exit("ERREUR : --baseline-prompt rejoue l'ancien chat, sans objet en format tirets")
    bot.config.update(overrides)
    val_path = args.val or DEFAULT_VALS[fmt_name]

    # 1. BPB du tour suivant sur la validation, rendue au format du modèle (sauté proprement si le
    #    fichier n'existe pas encore)
    if Path(val_path).exists():
        from chatbot.finetune import evaluate_windows, load_val_windows, token_byte_lengths
        windows = load_val_windows(val_path, bot.tok, args.seq_len,
                                   None if args.val_batches is None else args.val_batches * args.batch_size,
                                   fmt_name=fmt_name)
        val = evaluate_windows(bot.model, windows, token_byte_lengths(bot.tok), args.device,
                               args.batch_size)
        val.update({"path": val_path, "sha256": file_sha256(val_path), "windows": len(windows),
                    "format": fmt_name})
        print(f"BPB tour suivant ({val_path}) : {val['val_bpb']:.4f}", file=sys.stderr)
    else:
        val = {"skipped": f"{val_path} absent (données pas encore produites)"}
        print(f"BPB sauté : {val['skipped']}", file=sys.stderr)

    # 2. Panel fixe
    rows = run_panel(bot, baseline=args.baseline_prompt)
    # 3. Série « discussion » (6 discussions de 6 tours), à part : les groupes du panel et les clés
    #    n_replies / n_messages_played gardent leur sens et restent comparables aux anciens rapports.
    disc_rows = [] if args.no_discussion else run_discussions(bot, baseline=args.baseline_prompt)
    groups = {"tout": rows,
              # Même population que la baseline (qui n'a pas de série HARPAGON) : avant/après ici.
              # (nom gardé pour comparer aux anciens metrics.json : c'est la série « panel »)
              "commun_sans_harpagon": [r for r in rows if r["serie"] == "panel"],
              "persona_defaut_messages_isoles": [r for r in rows if r["theme"] != "conversation"
                                                 and r["serie"] == "panel"],
              "conversations_3_tours": [r for r in rows if r["theme"] == "conversation"],
              "harpagon": [r for r in rows if r["serie"] == "bonus_harpagon"]}
    metrics = {
        "mode": ("ancien prompt de la démo (baseline)" if args.baseline_prompt
                 else {"repliques": "format répliques", "tirets": "format tirets"}[fmt_name]),
        "format": fmt_name,
        "model": str(model_path), "ckpt_sha256": file_sha256(ckpt_path),
        "tokenizer_sha256": file_sha256(tok_path), "device": args.device,
        "generation": ({"temperature": 0.8, "top_k": 40, "max_new_tokens": 128, "candidates": 1,
                        "prompt": "Conversation en français. / Utilisateur : / Assistant :"}
                       if args.baseline_prompt else bot.config),
        "overrides": overrides,
        # Empreinte du code de mesure et de génération (panel, mots vides et regex compris) :
        # deux chiffres ne se comparent que si l'empreinte est la même.
        "code_sha256": {f: file_sha256(Path(__file__).parent / f)
                        for f in ("evaluate.py", "generate.py", "format.py", "finetune.py")},
        "seeds": SEEDS, "n_user_messages": len(SINGLES) + sum(len(c[1]) for c in CONVERSATIONS),
        # 40 en baseline ou avec --persona HARPAGON, 50 sinon (série bonus HARPAGON)
        "n_messages_played": len({(r["job"], r["turn"]) for r in rows}),
        "n_replies": len(rows),
        "val_next_turn": val,
        "panel": {k: summarize(v) for k, v in groups.items() if v},
        "discussion_6_tours": summarize_discussion(disc_rows) if disc_rows else None,
        "seconds_total": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "definitions": {
            "empty_rate": "part des réponses vides",
            "mean_chars": "longueur moyenne en caractères",
            "clean_end_rate": "fin propre APRÈS post-traitement : part des réponses finissant par "
                              ". ! ? … (guillemet fermant toléré) ; le nouveau format recoupe à la "
                              "dernière phrase si la limite est atteinte, pas l'ancien prompt",
            "natural_stop_rate": "arrêt naturel, sans retouche : part des réponses avec finish eot "
                                 "ou speaker (le modèle a fini sa réplique / passé la parole)",
            "repetition_rate": "moyenne, par réponse, de la part de 3-grammes de mots répétés "
                               "(dépend de la longueur : voir repetition_rate_15w)",
            "repetition_rate_15w": "repetition_rate sur les 15 premiers mots de chaque réponse",
            "distinct_2": "bigrammes de mots distincts / total, sur l'ensemble des réponses",
            "distinct_2_15w": "distinct_2 sur les 15 premiers mots de chaque réponse",
            "mean_words": "longueur moyenne en mots",
            "role_name_rate": "rôle encore visible APRÈS nettoyage : ligne de nom de rôle ou "
                              "« Utilisateur : » dans la réponse affichée",
            "role_leak_raw_rate": "rôle inventé par le modèle AVANT nettoyage (texte brut) : ligne "
                                  "de nom de rôle ou « Rôle : », SAUF le passage normal de la parole "
                                  "(à l'interlocuteur attendu dans le nouveau format, à "
                                  "« Utilisateur : » dans l'ancien prompt), qui est un arrêt "
                                  "naturel (natural_stop_rate) ; le persona qui reprend la parole "
                                  "compte comme rôle inventé",
            "overlap_rate": "part des réponses partageant un mot plein (>= 5 lettres, hors mots vides "
                            "et mots d'adresse) avec le message, parmi les messages qui en ont un "
                            "(overlap_eligible) ; signal grossier, gonflé par les échos",
            "overlap_non_echo_rate": "reprise d'un mot plein SANS recopie du message (chiffre "
                                     "principal de la reprise du sujet), même dénominateur",
            "echo_rate": "défaut : part des réponses qui recopient >= 60 % des mots du message",
            "chosen_vs_candidates_length": "longueur de la réponse retenue / longueur moyenne des "
                                           "candidates non vides (< 1 : le reclassement préfère "
                                           "les réponses courtes)",
            "mean_mmi_score": "score MMI de la réponse retenue, moyenné sur les réponses (nouveau "
                              "format seulement). Si generation.mmi_total est vrai (défaut depuis "
                              "le 13/11) : SOMME sur les tokens de la réponse + fin de réplique "
                              "(« \\n\\n ») de log P(r|conversation) − λ·log P(r|persona seul) ; "
                              "sinon (anciens rapports, clé absente) : moyenne par token. Ne "
                              "comparer qu'entre rapports ayant les mêmes generation.mmi_total, "
                              "generation.mmi_lambda et code_sha256 (ex. même modèle, λ=1 : 0,7291 "
                              "en moyenne contre 6,5763 en somme)",
            "history_gain_turns_2_3": "gain de l'historique : sur les tours 2 et 3 des conversations, "
                                      "somme sur les tokens de la réponse retenue (+ fin de tour) "
                                      "de [log P(réponse | tout l'historique) − log P(réponse | "
                                      "dernier message seul)], divisée par le nombre total de "
                                      "tokens notés (moyenne par token, en nats). Positif : le "
                                      "modèle se sert vraiment des tours précédents ; ~0 : il ne "
                                      "lit que le dernier message. Même mise en forme et même "
                                      "coupe du contexte que le chat ; None en baseline",
            "history_gain_turns_2_6": "même mesure sur les tours 2 à 6 (série discussion)",
            "discussion_6_tours": "6 discussions de 6 tours en français courant (DISCUSSIONS) × 3 "
                                  "graines, jouées par le persona du panel (aucun en tirets) ; "
                                  "self_copy_rate = part des réponses (tours 2 à 6) qui recopient "
                                  ">= 60 % des mots d'une réponse précédente du bot dans la même "
                                  "conversation ; truncated_conversation_rate = part des "
                                  "conversations dont au moins un tour a vu son contexte tronqué "
                                  "(début de conversation perdu) ; mean_context_tokens_last_turn "
                                  "= tokens du contexte au 6e tour ; summary = mêmes mesures que "
                                  "le panel (longueur : mean_words, mean_chars). Hors groupes du "
                                  "panel et hors n_replies",
            "groupes": "commun_sans_harpagon = série « panel » (40 messages, joués par le persona "
                       "du panel), population comparable à la baseline ; harpagon = série bonus "
                       "(10 messages joués par HARPAGON, rôle d'une pièce de validation : "
                       "généralisation si le modèle ne l'a pas vue), absente en baseline et avec "
                       "--persona HARPAGON",
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    no_persona = "ancien prompt de la démo" if args.baseline_prompt else "sans personnage (tirets)"
    write_transcript(out / "transcript.md", rows, f"Panel du chatbot — {metrics['mode']}", no_persona)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
    if disc_rows:
        write_transcript(out / "transcript_discussion.md", disc_rows,
                         f"Discussions de 6 tours — {metrics['mode']}", no_persona)
        (out / "rows_discussion.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in disc_rows), encoding="utf-8")
    print(json.dumps(metrics["panel"].get("commun_sans_harpagon", metrics["panel"]["tout"]),
                     indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    from chatbot.accel import setup_console
    setup_console()  # Windows : sortie redirigée en UTF-8 (sinon « ≈ » ferait planter print)
    main()
