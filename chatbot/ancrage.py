"""Choix de la réponse « ancré sur le dernier message » — option de décodage, sans réentraînement.

    from chatbot.generate import ChatBot
    from chatbot.ancrage import installer
    bot = installer(ChatBot.load("runs/chatbot_rtx/modeles/chatbot_moliere"), mu=1.0)
    bot.reply(messages, candidates=32)

Pourquoi. ChatBot.mmi_scores choisit, parmi les candidates, celle qui dépend le plus de la
conversation :
    S = somme sur les tokens de [ log P(réponse | conversation) - lambda * log P(réponse | persona seul) ]
Un mot que le bot a déjà dit deux tours plus tôt dépend lui aussi « de la conversation » : il est
récompensé, et le bot se met à tourner autour des mêmes mots.

Ce module retire ce qui était prévisible SANS lire le dernier message :
    S' = S - mu * somme de [ log P(réponse | conversation sans le dernier message) - log P(réponse | persona seul) ]
« sans le dernier message » = même historique, dernier message remplacé par « … » (la réplique vide
du format). Avec lambda = mu = 1 : S' = log P(réponse | conversation) - log P(réponse | conversation
sans le dernier message), c'est-à-dire ce que le dernier message, et lui seul, apporte.
mu = 0 rend exactement le choix d'origine (les méthodes d'origine sont appelées telles quelles).

Mesure (14/11/2026, modèle Molière M, 144 réponses appariées, 3 juges IA à l'aveugle, règle fixée
avant) : mu = 1 avec 32 candidates contre le décodage livré (16 candidates) : rapport au message
96 -> 114, français 125 -> 120, style 137 -> 133 ; aux tours 2 et suivants, rapport 29 -> 43 sur 54.
mu = 1 avec 16 candidates : +2 sur 72 seulement ; 32 candidates sans ancrage : -3 sur 72.
Coût : une passe de plus sur les candidates et deux fois plus de candidates (0,81 s contre 0,47 s
par réponse sur RTX 5090).
"""
import types

from chatbot.format import token_header

NEUTRE = "…"  # réplique vide du format (voir format.render_messages)


def installer(bot, mu=1.0):
    """Branche l'ancrage sur CE ChatBot (les autres ne changent pas). Renvoie le bot."""
    if not hasattr(bot, "_ancrage_origine"):
        bot._ancrage_origine = (bot.build_context, bot.mmi_scores)
        bot.build_context = types.MethodType(_build_context, bot)
        bot.mmi_scores = types.MethodType(_mmi_scores, bot)
    return regler(bot, mu)


def regler(bot, mu):
    """Change mu sans recharger le modèle ; 0 = choix d'origine."""
    mu = float(mu)
    if not 0.0 <= mu <= 2.0:
        raise ValueError("mu hors bornes (0 à 2)")
    bot._ancrage_mu = mu
    return bot


def contexte_neutre(bot, messages, persona, interlocutor, used):
    """Ids du contexte où le dernier message est remplacé par « … », avec la même coupe des tours
    anciens que le vrai contexte (`used` = nombre de messages gardés par build_context)."""
    gardes = messages[max(0, len(messages) - used):]
    # La coupe peut commencer sur une réponse du bot : on la garde, comme le vrai contexte.
    neutre = gardes[:-1] + [{"role": "user", "content": NEUTRE}]
    tours = bot.fmt["render_messages"](neutre, persona, interlocutor)
    return [bot.tok.eot_id] + bot.tok.encode("".join(tours) + token_header(bot.fmt, persona))


def _build_context(self, messages, persona, interlocutor, budget):
    ids, used, truncated = self._ancrage_origine[0](messages, persona, interlocutor, budget)
    # reply() appelle build_context puis mmi_scores sous le même verrou : le contexte neutre mémorisé
    # ici est bien celui de la réponse en cours.
    self._ancrage_neutre = contexte_neutre(self, messages, persona, interlocutor, used)
    return ids, used, truncated


def _mmi_scores(self, context_ids, persona, replies, lam):
    mu = self._ancrage_mu
    if not mu:
        return self._ancrage_origine[1](context_ids, persona, replies, lam)
    lead = self.fmt["reply_lead"]
    replies_ids = [self.tok.encode(lead + r + "\n\n") for r in replies]
    avec = self.mean_logprob(context_ids, replies_ids)
    seul = self.mean_logprob([self.tok.eot_id] + self.tok.encode(token_header(self.fmt, persona)), replies_ids)
    sans = self.mean_logprob(self._ancrage_neutre, replies_ids)
    scores = [a - lam * b - mu * (h - b) for a, b, h in zip(avec, seul, sans)]
    if self.config.get("mmi_total", True):
        scores = [s * len(r) for s, r in zip(scores, replies_ids)]
    return scores
