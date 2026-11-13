"""Génération des réponses du chatbot (format « repliques » : Molière ; « tirets » : français moderne).

    uv run python -m chatbot.generate export --ckpt runs/chat/best.pt \\
        --tokenizer submissions/2026-11-11_153839/tokenizer.json --out data/models/chat_v1
    uv run python -m chatbot.generate chat --model data/models/chat_v1 --persona HARPAGON \\
        "Bonjour, comment allez-vous ?"
    uv run python -m chatbot.generate export --format tirets --ckpt runs/chatbot_francais/best.pt \\
        --tokenizer submissions/2026-11-11_153839/tokenizer.json --out data/models/chatbot_francais
    uv run python -m chatbot.generate chat --model data/models/chatbot_francais "Salut, ça va ?"

Chemin d'une réponse :
1. la conversation est mise en scène (chatbot.format) : l'utilisateur = l'interlocuteur, le modèle
   écrit la réplique du persona ;
2. si elle dépasse la fenêtre de 512 tokens, on retire les répliques les plus anciennes, entières ;
3. on tire plusieurs réponses candidates en parallèle (température + top-p, sans répéter un
   4-gramme) et on s'arrête dès qu'un nouveau personnage prend la parole ;
4. on écarte les candidates vides, celles qui écrivent un nom de rôle comme une ligne « NOM. »,
   celles qui recopient le message ou une réponse précédente du bot (écho) et celles qui commencent
   par une minuscule, puis on garde celle qui dépend le plus de ce qu'a dit l'utilisateur
   (score MMI, Maximum Mutual Information = information mutuelle maximale). Chaque filtre a un
   repli : s'il écarterait toutes les candidates, on les garde toutes.

Format « tirets » (chat_config.json : "format": "tirets") : même chemin, sans personnage. Le prompt
finit par « — » + espace, une candidate s'arrête au premier retour à la ligne (fin du paragraphe =
fin du tour), le filtre des noms de rôle n'a pas d'objet et le terme « réponse passe-partout » du
score MMI est log P(réponse | « <|endoftext|>— »).
"""

import argparse
import json
import re
import secrets
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from chatbot.format import (EOT, FORMATS, NAME_CHARS, PERSONAS, DEFAULT_FORMAT, DEFAULT_PERSONA,
                            LEADING_DASH, SPEAKER_LINE, check_messages, clean_text, format_of_version,
                            get_format, normalize_speaker, persona_info, token_header)
from ftb.model import load_checkpoint
from ftb.tokenizer import load_tokenizer

DEFAULT_CHAT_CONFIG = {
    "format": DEFAULT_FORMAT,  # "repliques" (Molière) ou "tirets" (français moderne) : chatbot.format
    "temperature": 0.8,      # < 1 : on favorise les tokens probables, moins de hasard
    "top_p": 0.9,            # on tire seulement parmi les tokens qui font 90 % de la probabilité
    "max_new_tokens": 120,   # longueur maximale d'une réplique
    "candidates": 16,        # nombre de réponses tirées puis reclassées
    # Poids de la pénalité « réponse passe-partout » : 1 = information mutuelle exacte.
    "mmi_lambda": 1.0,
    # Score TOTAL (somme sur les tokens) et non moyen : mesuré le 13/11 sur le modèle affiné, le
    # score moyen par token choisissait « Oui. » ou « Non. » presque à chaque fois (6,7 mots en
    # moyenne contre 13 pour les candidates). La somme mesure tout ce que la réponse tire du
    # message ; pour qu'elle ne favorise pas les tirades, on ne compare que des réponses de
    # min_words à max_words mots (s'il y en a).
    "mmi_total": True,
    "min_words": 4,
    "max_words": 30,
    "no_repeat_ngram": 4,    # interdit de répéter un 4-gramme de tokens déjà produit
    # Candidate écartée si elle recopie >= 60 % des mots d'un message de la conversation (dernier
    # message, anciens messages utilisateur, réponses précédentes du bot). Limite connue (revue du 13/11) : une reprise PARTIELLE sous ce seuil passe, et le score
    # total la favorise (≈ 1 fois sur 2) ; souvent c'est une vraie réponse au sujet (« Que pensez-vous
    # de l'argent ? » -> « L'argent est nécessaire… »), donc on garde 0,6 et λ = 1 par choix.
    "echo_max": 0.6,
    "filter_role_names": True,  # candidate écartée si elle écrit le nom du persona/interlocuteur
    # Candidate écartée si elle commence par une minuscule : elle continue la phrase de
    # l'interlocuteur ou c'est une indication de scène (« chaise » -> « souriant.\nIl lui donne… »).
    "filter_lowercase_start": True,
    "strip_stage_directions": True,  # « (Il sort.) » retiré de la réponse affichée
    "default_persona": DEFAULT_PERSONA,
}
TOKENIZER_CHECK = ["HARPAGON.\nAu voleur ! au voleur ! à l'assassin !\n\nCLÉANTE.\nQu'avez-vous ?\n\n",
                   "Bonjour, ça va ? « Très bien » — merci ! L'œuvre d'Ève, 3,5 %…"]

# Début d'une nouvelle réplique : ligne vide, puis une ligne en capitales terminée par un point
# (« CLÉANTE. », « MONSIEUR JOURDAIN. », « M. LOYAL. »), puis un retour à la ligne. Mêmes caractères
# de nom que SPEAKER_LINE (chatbot.format) : l'arrêt et la découpe reconnaissent les mêmes noms.
NEW_TURN = re.compile(r"\n\n[A-ZÀ-ÖØ-ÞŒ]" + NAME_CHARS + r"{0,60}\.[ \t]*\n")
# Fin de phrase, sauf le point de « M. » / « MM. » (Monsieur, Messieurs), qui est une abréviation.
SENTENCE_END = re.compile(r"(?<!\bM)(?<!\bMM)[.!?…][»\")\]]*")
WORD = re.compile(r"\w+")


# Didascalie entre parenthèses, sur une ou plusieurs lignes (le « \n » est permis à l'intérieur).
STAGE_DIRECTION = re.compile(r"[ \t]*\([^()]{0,160}\)")
# Parenthèse ouverte jamais fermée jusqu'à la fin : réplique coupée au milieu d'une didascalie.
UNCLOSED_DIRECTION = re.compile(r"[ \t]*\([^()]*$")


def strip_stage_directions(text):
    """Retire les didascalies entre parenthèses (« (Il sort.) ») : dans un chat, on ne garde que
    la parole du personnage.

    - deux passes : la 2e retire la parenthèse extérieure d'une imbrication simple
      (« (bas (très bas)) ») ;
    - une « ( » restée ouverte en fin de texte (coupe à la longueur maximale) est retirée avec
      tout ce qui la suit ;
    - une virgule, un point-virgule ou deux-points laissé orphelin en tête de ligne est retiré
      (« (Il rit.), vraiment ? » -> « vraiment ? »). On garde « — » (tiret de dialogue) et « . »
      (« ... »), et on ne force pas de capitale : on ne réécrit pas le texte du modèle.
    Les espaces et lignes laissés vides sont nettoyés.
    """
    for _ in range(2):
        text = STAGE_DIRECTION.sub("", text)
    text = UNCLOSED_DIRECTION.sub("", text)
    lines = (re.sub(r" {2,}", " ", line).strip().lstrip(",;:").strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def echo_ratio(message, reply):
    """Part des mots du message (minuscules) qu'on retrouve dans la réponse : 1.0 = écho pur.

    Message de moins de 3 mots distincts : une part de mots n'a pas de sens (reprendre « Bonjour »
    dans « Bonjour, mon ami » ferait 100 %), et reprendre un salut une fois est naturel
    (« Au revoir, ma fille. »). On ne compte alors que l'écho DÉGÉNÉRÉ (1.0) : un mot du message
    répété (« Tu mens, tu mens. ») ou une réponse faite uniquement des mots du message
    (« Tu mens. » -> « Tu mens. »). Sinon 0.0.
    """
    m_words = WORD.findall(message.lower().replace("’", "'"))
    r_words = WORD.findall(reply.lower().replace("’", "'"))
    m = set(m_words)
    if len(m) >= 3:
        return len(m & set(r_words)) / len(m)
    if not m or not r_words:  # rien à comparer (« … » n'a aucun mot) : pas un écho
        return 0.0
    counts = Counter(r_words)
    return 1.0 if any(counts[w] >= 2 for w in m) or set(r_words) <= m else 0.0


def is_fake_role_line(text, names):
    """La candidate écrit-elle un nom de rôle comme le ferait la ligne « NOM. » du format ?

    - nom en capitales n'importe où (« N° LA GRANGE. ») : test historique, gardé tel quel ;
    - réponse de plusieurs lignes dont une ligne, comparée sans tenir compte de la casse, est
      juste un nom (≤ 4 mots, finit par « . ») : « Don JUAN.\\nDon Julie! » (le modèle a commencé
      une nouvelle réplique sans les capitales que NEW_TURN et SPEAKER_LINE attendent).
    Une réponse d'UNE ligne qui est juste un nom (« Sganarelle. ») reste permise : elle peut
    répondre à « Qui est là ? ». Un nom au milieu d'une phrase (« Bonjour, Don Juan. ») aussi.
    """
    if any(name in text for name in names):
        return True
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # `line.rstrip(".").strip()` : une ligne faite seulement de points (« ... », ou « . » laissé
    # par « (Il sort.). ») n'a pas de nom ; normalize_speaker lèverait ValueError et ferait
    # échouer toute la réponse pour UNE candidate. On ne compare donc que les lignes avec un nom.
    return len(lines) > 1 and any(line.endswith(".") and len(line.split()) <= 4
                                  and line.rstrip(".").strip()
                                  and normalize_speaker(line) in names for line in lines)


# ---------------------------------------------------------------------------------------------
# Passe avant avec cache des clés/valeurs (KV cache)


def forward_cached(model, idx, past=None):
    """Passe avant du GPT du kit qui garde les clés/valeurs déjà calculées.

    Sans cache, chaque nouveau token relance le modèle sur tout le contexte (~500 tokens) : c'est
    ~500 fois trop de calcul. Avec le cache, on ne calcule que le nouveau token ; ses requêtes
    regardent les clés/valeurs gardées. Mêmes modules et mêmes poids que ftb.model.GPT.forward
    (vérifié par un test). Deux cas seulement : tout le prompt d'un coup (past=None) ou un token.
    """
    B, T = idx.shape
    t0 = 0 if past is None else past[0][0].size(2)
    assert past is None or T == 1, "avec un cache, on avance d'un token à la fois"
    assert t0 + T <= model.context_length, "le contexte du modèle est dépassé"
    x = model.wte(idx) + model.wpe(torch.arange(t0, t0 + T, device=idx.device))
    new_past = []
    for i, block in enumerate(model.blocks):
        attn = block.attn
        C, H = x.size(-1), attn.n_head
        q, k, v = attn.qkv(block.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[i][0], k], dim=2)
            v = torch.cat([past[i][1], v], dim=2)
        # Prompt : masque causal ; un seul nouveau token : il a le droit de tout regarder.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        x = x + attn.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + block.mlp(block.ln2(x))
        new_past.append((k, v))
    return model.lm_head(model.ln_f(x)), new_past


# ---------------------------------------------------------------------------------------------
# Choix du token suivant


def top_p_filter(logits, top_p):
    """Nucleus (top-p) : on garde les tokens les plus probables jusqu'à cumuler top_p.

    Avec top-k (k fixe) on garde toujours k tokens, même quand le modèle hésite entre 2 ou entre
    500. Top-p s'adapte : peu de choix quand le modèle est sûr, plus quand il hésite. On coupe la
    longue traîne de tokens improbables qui produit le charabia.
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    # Masse cumulée AVANT chaque token : on retire un token si les meilleurs suffisaient déjà.
    # Le premier token (masse avant = 0) est donc toujours gardé.
    remove = probs.cumsum(dim=-1) - probs > top_p
    sorted_logits[remove] = -float("inf")
    return torch.full_like(logits, -float("inf")).scatter(-1, order, sorted_logits)


def banned_tokens(generated, n):
    """Tokens interdits pour ne pas répéter un n-gramme déjà écrit dans la réponse.

    Si les n-1 derniers tokens ont déjà été suivis de X plus haut, écrire X recopierait un
    n-gramme : c'est la boucle « je vous dis que je vous dis que je vous dis… » des petits modèles.
    """
    if n <= 1 or len(generated) < n - 1:
        return []
    prefix = tuple(generated[len(generated) - n + 1:])
    return [generated[i + n - 1] for i in range(len(generated) - n + 1)
            if tuple(generated[i:i + n - 1]) == prefix]


# ---------------------------------------------------------------------------------------------
# Post-traitement


def cut_reply(text, finish):
    """Ne garder que la réplique du persona, proprement terminée."""
    text = text.split(EOT, 1)[0]
    text = text.split("\n\n", 1)[0]  # une réplique ne contient jamais de ligne vide
    # Un nom de rôle seul sur sa ligne (« VALÈRE. ») : au début, le modèle a répété un nom (on
    # l'enlève) ; plus loin, il a enchaîné sur un autre personnage sans ligne vide (on coupe avant).
    lines = text.split("\n")
    while lines and SPEAKER_LINE.fullmatch(lines[0].strip()):
        lines.pop(0)
    for i in range(1, len(lines)):
        if SPEAKER_LINE.fullmatch(lines[i].strip()):
            lines = lines[:i]
            break
    text = "\n".join(lines)
    if finish == "length":
        ends = [m.end() for m in SENTENCE_END.finditer(text) if m.end() >= 20]
        if ends:
            text = text[:ends[-1]]
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


# Format tirets : un autre interlocuteur peut prendre la parole DANS le paragraphe, à la manière
# des vieux romans (« Oui.--Non. », « Oui. — Non. ») : un tiret juste après une fin de phrase.
SECOND_DASH_TURN = re.compile(r"(?<=[.!?…»])\s*(?:[—–―‒]|--)")


def cut_dash_reply(text, finish):
    """Format tirets : ne garder que le tour du bot, proprement terminé.

    - fin du tour = premier retour à la ligne (un tour est un seul paragraphe) ou <|endoftext|> ;
    - un second tour dans le même paragraphe (tiret après une fin de phrase) est coupé ;
    - la réponse ne commence jamais par un tiret (le « — » est dans le prompt, pas dans la bulle) ;
    - coupée à la longueur maximale : on recoupe à la dernière fin de phrase, comme cut_reply.
    """
    text = text.split(EOT, 1)[0].split("\n", 1)[0]
    text = SECOND_DASH_TURN.split(text, 1)[0]
    text = LEADING_DASH.sub("", text.strip()).strip()
    if finish == "length":
        ends = [m.end() for m in SENTENCE_END.finditer(text) if m.end() >= 20]
        if ends:
            text = text[:ends[-1]]
    return text.strip()


# Ce qui dépend du format côté génération : fin du tour pendant le tirage, découpe de la réponse.
# Repliques : un nouveau « NOM. » après une ligne vide. Tirets : un retour à la ligne (fin du
# paragraphe). Dans les deux cas l'arrêt s'appelle « speaker » : l'autre interlocuteur va parler.
TURN_OVER = {"repliques": lambda tail: bool(NEW_TURN.search(tail)),
             "tirets": lambda tail: "\n" in tail}
CUT_REPLY = {"repliques": cut_reply, "tirets": cut_dash_reply}


def checkpoint_format(meta):
    """Format appris par un checkpoint AFFINÉ (méta « format »), sinon None.

    Un checkpoint de pré-entraînement porte aussi « format » (celui de sa mesure de validation),
    mais il n'a appris aucun format de chat : on ne le compte pas.
    """
    return format_of_version(meta.get("format")) if meta.get("finetune") else None


class ChatBot:
    """Modèle + tokenizer + réglages par défaut. Une réponse à la fois (verrou)."""

    def __init__(self, model, tok, config=None, device="cpu"):
        self.model = model.to(device).eval()
        self.tok = tok
        self.device = device
        self.config = dict(DEFAULT_CHAT_CONFIG, **(config or {}))
        # Ancien chat_config.json sans clé "format" (ou "format": null) : répliques, comme avant.
        self.config["format"] = self.config.get("format") or DEFAULT_FORMAT
        self.fmt = get_format(self.config["format"])  # format inconnu : ValueError
        self.turn_over = TURN_OVER[self.config["format"]]
        self.cut = CUT_REPLY[self.config["format"]]
        self.context_limit = model.context_length
        self.lock = threading.Lock()

    @classmethod
    def load(cls, model_dir, device="cpu"):
        d = Path(model_dir)
        config = json.loads((d / "chat_config.json").read_text(encoding="utf-8"))
        return cls.from_checkpoint(d / "ckpt.pt", d / "tokenizer.json", config, device)

    @classmethod
    def from_checkpoint(cls, ckpt_path, tokenizer_path, config=None, device="cpu"):
        model, ckpt = load_checkpoint(ckpt_path)
        tok = load_tokenizer(tokenizer_path, check_texts=TOKENIZER_CHECK)
        meta = ckpt.get("meta", {})
        sha = meta.get("tokenizer_sha256")
        if sha and sha != tok.sha256:
            raise ValueError("ce tokenizer n'est pas celui du checkpoint")
        # Format : celui du chat_config s'il est donné, sinon celui appris par le checkpoint.
        # Les deux doivent concorder : un modèle affiné en tirets ne parle pas en répliques.
        config = dict(config or {})
        trained = checkpoint_format(meta)
        config["format"] = config.get("format") or trained or DEFAULT_FORMAT
        if trained and config["format"] != trained:
            raise ValueError(f"chat_config demande le format {config['format']!r}, mais ce "
                             f"checkpoint a été affiné au format {trained!r}")
        return cls(model, tok, config, device)

    @property
    def personas(self):
        """Personnages proposés au chat ; aucun en format tirets (deux interlocuteurs sans nom)."""
        return [dict(p) for p in PERSONAS] if self.fmt["personas"] else []

    def speakers(self, persona=None, interlocutor=None):
        """(persona, interlocuteur) validés. Format tirets : ("", ""), un nom donné est refusé."""
        if not self.fmt["personas"]:
            if persona or interlocutor:
                raise ValueError("le format tirets n'a pas de personnage : persona=None")
            return "", ""
        info = persona_info(persona or self.config["default_persona"])
        return info["name"], normalize_speaker(interlocutor or info["interlocutor"])

    # -- contexte ------------------------------------------------------------------------------

    def build_context(self, messages, persona, interlocutor, budget):
        """Ids du contexte <|endoftext|> + prompt, en au plus `budget` tokens.

        On retire les répliques les plus anciennes, entières : le modèle voit toujours une scène
        bien formée et la fin de conversation la plus récente. Si le dernier message seul est trop
        long, on garde sa fin (c'est elle qui précède la réponse).
        Renvoie (ids, nombre de messages gardés, tronqué ?).
        """
        render = self.fmt["render_messages"]
        turns = render(messages, persona, interlocutor)
        header = token_header(self.fmt, persona)  # repliques : « PERSONA.\n » ; tirets : « — »
        for first in range(len(turns)):
            ids = [self.tok.eot_id] + self.tok.encode("".join(turns[first:]) + header)
            if len(ids) <= budget:
                return ids, len(turns) - first, first > 0
        # Le dernier message user seul dépasse : on garde ses derniers tokens.
        text = clean_text(messages[-1]["content"])
        fixed = len(self.tok.encode(self.fmt["user_frame"](interlocutor) + header)) + 1
        keep = max(1, budget - fixed)
        while True:
            tail = self.tok.decode(self.tok.encode(text)[-keep:]).lstrip("�")
            last = render([{"role": "user", "content": tail or "…"}], persona, interlocutor)
            ids = [self.tok.eot_id] + self.tok.encode(last[0] + header)
            if len(ids) <= budget or keep == 1:
                return ids[-budget:], 1, True
            keep -= max(1, len(ids) - budget)

    # -- échantillonnage -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, context_ids, n, max_new_tokens, temperature, top_p, no_repeat, seed):
        """Tire n suites en parallèle. Renvoie [(ids générés, raison d'arrêt)]."""
        gen = torch.Generator().manual_seed(seed)  # tirage sur CPU : même graine -> même suite
        idx = torch.tensor([context_ids], device=self.device)
        logits, past = forward_cached(self.model, idx)
        # Le prompt est calculé une fois, puis recopié pour les n candidates.
        past = [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in past]
        logits = logits[:, -1, :].expand(n, -1)
        out = [[] for _ in range(n)]
        finish = [None] * n
        # Le prompt finit par « PERSONA.\n » : ce « \n » appartient au prompt, pas à out[r]. Si la
        # réponse commence par « \nCLÉANTE.\n » (réplique vide), le « \n\n » est à cheval : on remet
        # ce « \n » devant le texte testé tant que la réponse est courte.
        lead = "\n" if self.tok.decode(context_ids[-1:]).endswith("\n") else ""
        for step_i in range(max_new_tokens):
            logits = logits.float().cpu() / max(temperature, 1e-5)
            for r in range(n):
                ban = banned_tokens(out[r], no_repeat)
                if ban:
                    logits[r, ban] = -float("inf")
            probs = torch.softmax(top_p_filter(logits, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)[:, 0].tolist()
            for r in range(n):
                if finish[r] is None:
                    out[r].append(nxt[r])
                    if nxt[r] == self.tok.eot_id:
                        finish[r] = "eot"
                    elif self.turn_over((lead if len(out[r]) <= 40 else "")
                                        + self.tok.decode(out[r][-40:])):
                        finish[r] = "speaker"  # fin du tour : nouveau « NOM. » ou fin du paragraphe
            if all(f is not None for f in finish) or step_i == max_new_tokens - 1:
                break
            # Les suites terminées continuent d'avancer (calcul en lot) mais on ignore leurs tokens.
            step = torch.tensor(nxt, device=self.device).view(n, 1)
            logits, past = forward_cached(self.model, step, past)
            logits = logits[:, -1, :]
        return [(ids, f or "length") for ids, f in zip(out, finish)]

    @torch.no_grad()
    def mean_logprob(self, prefix_ids, replies_ids):
        """Log-probabilité moyenne par token de chaque réponse après un même préfixe (un lot)."""
        seqs = [(prefix_ids + r)[-self.context_limit:] for r in replies_ids]
        L = max(len(s) for s in seqs)
        x = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        logp = torch.log_softmax(self.model(x.to(self.device))[0].float(), dim=-1).cpu()
        scores = []
        for i, (s, r) in enumerate(zip(seqs, replies_ids)):
            pos = torch.arange(len(s) - len(r) - 1, len(s) - 1)  # la position t prédit le token t+1
            scores.append(logp[i, pos, torch.tensor(r)].mean().item())
        return scores

    def mmi_scores(self, context_ids, persona, replies, lam):
        """Score MMI (Maximum Mutual Information, information mutuelle maximale) =
        moyenne par token de log P(réponse | conversation)
        − λ · moyenne par token de log P(réponse | persona seul).

        Format tirets : « persona seul » devient « <|endoftext|>— », un début de dialogue vide.
        Le premier terme récompense une réponse plausible APRÈS ce qu'a dit l'utilisateur. Le second
        retire ce que la réponse a de probable de toute façon (« Ah ! mon Dieu ! » marche partout).
        Ce qui reste mesure combien la réponse dépend du message : on préfère celle qui y répond.
        Avec « mmi_total » (défaut), ce score moyen est multiplié par le nombre de tokens : c'est
        l'information totale que la réponse tire du message. Le score moyen seul favorisait les
        réponses d'un mot (« Oui. ») ; la somme favoriserait les tirades, d'où la fenêtre
        min_words–max_words de reply(). Le champ « score » renvoyé par reply() suit la même
        convention (somme si mmi_total, moyenne sinon). Limite connue : recopier le message (ou
        une réponse précédente du bot) en dépend au maximum, d'où le filtre anti-écho de reply().
        On note aussi la ligne vide finale : une réplique coupée au milieu est pénalisée.
        """
        # reply_lead : en tirets, l'espace qui suit « — » appartient au premier token de la réponse.
        lead = self.fmt["reply_lead"]
        replies_ids = [self.tok.encode(lead + r + "\n\n") for r in replies]
        with_ctx = self.mean_logprob(context_ids, replies_ids)
        alone = self.mean_logprob([self.tok.eot_id] + self.tok.encode(token_header(self.fmt, persona)),
                                  replies_ids)
        scores = [a - lam * b for a, b in zip(with_ctx, alone)]
        if self.config.get("mmi_total", True):
            # Somme sur les tokens = moyenne × nombre de tokens : information totale tirée du message.
            scores = [s * len(r) for s, r in zip(scores, replies_ids)]
        return scores

    def reply_logprob(self, messages, reply, persona=None, interlocutor=None):
        """Log-probabilité moyenne par token d'une réponse donnée après la conversation `messages`
        (mise en forme exactement comme au chat, même coupe des tours anciens).

        Sert à la mesure « gain de l'historique » (chatbot.evaluate) : on compare cette valeur
        avec tout l'historique et avec le dernier message seul.
        Renvoie (moyenne par token, nombre de tokens notés, contexte tronqué ?).
        """
        check_messages(messages)
        persona, interlocutor = self.speakers(persona, interlocutor)
        reply_ids = self.tok.encode(self.fmt["reply_lead"] + reply + "\n\n")
        ctx, _, truncated = self.build_context(messages, persona, interlocutor,
                                               max(1, self.context_limit - len(reply_ids)))
        return self.mean_logprob(ctx, [reply_ids])[0], len(reply_ids), truncated

    # -- réponse -------------------------------------------------------------------------------

    def reply(self, messages, persona=None, seed=None, temperature=None, top_p=None,
              max_new_tokens=None, candidates=None, interlocutor=None, mmi_lambda=None):
        cfg = self.config
        temperature = cfg["temperature"] if temperature is None else float(temperature)
        top_p = cfg["top_p"] if top_p is None else float(top_p)
        max_new_tokens = cfg["max_new_tokens"] if max_new_tokens is None else int(max_new_tokens)
        candidates = cfg["candidates"] if candidates is None else int(candidates)
        lam = cfg["mmi_lambda"] if mmi_lambda is None else float(mmi_lambda)
        if not 1 <= max_new_tokens < self.context_limit // 2 or not 1 <= candidates <= 32:
            raise ValueError("max_new_tokens ou candidates hors bornes")
        if not 0 < temperature <= 2 or not 0 < top_p <= 1:
            raise ValueError("temperature ou top_p hors bornes")
        check_messages(messages)
        persona_name, interlocutor = self.speakers(persona, interlocutor)  # tirets : ("", "")
        if seed is None:
            seed = secrets.randbelow(2**32)  # graine neuve, renvoyée pour pouvoir rejouer
        t0 = time.monotonic()
        no_repeat = cfg.get("no_repeat_ngram", 4)
        with self.lock:
            ctx, used, truncated = self.build_context(messages, persona_name, interlocutor,
                                                      self.context_limit - max_new_tokens)

            def draw(s):
                samples = self.sample(ctx, candidates, max_new_tokens, temperature, top_p, no_repeat, s)
                clean = strip_stage_directions if cfg.get("strip_stage_directions", True) else str
                texts = [(clean(self.cut(self.tok.decode(ids), f)), ids, f) for ids, f in samples]
                return samples, [t for t in texts if t[0]]  # réponse vide : candidate écartée

            samples, kept = draw(seed)
            retried = False
            if not kept:  # toutes vides : un seul nouveau tirage (graine suivante) avant d'abandonner
                samples, kept = draw((seed + 1) % 2**32)
                retried = True
            pool = kept
            # Sans personnage (tirets), il n'y a pas de nom de rôle à filtrer.
            if pool and self.fmt["personas"] and cfg.get("filter_role_names", True):
                # Une réplique n'écrit pas un nom de rôle comme une ligne « NOM. » : MMI favoriserait
                # cette recopie du contexte. Si toutes les candidates sont concernées, on les garde.
                names = {persona_name, interlocutor}
                pool = [t for t in pool if not is_fake_role_line(t[0], names)] or pool
            echo_max = cfg.get("echo_max", 0.6)
            if pool and echo_max is not None:
                # Recopier un texte du contexte dépend au maximum du contexte : MMI le récompense.
                # On interdit donc la copie (>= echo_max des mots) de TOUT message de la
                # conversation : le dernier message, les messages utilisateur plus anciens (sinon
                # il repose une question d'il y a deux tours) et chaque réponse déjà donnée par le
                # bot (sinon il tourne en boucle), sauf s'il n'y a pas d'autre choix.
                sources = [m["content"] for m in messages]  # rôles déjà validés : user/assistant
                pool = [t for t in pool
                        if all(echo_ratio(s, t[0]) < echo_max for s in sources)] or pool
            if pool and cfg.get("filter_lowercase_start", True):
                # Une réplique qui commence par une minuscule continue la phrase de l'interlocuteur
                # ou décrit la scène (« souriant.\nIl lui donne… ») : MMI la favorise, on l'écarte.
                pool = [t for t in pool if not t[0][:1].islower()] or pool
            if pool and cfg.get("mmi_total", True):
                # Ni un mot seul ni une tirade : on compare des réponses de longueur raisonnable.
                lo, hi = cfg.get("min_words", 4), cfg.get("max_words", 30)
                pool = [t for t in pool if lo <= len(WORD.findall(t[0])) <= hi] or pool
            if pool:
                scores = self.mmi_scores(ctx, persona_name, [t[0] for t in pool], lam)
                best = max(range(len(pool)), key=lambda i: scores[i])
                text, ids, finish = pool[best]
                score = scores[best]
            else:
                # Jamais de bulle vide : « … » est déjà la réplique vide du format (render_messages).
                text, ids, finish = "…", samples[0][0], samples[0][1]
                score = None
        return {"reply": text, "persona": persona_name, "interlocutor": interlocutor,
                "seed": seed, "tokens_generated": len(ids),
                "seconds": round(time.monotonic() - t0, 3), "context_tokens": len(ctx),
                "context_limit": self.context_limit, "context_truncated": truncated,
                "history_turns_used": used, "candidates": candidates, "finish": finish,
                "score": None if score is None else round(score, 4),
                "candidates_kept": len(kept), "candidates_eligible": len(pool), "retried": retried,
                # Pour les mesures (chatbot.evaluate) : texte brut avant découpe, longueur moyenne
                # des candidates non vides (le reclassement favorise-t-il les réponses courtes ?).
                "raw_reply": self.tok.decode(ids),
                "candidates_mean_chars": (round(sum(len(t[0]) for t in kept) / len(kept), 1)
                                          if kept else None)}


# ---------------------------------------------------------------------------------------------
# Export et ligne de commande


def export_model(run_checkpoint, tokenizer_path, out_dir, chat_config=None):
    """Dossier modèle prêt pour ChatBot.load : ckpt.pt (float32), tokenizer.json, chat_config.json.

    chat_config.json reçoit la clé "format" : celle demandée, sinon celle apprise par le
    checkpoint affiné (méta « format »), sinon "repliques". Une contradiction est refusée.
    """
    config = dict(DEFAULT_CHAT_CONFIG, **(chat_config or {}))
    ckpt = torch.load(run_checkpoint, map_location="cpu", weights_only=True)
    trained = checkpoint_format(ckpt.get("meta", {}))
    config["format"] = (chat_config or {}).get("format") or trained or DEFAULT_FORMAT
    if trained and config["format"] != trained:
        raise ValueError(f"format demandé {config['format']!r}, mais ce checkpoint a été affiné "
                         f"au format {trained!r}")
    if get_format(config["format"])["personas"]:
        persona_info(config["default_persona"])  # une faute de frappe est refusée dès l'export
    else:
        config["default_persona"] = None  # tirets : pas de personnage
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(ckpt.get("meta", {}), exported_from=str(run_checkpoint))
    tok = load_tokenizer(tokenizer_path, check_texts=TOKENIZER_CHECK)
    if meta.get("tokenizer_sha256") and meta["tokenizer_sha256"] != tok.sha256:
        raise ValueError("ce tokenizer n'est pas celui du checkpoint")
    meta["tokenizer_sha256"] = tok.sha256
    torch.save({"model": {k: v.float() for k, v in ckpt["model"].items()},
                "model_config": ckpt["model_config"], "meta": meta}, out / "ckpt.pt")
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    (out / "chat_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m chatbot.generate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("export", help="prépare un dossier modèle pour le chat")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--tokenizer", required=True)
    pe.add_argument("--out", required=True)
    for key in ("temperature", "top_p", "mmi_lambda"):
        pe.add_argument("--" + key.replace("_", "-"), type=float, default=None)
    for key in ("max_new_tokens", "candidates"):
        pe.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    pe.add_argument("--default-persona", default=None)
    pe.add_argument("--format", choices=sorted(FORMATS), default=None,
                    help="défaut : le format appris par le checkpoint affiné, sinon repliques")

    pc = sub.add_parser("chat", help="parle au modèle ; plusieurs messages = conversation suivie")
    pc.add_argument("--model", required=True)
    pc.add_argument("--persona", default=None)
    pc.add_argument("--seed", type=int, default=None)
    pc.add_argument("--candidates", type=int, default=None)
    pc.add_argument("--device", default="cpu")
    pc.add_argument("messages", nargs="*", help="sans message : mode interactif")
    args = p.parse_args(argv)

    if args.cmd == "export":
        overrides = {k: getattr(args, k) for k in ("temperature", "top_p", "mmi_lambda",
                                                    "max_new_tokens", "candidates", "default_persona",
                                                    "format")
                     if getattr(args, k) is not None}
        out = export_model(args.ckpt, args.tokenizer, args.out, overrides)
        print(f"modèle exporté dans {out}")
        return

    bot = ChatBot.load(args.model, device=args.device)
    history = []
    inputs = iter(args.messages) if args.messages else iter(lambda: input("vous> "), "")
    for turn, message in enumerate(inputs):
        history.append({"role": "user", "content": message})
        # Graine fixe : décalée à chaque tour, sinon chaque tour rejouerait les mêmes tirages.
        seed = None if args.seed is None else (args.seed + turn) % 2**32
        r = bot.reply(history, persona=args.persona, seed=seed, candidates=args.candidates)
        history.append({"role": "assistant", "content": r["reply"]})
        # Format tirets : pas de nom de personnage, on affiche « VOUS » et « BOT ».
        print(f"\n{r['interlocutor'] or 'VOUS'} (vous) : {message}\n{r['persona'] or 'BOT'} : "
              f"{r['reply']}")
        print(f"   [{r['seconds']:.1f} s, {r['tokens_generated']} tokens, fin={r['finish']}, "
              f"score={r['score']}, graine={r['seed']}, contexte={r['context_tokens']}/"
              f"{r['context_limit']}, tronqué={r['context_truncated']}]", file=sys.stderr)


if __name__ == "__main__":
    from chatbot.accel import setup_console
    setup_console()  # Windows : sortie redirigée en UTF-8 (sinon « ≈ » ferait planter print)
    main()
