"""Ancrage du choix de la réponse sur le dernier message (chatbot/ancrage.py)."""
from pathlib import Path

import pytest
import torch

from chatbot import finetune as FT
from chatbot import generate as G
from chatbot.ancrage import contexte_neutre, installer, regler
from ftb.model import build_model

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/2026-11-11_153839"
TINY = {"vocab_size": 8192, "context_length": 64, "d_model": 32, "n_layer": 2, "n_head": 2,
        "mlp_ratio": 4, "dropout": 0.0, "tie_embeddings": True}
CONV = [{"role": "user", "content": "Bonjour, cher ami."},
        {"role": "assistant", "content": "Je vous salue, monsieur."},
        {"role": "user", "content": "Quel temps fait-il ?"}]
REPONSES = ["Il fait beau, monsieur.", "Je ne sais."]


@pytest.fixture(scope="module")
def tok():
    if not (SUB / "tokenizer.json").exists():
        pytest.skip("tokenizer absent")
    return FT.open_tokenizer(SUB / "tokenizer.json")


def nouveau_bot(tok):
    torch.manual_seed(0)
    return G.ChatBot(build_model(TINY), tok, {"max_new_tokens": 12, "candidates": 3})


def test_contexte_neutre_remplace_seulement_le_dernier_message(tok):
    bot = nouveau_bot(tok)
    ids, used, _ = bot.build_context(CONV, "MOLIÈRE", "LA GRANGE", 60)
    vrai = bot.tok.decode(ids)
    neutre = bot.tok.decode(contexte_neutre(bot, CONV, "MOLIÈRE", "LA GRANGE", used))
    assert "Quel temps fait-il" in vrai and "Quel temps fait-il" not in neutre
    assert neutre == vrai.replace("Quel temps fait-il?", "…")
    assert neutre.endswith("LA GRANGE.\n…\n\nMOLIÈRE.\n")


def test_contexte_neutre_garde_la_meme_coupe_quand_elle_commence_sur_une_reponse(tok):
    """Longue conversation coupée : si le premier message gardé est une réponse du bot, le contexte
    neutre la garde aussi. Seul le dernier message diffère du vrai contexte."""
    conv = CONV[:2] + [{"role": "user", "content": "Il pleut sur Paris."},
                       {"role": "assistant", "content": "Et demain ?"},
                       {"role": "user", "content": "Quel temps fait-il ?"}]
    bot = nouveau_bot(tok)
    coupes = 0
    for budget in range(12, 61):
        try:
            ids, used, _ = bot.build_context(conv, "MOLIÈRE", "LA GRANGE", budget)
        except ValueError:
            continue
        if used == len(conv) or conv[len(conv) - used]["role"] != "assistant":
            continue
        coupes += 1
        vrai = bot.tok.decode(ids)
        neutre = bot.tok.decode(contexte_neutre(bot, conv, "MOLIÈRE", "LA GRANGE", used))
        assert neutre == vrai.replace("Quel temps fait-il?", "…")
    assert coupes, "aucun budget ne coupe sur une réponse du bot : le test ne vérifie rien"


def test_mu_zero_rend_le_choix_d_origine(tok):
    bot = nouveau_bot(tok)
    ctx, _, _ = bot.build_context(CONV, "MOLIÈRE", "LA GRANGE", 60)
    attendu = bot.mmi_scores(ctx, "MOLIÈRE", REPONSES, 1.0)
    installer(bot, mu=0.0)
    ctx2, _, _ = bot.build_context(CONV, "MOLIÈRE", "LA GRANGE", 60)
    assert ctx2 == ctx
    assert bot.mmi_scores(ctx2, "MOLIÈRE", REPONSES, 1.0) == attendu


def test_mu_un_mesure_ce_que_le_dernier_message_apporte(tok):
    bot = installer(nouveau_bot(tok), mu=1.0)
    ctx, used, _ = bot.build_context(CONV, "MOLIÈRE", "LA GRANGE", 60)
    scores = bot.mmi_scores(ctx, "MOLIÈRE", REPONSES, 1.0)
    ids = [bot.tok.encode(r + "\n\n") for r in REPONSES]
    avec = bot.mean_logprob(ctx, ids)
    sans = bot.mean_logprob(contexte_neutre(bot, CONV, "MOLIÈRE", "LA GRANGE", used), ids)
    for s, a, h, r in zip(scores, avec, sans, ids):
        assert s == pytest.approx((a - h) * len(r), abs=1e-4)


def test_installer_deux_fois_ne_s_empile_pas_et_mu_est_borne(tok):
    bot = installer(installer(nouveau_bot(tok), mu=1.0), mu=0.5)
    assert bot._ancrage_mu == 0.5
    assert bot._ancrage_origine[0].__func__ is G.ChatBot.build_context
    with pytest.raises(ValueError):
        regler(bot, 3.0)


def test_reply_fonctionne_avec_l_ancrage(tok):
    bot = installer(nouveau_bot(tok), mu=1.0)
    r = bot.reply(CONV, seed=1)
    assert isinstance(r["reply"], str) and r["reply"]
    assert bot.reply(CONV, seed=1)["reply"] == r["reply"]  # même graine, même réponse


def test_anchor_mu_du_chat_config_branche_l_ancrage(tok):
    # Défaut 0 : aucune méthode n'est remplacée, les modèles déjà exportés ne changent pas.
    sans = nouveau_bot(tok)
    assert G.DEFAULT_CHAT_CONFIG["anchor_mu"] == 0.0 and not hasattr(sans, "_ancrage_origine")
    assert sans.reply(CONV, seed=1)["anchor_mu"] == 0.0
    # « anchor_mu » dans le chat_config = installer() à la main : mêmes poids, même réponse.
    torch.manual_seed(0)
    avec = G.ChatBot(build_model(TINY), tok, {"max_new_tokens": 12, "candidates": 3, "anchor_mu": 1.0})
    temoin = installer(nouveau_bot(tok), mu=1.0)
    r = avec.reply(CONV, seed=1)
    assert avec._ancrage_mu == 1.0 and r["anchor_mu"] == 1.0
    assert r["reply"] == temoin.reply(CONV, seed=1)["reply"] and r["score"] == temoin.reply(CONV, seed=1)["score"]
    with pytest.raises(ValueError):
        G.ChatBot(build_model(TINY), tok, {"anchor_mu": 3.0})


def test_export_refuse_un_anchor_mu_hors_bornes(tmp_path):
    with pytest.raises(SystemExit):  # refusé par la ligne de commande, avant de lire le checkpoint
        G.main(["export", "--ckpt", str(tmp_path / "absent.pt"), "--tokenizer", str(tmp_path / "t.json"),
                "--out", str(tmp_path / "m"), "--anchor-mu", "3"])
    assert not (tmp_path / "m").exists()
