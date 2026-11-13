"""Pré-entraînement du modèle de démo (chatbot/pretrain.py) : données, échantillonneur, entraînement
minuscule sur CPU, reprise, benchmark, puis affinage par chatbot/finetune.py (inchangé)."""
import gzip
import hashlib
import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from chatbot import finetune as FT
from chatbot import pretrain as P
from chatbot.theatre import EXCLUSIONS, KIT_VAL
from ftb.model import load_checkpoint
from pipeline.io import write_docs

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "submissions/2026-11-11_153839/tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="tokenizer absent")

TINY = '{"d_model":32,"n_layer":1,"n_head":2,"context_length":64}'

PROSE = ("Le château se dressait au bord de la rivière, entouré de vieux chênes que le vent "
         "faisait gémir chaque soir.  Les paysans du village racontaient ’ des histoires ; "
         "personne ne les croyait vraiment.\n\n\n"
         "Un matin d'automne, le jeune comte revint de la ville avec une lettre cachetée ?\n"
         "Il la lut deux fois avant de la jeter au feu.\n\n"
         "Personne ne sut jamais ce qu'elle contenait, mais dès ce jour la maison changea.")
SCENE = [{"speaker": "Harpagon", "text": "Au voleur ! au voleur ! à l'assassin !"},
         {"speaker": "Cléante", "text": "Qu'avez-vous, mon père ? Que vous est-il arrivé ?"},
         {"speaker": "HARPAGON", "text": "On m'a dérobé mon argent.\nJe suis perdu."}]


def prose(i):
    return f"Chapitre {i}.\n\n" + PROSE


def val_passage():
    """Un passage de 120 mots copié d'une pièce de validation : doit être écarté."""
    words = KIT_VAL.read_text(encoding="utf-8").split()
    return " ".join(words[5000:5120])


@pytest.fixture(scope="module")
def tok():
    return FT.open_tokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def ref():
    return P.build_reference()


@pytest.fixture(scope="module")
def ids():
    """Un doc_id exclu pour obscénité (écarté) et un exclu car « pas du dialogue » (gardé)."""
    excl = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["documents"]
    obscene = next(k for k, v in excl.items() if v == "erotique_ou_obscene")
    not_dialogue = next(k for k, v in excl.items() if v == "pas_du_dialogue")
    return obscene, not_dialogue


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory, ids):
    d = tmp_path_factory.mktemp("sources")
    obscene, not_dialogue = ids
    docs = [{"id": f"doc:{i}", "text": prose(i)} for i in range(6)]
    docs += [{"id": "court", "text": "Trop court."},
             {"id": "cru", "text": PROSE + " foutre bordel putain salope " * 4},
             {"id": "val", "text": PROSE + "\n\n" + val_passage()},
             {"id": obscene, "text": PROSE},
             {"id": not_dialogue, "text": prose(99)}]
    write_docs(d / "docs.jsonl.zst", docs)
    with gzip.open(d / "melange.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps({"text": prose(i)}, ensure_ascii=False) + "\n")
    scenes = [{"id": f"s{i}", "doc_id": "piece", "turns": SCENE} for i in range(40)]
    write_docs(d / "scenes.jsonl.zst", scenes)
    return d


def read_bin(path):
    return np.fromfile(path, dtype=np.uint16)


# ---------------------------------------------------------------------------------------------
# Données


def test_clean_prose_keeps_paragraphs():
    out = P.clean_prose(PROSE)
    assert out.count("\n\n") == 2  # trois paragraphes, une seule ligne vide entre eux
    assert "\n\n\n" not in out and "’" not in out and " ;" not in out and " ?" not in out
    assert "cachetée?\nIl la lut" in out  # saut de ligne simple gardé dans un paragraphe


def test_data_filters_counted(tmp_path, tok, ref, fixtures, ids):
    stats = P.prepare_source("docs", [str(fixtures / "docs.jsonl.zst")], None, tmp_path, tok, ref)
    assert stats["documents_rejected"] == {"exclusion_erotique_ou_obscene": 1,
                                           "recouvrement_validation_moliere": 1,
                                           "trop_court": 1, "vocabulaire_cru": 1}
    assert stats["documents_kept"] == 7  # 6 + le document exclu du théâtre seulement
    assert ids[1] not in sum(stats["rejected_examples"].values(), [])
    arr = read_bin(tmp_path / "docs.bin")
    assert len(arr) == stats["tokens"]
    # <|endoftext|> au début, puis après chaque document : 7 documents -> 8 séparateurs
    assert arr[0] == tok.eot_id and arr[-1] == tok.eot_id
    assert int((arr == tok.eot_id).sum()) == 8
    text = tok.decode(arr.tolist())
    assert "Trop court" not in text and "foutre" not in text and "Chapitre 99." in text
    assert "rivière, entouré" in text and "\n\nUn matin" in text  # paragraphes conservés
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["sources"]["docs"]["key"]["tokenizer_sha256"] == tok.sha256
    assert stats["bytes"] == len("".join(d for d in text.split("<|endoftext|>")).encode())


def test_data_scenes_max_tokens_and_resume(tmp_path, tok, ref, fixtures, capsys):
    files = [str(fixtures / "scenes.jsonl.zst")]
    stats = P.prepare_source("theatre", files, 500, tmp_path, tok, ref)
    assert stats["tokens"] == 500 and stats["truncated_at_max_tokens"]
    arr = read_bin(tmp_path / "theatre.bin")
    assert len(arr) == 500
    text = tok.decode(arr.tolist())
    assert text.startswith("<|endoftext|>HARPAGON.\nAu voleur! au voleur!")
    assert "\n\n<|endoftext|>HARPAGON." in text  # une scène, un séparateur, la suivante
    mtime = os.path.getmtime(tmp_path / "theatre.bin")
    again = P.prepare_source("theatre", files, 500, tmp_path, tok, ref)
    assert again == stats and os.path.getmtime(tmp_path / "theatre.bin") == mtime
    assert "rien à refaire" in capsys.readouterr().err
    other = P.prepare_source("theatre", files, 400, tmp_path, tok, ref)  # autre clé : refait
    assert other["tokens"] == 400


def test_data_shuffle_and_workers_identical(tmp_path, tok, ref, fixtures):
    files = [str(fixtures / "melange.jsonl.gz")]
    a = P.prepare_source("m", files, None, tmp_path / "a", tok, ref, seed=1, workers=1)
    b = P.prepare_source("m", files, None, tmp_path / "b", tok, ref, seed=1, workers=2)
    c = P.prepare_source("m", files, None, tmp_path / "c", tok, ref, seed=2, workers=1)
    ta, tb, tc = (read_bin(tmp_path / x / "m.bin") for x in "abc")
    assert a["documents_kept"] == 300 and np.array_equal(ta, tb)  # même graine : même flux
    assert len(ta) == len(tc) and not np.array_equal(ta, tc)       # autre graine : autre ordre
    first = tok.decode(ta[:20].tolist())
    assert not first.startswith("<|endoftext|>Chapitre 0.\n")  # ordre du fichier mélangé


def write_mix(d, texts, prov_rows):
    """Un mélange façon officiel (champ text seul) et sa provenance.jsonl.gz à côté."""
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "train.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
    with gzip.open(d / P.PROVENANCE, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in prov_rows)
    return [str(d / "train.jsonl.gz")]


def prov_row(i, text, doc_id):
    return {"row_zero_based": i, "id": doc_id,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def test_mix_without_id_uses_provenance_for_exclusions(tmp_path, tok, ref, ids):
    """Le mélange officiel n'a pas d'id : on le reprend de provenance.jsonl.gz, vérifié."""
    texts = [prose(i) for i in range(3)]
    rows = [prov_row(0, texts[0], "a:0"), prov_row(1, texts[1], ids[0]), prov_row(2, texts[2], "a:2")]
    files = write_mix(tmp_path / "mix", texts, rows)
    stats = P.prepare_source("m", files, None, tmp_path / "out", tok, ref)
    assert stats["documents_rejected"] == {"exclusion_erotique_ou_obscene": 1}
    assert stats["rejected_examples"]["exclusion_erotique_ou_obscene"] == [ids[0]]
    assert stats["documents_kept"] == 2 and stats["ids_from_provenance"] == 3
    assert stats["documents_without_id"] == 0
    assert stats["key"]["files"][0]["provenance_sha256"]  # provenance changée = refaire

    # Provenance décalée (empreinte d'un autre texte) ou trop longue : arrêt, jamais d'id faux.
    bad = write_mix(tmp_path / "bad", texts, [rows[0], prov_row(1, texts[2], "x"), rows[2]])
    with pytest.raises(SystemExit):
        P.prepare_source("m", bad, None, tmp_path / "out_bad", tok, ref)
    longer = write_mix(tmp_path / "long", texts[:2], rows)
    with pytest.raises(SystemExit):
        P.prepare_source("m", longer, None, tmp_path / "out_long", tok, ref)


def test_mix_without_id_nor_provenance_is_reported(tmp_path, tok, ref, fixtures, capsys):
    stats = P.prepare_source("m", [str(fixtures / "melange.jsonl.gz")], None, tmp_path, tok, ref)
    assert stats["documents_without_id"] == 300 and stats["ids_from_provenance"] == 0
    assert "liste d'exclusion" in capsys.readouterr().err


def test_cli_data_limit_docs(tmp_path, fixtures):
    out = tmp_path / "pre"
    res = P.main(["data", "--out", str(out), "--source",
                  f"g={fixtures}/melange.jsonl.*:100000", "--limit-docs", "10"])
    assert res["g"]["documents_read"] == 10
    with pytest.raises(SystemExit):
        P.main(["data", "--out", str(out), "--source", f"g={fixtures}/absent*.zst"])


# ---------------------------------------------------------------------------------------------
# Échantillonneur


def test_sampler_shares_no_crossing_and_shift():
    arrays = [np.arange(i * 10_000, i * 10_000 + n) for i, n in enumerate([200, 5000, 900])]
    s = P.ShareSampler(arrays, [0.5, 0.2, 0.3], seq_len=16, seed=0)
    xy = s.batch(5000).numpy()
    share = np.array(s.windows_per_file) / 5000
    assert np.all(np.abs(share - [0.5, 0.2, 0.3]) < 0.03)  # poids = part, pas poids × taille
    src = xy[:, 0] // 10_000
    assert np.all(xy // 10_000 == src[:, None])  # jamais à cheval sur deux sources
    assert np.all(xy[:, 1:] == xy[:, :-1] + 1)   # y = x décalé d'un token
    st = s.state()
    nxt = s.batch(3)
    s.load_state(st)
    assert torch.equal(s.batch(3), nxt)  # l'état du générateur se sauvegarde


# ---------------------------------------------------------------------------------------------
# Entraînement


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, tok, ref, fixtures):
    d = tmp_path_factory.mktemp("pre")
    P.prepare_source("melange", [str(fixtures / "melange.jsonl.gz")], None, d, tok, ref)
    P.prepare_source("theatre", [str(fixtures / "scenes.jsonl.zst")], None, d, tok, ref)
    return d


def train_args(data_dir, fixtures, out, *extra, tokens=8 * 32 * 20):
    return ["train", "--data", str(data_dir), "--weights", "melange=0.7", "theatre=0.3",
            "--model-config", TINY, "--val", str(fixtures / "scenes.jsonl.zst"),
            "--out", str(out), "--tokens", str(tokens), "--batch-size", "8", "--micro-batch", "4",
            "--seq-len", "32", "--lr", "1e-2", "--min-lr", "1e-3", "--warmup-steps", "3",
            "--eval-every", "5", "--device", "cpu", *extra]


@pytest.fixture(scope="module")
def run_a(tmp_path_factory, data_dir, fixtures):
    out = tmp_path_factory.mktemp("run_a")
    report = P.main(train_args(data_dir, fixtures, out))
    return out, report


def test_train_end_to_end(run_a, data_dir, fixtures):
    out, report = run_a
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    assert [r["step"] for r in log] == [0, 5, 10, 15, 20]
    assert log[-1]["val_loss"] < log[0]["val_loss"] - 1.0  # la loss baisse nettement
    assert log[-1]["train_loss"] < log[1]["train_loss"]
    for name in ("best.pt", "last.pt"):
        model, ckpt = load_checkpoint(out / name)
        assert ckpt["model_config"]["d_model"] == 32 and ckpt["meta"]["pretrain"]
    assert not (out / "state.pt").exists()
    saved = json.loads((out / "report.json").read_text())
    assert "aucun poids externe" in saved["note"] and saved["tokens_seen"] == 8 * 32 * 20
    assert sum(s["windows_sampled"] for s in saved["sources"]) == 8 * 20
    assert saved["best"]["val_loss"] == min(r["val_loss"] for r in log)
    with pytest.raises(SystemExit):  # run terminé : on ne l'écrase pas
        P.main(train_args(data_dir, fixtures, out))


def test_resume_continues_same_curve(tmp_path, data_dir, fixtures, run_a):
    out = tmp_path / "run_b"
    res = P.main(train_args(data_dir, fixtures, out, "--stop-at-step", "10"))
    assert res["status"].startswith("interrompu") and (out / "state.pt").exists()
    with pytest.raises(SystemExit):  # sans --resume : refus
        P.main(train_args(data_dir, fixtures, out))
    with pytest.raises(SystemExit):  # autres réglages : reprise refusée
        P.main(train_args(data_dir, fixtures, out, "--resume", "--lr", "5e-3"))
    with pytest.raises(SystemExit, match="val"):  # autre validation : courbe incomparable
        P.main(train_args(data_dir, fixtures, out, "--resume", "--val-max-windows", "1"))
    P.main(train_args(data_dir, fixtures, out, "--resume"))
    a, _ = load_checkpoint(run_a[0] / "last.pt")
    b, _ = load_checkpoint(out / "last.pt")
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.allclose(va, vb, atol=1e-5), ka
    la = [json.loads(l) for l in open(run_a[0] / "log.jsonl")]
    lb = [json.loads(l) for l in open(out / "log.jsonl")]
    assert [r["val_loss"] for r in la] == pytest.approx([r["val_loss"] for r in lb], abs=1e-4)
    rep = json.loads((out / "report.json").read_text())
    assert rep["resumed_from_steps"] == [10]


def test_sigterm_saves_state(tmp_path, data_dir, fixtures, monkeypatch):
    real = P.train_step

    def step_then_signal(*args, **kw):
        res = real(*args, **kw)
        if args[4] == 6:  # pas d'indice 6 : on s'envoie SIGTERM, comme `kill` pendant la nuit
            os.kill(os.getpid(), signal.SIGTERM)
        return res

    monkeypatch.setattr(P, "train_step", step_then_signal)
    before = signal.getsignal(signal.SIGTERM)
    out = tmp_path / "sig"
    res = P.main(train_args(data_dir, fixtures, out))
    assert res == {"status": "interrompu (SIGTERM)", "step": 7}
    assert torch.load(out / "state.pt", weights_only=True)["step"] == 7
    assert signal.getsignal(signal.SIGTERM) == before  # gestionnaire d'origine remis


def test_nan_guard_stops(tmp_path, data_dir, fixtures, monkeypatch, capsys):
    real = P.train_step

    def nan_at_4(*args, **kw):
        loss, gnorm, lr = real(*args, **kw)
        return (float("nan"), gnorm, lr) if args[4] == 3 else (loss, gnorm, lr)

    monkeypatch.setattr(P, "train_step", nan_at_4)
    out = tmp_path / "nan"
    res = P.main(train_args(data_dir, fixtures, out))
    assert res == {"status": "nan", "step": 4}
    assert not (out / "report.json").exists() and (out / "best.pt").exists()
    # Divergence avant la 1re évaluation (pas 5) : pas de state.pt, et le message le dit.
    assert not (out / "state.pt").exists()
    assert "aucun état de reprise" in capsys.readouterr().err


def test_exit_codes():
    """`train ... && finetune` : le && ne passe que si le run est allé au bout."""
    assert P.exit_code({"status": "nan", "step": 4}) == 1
    assert P.exit_code({"status": "interrompu (SIGTERM)", "step": 7}) == 3
    assert P.exit_code({"status": "interrompu (--stop-at-step)", "step": 10}) == 3
    assert P.exit_code({"note": "...", "best": {}}) == 0           # report.json d'un run terminé
    assert P.exit_code({"tokens_per_s": 1.0}) == 0                 # benchmark
    assert P.exit_code({"status": {"tokens": 3}}) == 0             # commande data, source « status »


def test_best_compares_exact_loss(run_a):
    out, report = run_a
    best = report["best"]
    assert best["val_loss_exact"] == pytest.approx(best["val_loss"], abs=1e-5)
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    assert best["step"] == min(log, key=lambda r: r["val_loss"])["step"]


def test_benchmark_writes_nothing(tmp_path, data_dir, fixtures, capsys):
    out = tmp_path / "bench"
    res = P.main(train_args(data_dir, fixtures, out, "--benchmark-steps", "2"))
    assert res["tokens_per_s"] > 0 and not out.exists()
    assert "tokens/s" in capsys.readouterr().err


def test_refuses_bad_settings(tmp_path, data_dir, fixtures):
    with pytest.raises(SystemExit):  # 20 pas : warmup 20 refusé
        P.main(train_args(data_dir, fixtures, tmp_path / "w") + ["--warmup-steps", "20"])
    with pytest.raises(SystemExit):  # source inconnue
        P.main(train_args(data_dir, fixtures, tmp_path / "x") + ["--weights", "absente=1"])
    with pytest.raises(SystemExit):  # micro-batch qui ne divise pas le lot
        P.main(train_args(data_dir, fixtures, tmp_path / "y") + ["--micro-batch", "3"])
    assert not (tmp_path / "w").exists()


def test_finetune_accepts_pretrained_checkpoint(tmp_path, run_a, fixtures):
    """chatbot/finetune.py (non modifié) affine le checkpoint de taille différente."""
    scenes = str(fixtures / "scenes.jsonl.zst")
    report = FT.main(["--init", str(run_a[0] / "best.pt"), "--tokenizer", str(TOKENIZER),
                      "--data", scenes, "--val", scenes, "--out", str(tmp_path / "ft"),
                      "--tokens", str(4 * 32 * 6), "--batch-size", "4", "--seq-len", "32",
                      "--warmup-steps", "1", "--eval-every", "3", "--device", "cpu"])
    assert report["n_params"] == load_checkpoint(run_a[0] / "best.pt")[0].num_params()
    assert report["best"]["val_loss"] <= report["val_step0"]["val_loss"]
    model, ckpt = load_checkpoint(tmp_path / "ft" / "best.pt")
    assert ckpt["model_config"]["d_model"] == 32 and ckpt["meta"]["finetune"]
