"""Lanceur RTX (chatbot/pipeline_rtx.py) avec des commandes SIMULÉES : aucun entraînement ici.

Le faux exécutant lit la commande (module, --out) et écrit les fichiers que la vraie commande
écrirait (report.json, best.pt, chat_config.json, metrics.json, ligne de benchmark) : on teste
l'état, la reprise, le choix « auto », l'étape synthétique facultative, les échecs, l'archive.
"""
import argparse
import json
import zipfile
from pathlib import Path

import pytest

from chatbot import pipeline_rtx as L

ROOT = Path(__file__).resolve().parents[1]
REAL = L.charger_recettes()


# ---------------------------------------------------------------------------------------------
# Recette de test et faux exécutant


def petite_recette(tmp_path):
    """La vraie recette, avec des chemins de données créés dans tmp_path (fichiers vides)."""
    rec = json.loads(json.dumps(REAL))
    (tmp_path / "tok.json").write_text("{}")
    rec["commun"]["tokenizer"] = {"valeur": "tok.json"}
    data = tmp_path / "pre"
    data.mkdir()
    meta = {"tokenizer_sha256": "x", "sources": {n: {"file": f"{n}.bin", "tokens": 10, "key": {}}
                                                 for n in L.val(rec["base"]["parts"])}}
    (data / "meta.json").write_text(json.dumps(meta))
    for n in meta["sources"]:
        (data / f"{n}.bin").write_bytes(b"\0" * 20)
    rec["base"]["donnees"] = {"valeur": "pre"}
    for f in ("val.zst", "a.zst", "b.zst", "c.zst", "d.zst", "val2.zst"):
        (tmp_path / f).write_text("x")
    rec["base"]["validation"] = {"valeur": "val.zst"}
    for cb, (d1, d2, v) in {"moliere": ("a.zst", "b.zst", "val.zst"),
                            "francais": ("c.zst", "d.zst", "val2.zst")}.items():
        c = rec["chatbots"][cb]
        c["validation"] = {"valeur": v}
        c["etape1"]["donnees"] = {"valeur": [d1, d2]}
        c["synthetique"]["fichier"] = {"valeur": f"syn_{cb}.zst"}
        c["synthetique"]["avec"] = {"valeur": [d2]}
    return rec


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class FauxExecutant:
    """Remplace executer() : écrit les sorties attendues ; codes de retour programmables."""

    def __init__(self, codes=None, tps=None):
        self.appels = []
        self.codes = dict(codes or {})  # {(module, out_contient): [codes successifs]}
        self.tps = tps or {}           # benchmark : {micro_lot ou None: tokens/s ; absent = échec}

    def code_pour(self, cmd):
        for (mod, morceau), seq in self.codes.items():
            if mod in cmd and any(morceau in str(c) for c in cmd) and seq:
                return seq.pop(0)
        return 0

    def __call__(self, cmd, journal, root):
        cmd = [str(c) for c in cmd]
        self.appels.append(cmd)
        Path(journal).parent.mkdir(parents=True, exist_ok=True)
        if "--benchmark-steps" in cmd:
            micro = int(arg(cmd, "--micro-batch")) if "--micro-batch" in cmd else None
            if micro not in self.tps:
                Path(journal).write_text("CUDA out of memory\n", encoding="utf-8")
                return 1, False
            Path(journal).write_text(
                f"modèle ...\nbenchmark : 30 pas en 10.0 s = {self.tps[micro]:,.0f} tokens/s | budget "
                f"x tokens ≈ 1.0 h (hors évaluations) | précision bf16 | compile actif | mémoire GPU "
                f"max 12.5 Go\n", encoding="utf-8")
            return 0, False
        code = self.code_pour(cmd)
        if code == 1:  # vraie erreur Python : une trace dans le journal
            Path(journal).write_text("Traceback (most recent call last):\nRuntimeError: simulé\n",
                                     encoding="utf-8")
        if code != 0:
            return code, False
        out = Path(root) / arg(cmd, "--out")
        out.mkdir(parents=True, exist_ok=True)
        if "chatbot.pretrain" in cmd or "chatbot.finetune" in cmd:
            data = cmd[cmd.index("--data") + 1:cmd.index("--weights")] if "chatbot.finetune" in cmd else []
            rep = {"best": {"val_loss": 2.0, "val_bpb": 1.2, "step": 10}, "tokens_seen": 1000,
                   "val_step0": {"val_loss": 3.0, "val_bpb": 1.5}, "steps": 10, "seconds": 5.0,
                   "resumed_from_steps": [], "n_params": 123,
                   "data": [{"path": d, "tokens_trained": 300 if i == 0 else 700} for i, d in enumerate(data)]}
            (out / "report.json").write_text(json.dumps(rep))
            (out / "log.jsonl").write_text('{"step": 0, "val_loss": 3.0}\n')
            (out / "best.pt").write_bytes(b"poids")
            (out / "state.pt").unlink(missing_ok=True)
        elif "export" in cmd:
            for f in ("ckpt.pt", "tokenizer.json"):
                (out / f).write_bytes(b"x")
            (out / "chat_config.json").write_text(json.dumps({"format": arg(cmd, "--format")}))
        elif "chatbot.evaluate" in cmd:
            (out / "metrics.json").write_text(json.dumps(
                {"val_next_turn": {"val_bpb": 1.1}, "panel": {"tout": {"n": 3}}, "n_replies": 3}))
            (out / "transcript.md").write_text("# panel\n")
        return 0, False

    def etapes(self):
        """Noms d'étapes appelées, dans l'ordre (déduits de --out)."""
        return [Path(arg(c, "--out")).name for c in self.appels if "--out" in c]


def ns(sortie, **kw):
    base = dict(sortie=str(sortie), taille="S", budget_heures=None, seulement=None, repartir=False,
                micro_lot=None)
    base.update(kw)
    return argparse.Namespace(**base)


def lancer(tmp_path, rec, runner, **kw):
    return L.cmd_lancer(ns(tmp_path / "sorties", **kw), rec, runner, device="cuda", root=tmp_path)


# ---------------------------------------------------------------------------------------------
# Recettes


def test_recettes_chaque_valeur_a_un_pourquoi():
    def walk(node, path):
        if isinstance(node, dict) and "valeur" in node:
            assert node.get("pourquoi"), f"{path} : pas de « pourquoi »"
        elif isinstance(node, dict):
            for k, v in node.items():
                if k != "note":
                    walk(v, f"{path}.{k}")
        else:
            pytest.fail(f"{path} : valeur sans justification")
    walk(REAL, "recettes")


def test_tailles_des_modeles():
    import torch
    from ftb.model import build_model
    from chatbot.pretrain import KIT_MODEL
    n = {}
    for t in ("S", "M"):
        with torch.device("meta"):
            n[t] = build_model(dict(KIT_MODEL, **L.val(REAL["tailles"][t]["modele"]))).num_params()
    with torch.device("meta"):
        n["essai"] = build_model(dict(KIT_MODEL, **L.val(REAL["essai"]["modele"]))).num_params()
    assert 29.0e6 < n["S"] < 30.0e6 and 53e6 < n["M"] < 57e6 and n["essai"] < 1e6
    parts = L.val(REAL["base"]["parts"])
    assert abs(sum(parts.values()) - 1) < 1e-9


def test_commandes_reelles_acceptees_par_les_vrais_parseurs():
    """Chaque option écrite par le lanceur existe bien dans pretrain et finetune."""
    from chatbot import finetune, pretrain
    ctx = L.Ctx(REAL, "S", ROOT / "sorties_test_inexistant", ROOT, "cuda", micro=64)
    etapes = L.plan(ctx)
    for e in etapes:
        if "chatbot.pretrain" in e["cmd"]:
            a = pretrain.build_parser().parse_args(e["cmd"][3:])
            assert a.tokens == 1_500_000_000 and a.batch_size == 128 and a.micro_batch == 64
            assert a.compile and a.resume and a.device == "cuda" and a.eval_every == 1000
        elif "chatbot.finetune" in e["cmd"]:
            a = finetune.build_parser().parse_args(e["cmd"][3:])
            assert a.resume and a.micro_batch is None  # lot 64 : le micro-lot 64 = lot entier
    fmts = {arg(e["cmd"], "--format") for e in etapes if e["type"] == "export"}
    assert fmts == {"repliques", "tirets"}
    # Sur Mac / processeur : jamais de compilation (recette « cuda »)
    assert "--compile" not in L.plan(L.Ctx(REAL, "S", ROOT / "x", ROOT, "cpu"))[0]["cmd"]


# ---------------------------------------------------------------------------------------------
# Plan : étape synthétique facultative


def test_etape_synthetique_absente_puis_presente(tmp_path):
    rec = petite_recette(tmp_path)
    ctx = L.Ctx(rec, "S", tmp_path / "sorties", tmp_path, "cuda")
    noms = [e["nom"] for e in L.plan(ctx)]
    assert noms == ["base", "moliere_etape1", "francais_etape1", "moliere_export", "francais_export",
                    "moliere_evaluation", "francais_evaluation"]
    (tmp_path / "syn_moliere.zst").write_text("x")
    etapes = {e["nom"]: e for e in L.plan(ctx)}
    assert "moliere_synthetique" in etapes and "francais_synthetique" not in etapes
    syn = etapes["moliere_synthetique"]["cmd"]
    assert arg(syn, "--init").endswith("moliere_etape1/best.pt")
    assert arg(syn, "--data") == "syn_moliere.zst"  # le synthétique en premier
    assert arg(etapes["moliere_export"]["cmd"], "--ckpt").endswith("moliere_synthetique/best.pt")
    assert arg(etapes["francais_export"]["cmd"], "--ckpt").endswith("francais_etape1/best.pt")


def test_synthetique_apparu_en_cours_de_route_n_est_pas_ajoute(tmp_path):
    """Composition figée sans synthétique : un fichier arrivé ensuite ne change rien."""
    rec = petite_recette(tmp_path)
    run = FauxExecutant(codes={("chatbot.finetune", "moliere_etape1"): [3]})
    assert lancer(tmp_path, rec, run) == 3  # coupure pendant l'étape 1
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["composition"] == {"moliere": ["moliere_etape1"], "francais": ["francais_etape1"]}
    (tmp_path / "syn_moliere.zst").write_text("x")
    run2 = FauxExecutant()
    assert lancer(tmp_path, rec, run2) == 0
    assert "moliere_synthetique" not in run2.etapes()


def test_synthetique_garde_si_l_etape1_est_coupee(tmp_path):
    """Fichier synthétique présent dès le départ, coupure pendant moliere_etape1 : à la relance,
    moliere_synthetique reste au plan, l'export prend SON best.pt et le bilan donne sa part."""
    rec = petite_recette(tmp_path)
    (tmp_path / "syn_moliere.zst").write_text("x")
    run = FauxExecutant(codes={("chatbot.finetune", "moliere_etape1"): [3]})
    assert lancer(tmp_path, rec, run) == 3
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["composition"]["moliere"] == ["moliere_etape1", "moliere_synthetique"]
    assert "moliere_synthetique" not in etat["etapes"]  # pas encore commencée
    ctx = L.Ctx(rec, "S", tmp_path / "sorties", tmp_path, "cuda")
    etapes = {e["nom"]: e for e in L.plan(ctx, etat)}
    assert "moliere_synthetique" in etapes
    assert arg(etapes["moliere_export"]["cmd"], "--ckpt").endswith("moliere_synthetique/best.pt")
    run2 = FauxExecutant()
    assert lancer(tmp_path, rec, run2) == 0
    assert run2.etapes()[:2] == ["moliere_etape1", "moliere_synthetique"]
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["moliere_synthetique"]["statut"] == "fini"
    rep = json.loads((tmp_path / "sorties/modeles/chatbot_moliere/training_report.json").read_text())
    assert rep["part_synthetique"]["part_dans_l_etape"] == 0.3


def test_part_synthetique_dans_le_bilan(tmp_path):
    rec = petite_recette(tmp_path)
    (tmp_path / "syn_moliere.zst").write_text("x")
    run = FauxExecutant()
    assert lancer(tmp_path, rec, run) == 0
    rep = json.loads((tmp_path / "sorties/modeles/chatbot_moliere/training_report.json").read_text())
    assert rep["part_synthetique"]["part_dans_l_etape"] == 0.3
    assert rep["part_synthetique"]["part_de_tout_l_affinage"] == round(300 / 2000, 4)
    assert rep["format"] == "repliques"
    fr = json.loads((tmp_path / "sorties/modeles/chatbot_francais/training_report.json").read_text())
    assert fr["part_synthetique"] is None and fr["format"] == "tirets"


# ---------------------------------------------------------------------------------------------
# État, reprise, échec


def test_reprise_apres_interruption_sans_refaire_les_etapes_finies(tmp_path, capsys):
    rec = petite_recette(tmp_path)
    run = FauxExecutant(codes={("chatbot.finetune", "moliere_etape1"): [3]})
    assert lancer(tmp_path, rec, run) == 3
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["base"]["statut"] == "fini"
    assert etat["etapes"]["moliere_etape1"]["statut"] == "interrompu"
    assert "relance la MÊME commande" in capsys.readouterr().out
    # Une coupure laisse state.pt : la reprise doit le constater et passer --resume
    (tmp_path / "sorties/runs/moliere_etape1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sorties/runs/moliere_etape1/state.pt").write_bytes(b"etat")
    run2 = FauxExecutant()
    assert lancer(tmp_path, rec, run2) == 0
    assert run2.etapes()[0] == "moliere_etape1"  # la base n'est pas refaite
    assert "base" not in run2.etapes()
    assert all("--resume" in c for c in run2.appels if "chatbot.finetune" in c)
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["moliere_etape1"]["reprises"] == 1
    assert all(etat["etapes"][n]["statut"] == "fini" for n in etat["plan"])
    assert (tmp_path / "sorties/bilan.json").exists()
    # Troisième lancement : rien à refaire
    run3 = FauxExecutant()
    assert lancer(tmp_path, rec, run3) == 0 and run3.appels == []


def test_echec_d_une_etape_arrete_proprement(tmp_path, capsys):
    rec = petite_recette(tmp_path)
    run = FauxExecutant(codes={("chatbot.finetune", "francais_etape1"): [1]})
    assert lancer(tmp_path, rec, run) == 1
    out = capsys.readouterr().out
    assert "ARRÊT à l'étape francais_etape1 (echec, code 1)" in out
    assert "journaux/francais_etape1.log" in out
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["francais_etape1"]["statut"] == "echec"
    assert "moliere_export" not in run.etapes() and "chatbot_moliere" not in run.etapes()


def test_sortie_attendue_absente_est_un_echec(tmp_path):
    rec = petite_recette(tmp_path)

    def muet(cmd, journal, root):  # code 0 mais aucun fichier écrit
        return 0, False
    assert lancer(tmp_path, rec, muet) == 1
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert "sortie attendue absente" in etat["etapes"]["base"]["message"]


def test_fini_mais_non_note_est_constate(tmp_path):
    """Lanceur coupé juste après la fin d'une étape : report.json existe, on ne la relance pas."""
    rec = petite_recette(tmp_path)
    run = FauxExecutant(codes={("chatbot.finetune", "moliere_etape1"): [3]})
    lancer(tmp_path, rec, run)
    out = tmp_path / "sorties/runs/moliere_etape1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"best": {}, "tokens_seen": 1}))
    run2 = FauxExecutant()
    assert lancer(tmp_path, rec, run2) == 0
    assert "moliere_etape1" not in run2.etapes()


def test_autre_taille_refusee(tmp_path):
    rec = petite_recette(tmp_path)
    lancer(tmp_path, rec, FauxExecutant(codes={("chatbot.pretrain", "base"): [3]}))
    assert lancer(tmp_path, rec, FauxExecutant(), taille="M") == 2
    run = FauxExecutant()
    assert lancer(tmp_path, rec, run, taille="auto") == 0  # auto = la taille commencée
    assert json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())["taille"] == "S"


def test_donnees_absentes_refus(tmp_path, capsys):
    rec = petite_recette(tmp_path)
    (tmp_path / "pre/wikipedia.bin").unlink()
    assert lancer(tmp_path, rec, FauxExecutant()) == 2
    assert "pre/wikipedia.bin" in capsys.readouterr().out


def test_seulement_refait_un_chatbot_sans_la_base(tmp_path):
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant()) == 0
    (tmp_path / "syn_moliere.zst").write_text("x")  # nouveau fichier : pris en compte au refait
    run = FauxExecutant()
    assert lancer(tmp_path, rec, run, seulement="moliere") == 0
    assert run.etapes() == ["moliere_etape1", "moliere_synthetique", "chatbot_moliere", "moliere"]
    assert list((tmp_path / "sorties/archives").glob("*/runs/moliere_etape1/report.json"))
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["francais_export"]["statut"] == "fini"


def test_repartir_sans_seulement_refuse(tmp_path):
    assert lancer(tmp_path, petite_recette(tmp_path), FauxExecutant(), repartir=True) == 2


def test_mesurer_prend_le_verrou(tmp_path, monkeypatch):
    """`mesurer` pendant un `lancer` : refusé (un seul calcul à la fois sur la carte)."""
    monkeypatch.setattr(L, "installer_arrets", lambda: None)  # ne pas toucher aux signaux de pytest
    monkeypatch.setattr(L, "mesurer", lambda *a, **k: pytest.fail("mesure lancée malgré le verrou"))
    with L.Verrou(tmp_path / "lanceur.verrou"):
        with pytest.raises(SystemExit, match="autre lanceur"):
            L.main(["--sortie", str(tmp_path), "mesurer", "--tailles", "essai"])


def test_arret_demande_relaye_sigterm_et_leve_keyboardinterrupt(monkeypatch):
    import signal

    class Enfant:
        recu = []

        def poll(self):
            return None

        def send_signal(self, sig):
            self.recu.append(sig)

    monkeypatch.setattr(L, "ENFANT", Enfant())
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        L.arret_demande(signal.SIGTERM, None)
    if L.os.name != "nt":
        assert Enfant.recu == [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):  # Windows : l'enfant le reçoit déjà de la console
        with pytest.raises(KeyboardInterrupt):
            L.arret_demande(signal.SIGBREAK, None)
        assert Enfant.recu == [] or Enfant.recu == [signal.SIGTERM]


ENFANT_SCRIPT = """
import os, sys, time, pathlib
from chatbot.accel import StopRequest
flag, pidfile = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with StopRequest() as stop:
    pidfile.write_text(str(os.getpid()))
    print("pret", flush=True)
    while not stop.requested:
        time.sleep(0.05)
flag.write_text(stop.requested)          # = « state.pt sauvé »
print("etat sauve", flush=True)
sys.exit(3)
"""

LANCEUR_SCRIPT = """
import sys
from pathlib import Path
from chatbot import pipeline_rtx as L
L.installer_arrets()
code, stop = L.executer([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]], Path(sys.argv[4]),
                        Path(sys.argv[5]))
print("RESULTAT", code, stop, flush=True)
"""


def demarrer(tmp_path):
    import os
    import subprocess
    import sys
    import time
    (tmp_path / "enfant.py").write_text(ENFANT_SCRIPT, encoding="utf-8")
    (tmp_path / "lanceur.py").write_text(LANCEUR_SCRIPT, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen([sys.executable, str(tmp_path / "lanceur.py"), str(tmp_path / "enfant.py"),
                             str(tmp_path / "sauve"), str(tmp_path / "pid"), str(tmp_path / "j.log"),
                             str(ROOT)], env=env, stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    while not (tmp_path / "pid").exists() or not (tmp_path / "pid").read_text():
        assert time.time() - t0 < 30, "l'enfant n'a pas démarré"
        time.sleep(0.05)
    time.sleep(0.3)  # gestionnaires installés
    return proc, int((tmp_path / "pid").read_text())


@pytest.mark.skipif(L.os.name == "nt", reason="signaux POSIX (sous Windows : Ctrl-Break réel)")
def test_sigterm_au_lanceur_l_etape_sauve_puis_code_3(tmp_path):
    """Vrais processus : SIGTERM au lanceur -> relayé à l'étape, qui sauve ; le lanceur l'attend."""
    import signal
    proc, _ = demarrer(tmp_path)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert (tmp_path / "sauve").read_text() == "SIGTERM"
    assert "RESULTAT 3 True" in out and "attend que l'étape sauve" in out


@pytest.mark.skipif(L.os.name == "nt", reason="signaux POSIX")
def test_lanceur_mort_l_etape_sauve_quand_meme(tmp_path):
    """Lanceur tué net (tube cassé), puis arrêt demandé à l'étape : elle sauve malgré tout."""
    import os
    import signal
    import time
    proc, pid = demarrer(tmp_path)
    proc.kill()
    proc.wait(timeout=10)
    os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while not (tmp_path / "sauve").exists():
        assert time.time() - t0 < 20, "l'étape est morte sans sauver"
        time.sleep(0.05)
    assert (tmp_path / "sauve").read_text() == "SIGTERM"


def test_verrou_un_seul_lanceur(tmp_path):
    with L.Verrou(tmp_path / "v"):
        with pytest.raises(SystemExit, match="autre lanceur"):
            with L.Verrou(tmp_path / "v"):
                pass
    with L.Verrou(tmp_path / "v"):  # libéré à la sortie
        pass


# ---------------------------------------------------------------------------------------------
# mesurer et choix « auto »


def test_lire_benchmark_sortie_reelle_de_pretrain():
    texte = ("benchmark : 20 pas en 12.3 s = 1,234,567 tokens/s | budget 1,500,000,000 tokens "
             "(22889 pas) ≈ 0.3 h (hors évaluations) | précision bf16 | compile échec, repli sans "
             "compilation | mémoire GPU max 18.42 Go")
    m = L.lire_benchmark(texte)
    assert m["tokens_par_s"] == 1234567 and m["pas"] == 20 and m["memoire_gpu_max_go"] == 18.42
    assert m["precision"] == "bf16" and m["compile"] == "échec, repli sans compilation"
    assert L.lire_benchmark("rien") is None


def test_mesurer_essaie_un_micro_lot_plus_petit(tmp_path):
    rec = petite_recette(tmp_path)
    run = FauxExecutant(tps={64: 500_000})  # lot entier : mémoire insuffisante
    res = L.mesurer(rec, tmp_path / "sorties", "cuda", ["S"], run, root=tmp_path)
    s = res["tailles"]["S"]
    assert s["micro_lot"] == 64 and s["tokens_par_s"] == 500_000
    base_h = 1_500_000_000 / 500_000 / 3600
    assert s["etapes_heures"]["base"] == round(base_h, 2)
    total = (base_h + (90e6 + 150e6) / 500_000 / 3600) * 1.15 + 0.5
    assert s["total_heures"] == round(total, 2)
    assert (tmp_path / "sorties/mesures.json").exists()


def test_choisir_taille():
    m = {"tailles": {"S": {"total_heures": 2.0}, "M": {"total_heures": 3.5}}}
    assert L.choisir_taille(m, 4) == "M"
    assert L.choisir_taille(m, 3) == "S"
    assert L.choisir_taille(m, 1) is None
    assert L.choisir_taille({"tailles": {"S": {"erreur": "x"}}}, 10) is None


def test_lancer_auto_mesure_puis_choisit(tmp_path, capsys):
    rec = petite_recette(tmp_path)
    # S très rapide, M lent : avec 1 h, seule S tient
    def run(cmd, journal, root):
        if "--benchmark-steps" in cmd:
            fast = '"d_model":512' in " ".join(map(str, cmd))
            return FauxExecutant(tps={None: 10_000_000 if fast else 100_000})(cmd, journal, root)
        return FauxExecutant()(cmd, journal, root)
    assert lancer(tmp_path, rec, run, taille="auto", budget_heures=1) == 0
    assert json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())["taille"] == "S"
    assert "Taille choisie : S" in capsys.readouterr().out


def test_lancer_auto_rien_ne_tient(tmp_path):
    rec = petite_recette(tmp_path)
    (tmp_path / "sorties").mkdir()
    (tmp_path / "sorties/mesures.json").write_text(json.dumps(
        {"tailles": {"S": {"total_heures": 9}, "M": {"total_heures": 12}}}))
    assert lancer(tmp_path, rec, FauxExecutant(), taille="auto", budget_heures=4) == 2


# ---------------------------------------------------------------------------------------------
# Paquet, résultats, import


def test_verifier_manifeste_detecte_un_fichier_modifie(tmp_path):
    (tmp_path / "a.txt").write_text("bonjour")
    man = {"fichiers": {"a.txt": {"sha256": L.sha256_fichier(tmp_path / "a.txt"), "octets": 7},
                        "b.txt": {"sha256": "0", "octets": 1}}}
    (tmp_path / L.MANIFEST).write_text(json.dumps(man))
    _, err = L.verifier_manifeste(tmp_path, progress=False)
    assert err == ["manquant : b.txt"]
    (tmp_path / "a.txt").write_text("bonjouR")
    _, err = L.verifier_manifeste(tmp_path, progress=False)
    assert "contenu modifié (sha256) : a.txt" in err
    assert L.verifier_manifeste(tmp_path / "vide", progress=False)[0] is None


def test_archive_resultats_sans_donnees_et_import(tmp_path, monkeypatch):
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant()) == 0
    sortie = tmp_path / "sorties"
    (sortie / "runs/base/state.pt").write_bytes(b"gros")   # ne doit jamais partir
    (sortie / "runs/base/tokens_0.npy").write_bytes(b"cache")
    args = argparse.Namespace(sortie=str(sortie), autotest=False, partiel=False, avec_base=False,
                              archive=str(tmp_path / "R.zip"))
    assert L.cmd_resultats(args, rec) == 0
    with zipfile.ZipFile(tmp_path / "R.zip") as z:
        names = z.namelist()
    assert "RESULTATS_CHATBOT_RTX/modeles/chatbot_moliere/ckpt.pt" in names
    assert "RESULTATS_CHATBOT_RTX/evaluations/francais/metrics.json" in names
    assert "RESULTATS_CHATBOT_RTX/etat_lanceur.json" in names
    assert not any(n.endswith((".bin", "state.pt", ".npy", "best.pt")) for n in names)

    import tools.import_chatbot_results as I
    racine, man = I.extraire(tmp_path / "R.zip", tmp_path / "x")
    assert man["complet"] and not man["autotest"] and racine.name == "RESULTATS_CHATBOT_RTX"
    # Archive altérée : refusée
    with zipfile.ZipFile(tmp_path / "R.zip") as z, zipfile.ZipFile(tmp_path / "bad.zip", "w") as w:
        for n in z.namelist():
            data = z.read(n)
            w.writestr(n, data + b"!" if n.endswith("metrics.json") else data)
    with pytest.raises(ValueError, match="modifié"):
        I.extraire(tmp_path / "bad.zip", tmp_path / "y")


def test_resultats_refuse_un_lancement_pas_fini(tmp_path):
    rec = petite_recette(tmp_path)
    lancer(tmp_path, rec, FauxExecutant(codes={("chatbot.finetune", "francais_etape1"): [1]}))
    args = argparse.Namespace(sortie=str(tmp_path / "sorties"), autotest=False, partiel=False,
                              avec_base=False, archive=str(tmp_path / "R.zip"))
    assert L.cmd_resultats(args, rec) == 2
    args.partiel = True
    assert L.cmd_resultats(args, rec) == 0


def test_import_refuse_chemin_dangereux(tmp_path):
    import tools.import_chatbot_results as I
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="dangereux"):
        I.extraire(tmp_path / "evil.zip", tmp_path / "out")


# ---------------------------------------------------------------------------------------------
# Défauts restants du chantier précédent (un test par défaut)


def test_debordement_memoire_micro_lot_en_reprise(tmp_path, capsys):
    """Débordement mémoire en plein lancement : le message donne --micro-lot, et il est accepté
    en reprise (avant : le micro-lot de l'état était figé, la même erreur se reproduisait)."""
    rec = petite_recette(tmp_path)

    def oom(cmd, journal, root):
        if "chatbot.pretrain" in cmd:
            Path(journal).parent.mkdir(parents=True, exist_ok=True)
            Path(journal).write_text("torch.OutOfMemoryError: CUDA out of memory.\n", encoding="utf-8")
            return 1, False
        return FauxExecutant()(cmd, journal, root)
    assert lancer(tmp_path, rec, oom) == 1
    out = capsys.readouterr().out
    assert "mémoire de la carte insuffisante" in out and "--micro-lot 64" in out
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["base"]["statut"] == "echec" and etat["micro_lot"] is None
    run = FauxExecutant()
    assert lancer(tmp_path, rec, run, micro_lot=64) == 0
    base = next(c for c in run.appels if "chatbot.pretrain" in c)
    assert arg(base, "--micro-batch") == "64" and "--resume" in base
    assert json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())["micro_lot"] == 64
    assert lancer(tmp_path, rec, FauxExecutant(), micro_lot=48) == 2  # ne divise pas 128


def test_debordement_memoire_mesures_refaites(tmp_path, capsys):
    """Autre chemin conseillé : refaire `mesurer` ; un micro-lot plus petit est alors repris."""
    rec = petite_recette(tmp_path)
    lancer(tmp_path, rec, FauxExecutant(codes={("chatbot.pretrain", "base"): [3]}))
    (tmp_path / "sorties/mesures.json").write_text(json.dumps({"tailles": {"S": {"micro_lot": 32}}}))
    run = FauxExecutant()
    assert lancer(tmp_path, rec, run) == 0
    assert arg(next(c for c in run.appels if "chatbot.pretrain" in c), "--micro-batch") == "32"
    assert "micro-lot plus petit (32)" in capsys.readouterr().out


def test_seulement_ne_fait_pas_passer_un_lancement_inacheve_pour_complet(tmp_path):
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant(codes={("chatbot.finetune", "francais_etape1"): [1]})) == 1
    assert lancer(tmp_path, rec, FauxExecutant(), seulement="moliere", repartir=True) == 0
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert len(etat["plan"]) == 7 and "francais_etape1" in etat["plan"]
    args = argparse.Namespace(sortie=str(tmp_path / "sorties"), autotest=False, partiel=False,
                              avec_base=False, archive=str(tmp_path / "R.zip"))
    assert L.cmd_resultats(args, rec) == 2  # le chatbot français n'est pas fini
    args.partiel = True
    assert L.cmd_resultats(args, rec) == 0
    with zipfile.ZipFile(tmp_path / "R.zip") as z:
        man = json.loads(z.read("RESULTATS_CHATBOT_RTX/MANIFESTE_RESULTATS.json"))
    assert man["complet"] is False


def test_lanceur_tue_etape_orpheline_bloque_un_second_calcul(tmp_path):
    import subprocess
    import sys
    rec = petite_recette(tmp_path)
    journaux = tmp_path / "sorties/journaux"
    journaux.mkdir(parents=True)
    # Une « étape » encore vivante, dont le lanceur est mort (son fichier .pid est resté)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "chatbot.pretrain"])
    try:
        (journaux / "base.pid").write_text(str(proc.pid))
        if L.os.name != "nt":
            with pytest.raises(L.Refus, match=f"PID {proc.pid}"):
                lancer(tmp_path, rec, FauxExecutant())
            assert not (tmp_path / "sorties/etat_lanceur.json").exists()
    finally:
        proc.kill()
        proc.wait()
    assert lancer(tmp_path, rec, FauxExecutant()) == 0  # processus mort : fichier .pid effacé
    assert not (journaux / "base.pid").exists()


def test_executer_ecrit_puis_efface_le_fichier_pid(tmp_path):
    import sys
    code, stop = L.executer([sys.executable, "-c", "print('ok')"], tmp_path / "j/x.log", ROOT)
    assert code == 0 and not stop and not (tmp_path / "j/x.pid").exists()
    texte = (tmp_path / "j/x.log").read_text(encoding="utf-8")
    assert "ok" in texte and sys.executable not in texte and "===== " in texte  # pas de chemin personnel


def test_etape_tuee_de_l_exterieur_ne_conseille_pas_repartir(tmp_path, capsys):
    rec = petite_recette(tmp_path)
    run = FauxExecutant(codes={("chatbot.finetune", "moliere_etape1"): [-9]})  # kill, sans trace
    assert lancer(tmp_path, rec, run) == 3
    out = capsys.readouterr().out
    assert "arrêté de l'extérieur" in out and "relance la MÊME commande" in out
    assert "N'utilise PAS --repartir" in out
    etat = json.loads((tmp_path / "sorties/etat_lanceur.json").read_text())
    assert etat["etapes"]["moliere_etape1"]["statut"] == "interrompu"
    # Une vraie divergence garde, elle, le conseil de baisser le lr
    st, msg = L.diagnostic(1, _journal(tmp_path, "ARRÊT : loss nan / gradient nan non finis au pas 9"),
                           ["x", "--batch-size", "64"], "S")
    assert st == "echec" and "lr" in msg and "--repartir" in msg


def _journal(tmp_path, texte):
    j = tmp_path / "jd.log"
    j.write_text("\n===== date : cmd\n" + texte + "\n", encoding="utf-8")
    return j


def test_seulement_refuse_ne_fige_pas_la_taille(tmp_path):
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant(), taille="M", seulement="moliere") == 2
    assert not (tmp_path / "sorties/etat_lanceur.json").exists()
    assert lancer(tmp_path, rec, FauxExecutant(), taille="S") == 0


def test_import_archive_refusee_ne_laisse_aucun_reste(tmp_path, capsys):
    import tools.import_chatbot_results as I
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant()) == 0
    args = argparse.Namespace(sortie=str(tmp_path / "sorties"), autotest=False, partiel=False,
                              avec_base=False, archive=str(tmp_path / "R.zip"))
    assert L.cmd_resultats(args, rec) == 0
    with zipfile.ZipFile(tmp_path / "R.zip") as z, zipfile.ZipFile(tmp_path / "bad.zip", "w") as w:
        for n in z.namelist():
            w.writestr(n, z.read(n) + (b"!" if n.endswith("bilan.json") else b""))
    stages = tmp_path / "stages"
    with pytest.raises(SystemExit, match="archive refusée"):
        I.main([str(tmp_path / "bad.zip"), "--stages-dir", str(stages), "--runs-dir", str(tmp_path / "runs")])
    assert list(stages.iterdir()) == []


@pytest.mark.parametrize("module", ["chatbot.generate", "chatbot.evaluate"])
def test_console_utf8_pour_generate_et_evaluate(module):
    """Sortie redirigée en cp1252 (Windows) : le module repasse en UTF-8 au lieu de planter."""
    import os
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True,
                       env=dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONPATH=str(ROOT)))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-300:]
    assert "é".encode("utf-8") in r.stdout


# ---------------------------------------------------------------------------------------------
# Route git : faux gh (exécutable factice dans le PATH qui sert des fichiers locaux)


FAUX_GH = r'''import json, os, shutil, sys
from pathlib import Path
stock = Path(os.environ["FAUX_GH_STOCK"])
args = sys.argv[1:]
with open(stock / "appels.log", "a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\n")
def opt(n):
    return args[args.index(n) + 1] if n in args else None
def rel(depot, tag):
    return stock / depot.replace("/", "__") / tag
if args[:2] == ["auth", "status"]:
    sys.exit(0 if (stock / "connecte").exists() else 1)
if args[:2] == ["repo", "view"]:  # dépôt vide (aucun commit) si le marqueur « vide » existe
    print(json.dumps({"visibility": "PRIVATE", "isEmpty": (stock / "vide").exists()})); sys.exit(0)
if args[:2] == ["release", "view"]:
    d = rel(opt("--repo"), args[2])
    if not d.exists():
        print("release not found", file=sys.stderr); sys.exit(1)
    print(json.dumps({"assets": [{"name": p.name, "size": p.stat().st_size} for p in sorted(d.iterdir())]}))
    sys.exit(0)
if args[:2] == ["release", "download"]:
    src = rel(opt("--repo"), args[2]) / opt("--pattern")
    if not src.exists():
        print("no assets match the file pattern", file=sys.stderr); sys.exit(1)
    Path(opt("--dir")).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, Path(opt("--dir")) / src.name); sys.exit(0)
if args[:2] == ["release", "create"]:
    rel(opt("--repo"), args[2]).mkdir(parents=True); sys.exit(0)
if args[:2] == ["release", "upload"]:
    d, src = rel(opt("--repo"), args[2]), Path(args[3])
    if (d / src.name).exists() and "--clobber" not in args:
        print("asset already exists", file=sys.stderr); sys.exit(1)
    shutil.copyfile(src, d / src.name); sys.exit(0)
print("commande inconnue", args, file=sys.stderr); sys.exit(2)
'''


@pytest.fixture
def faux_gh(tmp_path, monkeypatch):
    """PATH = faux gh + outils système : le vrai gh ne peut JAMAIS être appelé ici."""
    import sys
    bin_dir, stock = tmp_path / "bin", tmp_path / "stock_gh"
    bin_dir.mkdir()
    stock.mkdir()
    (bin_dir / "faux_gh.py").write_text(FAUX_GH, encoding="utf-8")
    gh = bin_dir / "gh"  # petit script shell : le chemin de Python peut contenir des espaces
    gh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "faux_gh.py"}" "$@"\n', encoding="utf-8")
    gh.chmod(0o755)
    (stock / "connecte").write_text("oui")
    monkeypatch.setenv("PATH", f"{bin_dir}{L.os.pathsep}/usr/bin{L.os.pathsep}/bin")
    monkeypatch.setenv("FAUX_GH_STOCK", str(stock))
    return stock


def appels(stock, *debut):
    lignes = (stock / "appels.log").read_text(encoding="utf-8").splitlines() if (stock / "appels.log").exists() else []
    return [a for a in map(json.loads, lignes) if a[:len(debut)] == list(debut)]


def petite_release(tmp_path, stock, depot="moi/donnees", tag="d1"):
    """Deux fichiers servis par le faux gh et leur donnees_rtx.json."""
    rel = stock / depot.replace("/", "__") / tag
    rel.mkdir(parents=True)
    fichiers = []
    for nom, dest, contenu in [("pretrain_a.bin", "data/stages/x/pretrain/a.bin", b"A" * 1000),
                               ("dialogues_b.jsonl.zst", "data/stages/y/b.jsonl.zst", b"B" * 10)]:
        (rel / nom).write_bytes(contenu)
        fichiers.append({"piece_jointe": nom, "destination": dest, "octets": len(contenu),
                         "sha256": L.sha256_fichier(rel / nom)})
    don = {"depot": "À RENSEIGNER", "tag": "À RENSEIGNER", "fichiers": fichiers}
    (tmp_path / "donnees.json").write_text(json.dumps(don), encoding="utf-8")
    return tmp_path / "donnees.json", rel


def ns_tel(donnees, depot="moi/donnees", tag="d1"):
    return argparse.Namespace(donnees=str(donnees), depot=depot, tag=tag)


def test_telecharger_puis_reprise_sans_retelecharger(tmp_path, faux_gh, capsys):
    donnees, _ = petite_release(tmp_path, faux_gh)
    clone = tmp_path / "clone"
    clone.mkdir()
    assert L.cmd_telecharger(ns_tel(donnees), REAL, root=clone) == 0
    assert (clone / "data/stages/x/pretrain/a.bin").read_bytes() == b"A" * 1000
    assert not (clone / "data/stages/x/pretrain/.telechargement_rtx").exists()
    assert len(appels(faux_gh, "release", "download")) == 2
    assert "TÉLÉCHARGEMENT RÉUSSI" in capsys.readouterr().out
    # Reprise : un fichier faux (coupure) est retéléchargé, les bons sont sautés
    (clone / "data/stages/y/b.jsonl.zst").write_bytes(b"coupe")
    assert L.cmd_telecharger(ns_tel(donnees), REAL, root=clone) == 0
    dl = appels(faux_gh, "release", "download")
    assert len(dl) == 3 and dl[-1][dl[-1].index("--pattern") + 1] == "dialogues_b.jsonl.zst"
    assert "1 déjà bons" in capsys.readouterr().out
    assert L.cmd_telecharger(ns_tel(donnees), REAL, root=clone) == 0
    assert len(appels(faux_gh, "release", "download")) == 3  # tout est bon : rien de téléchargé


def test_telecharger_refuse_un_sha256_faux(tmp_path, faux_gh):
    donnees, rel = petite_release(tmp_path, faux_gh)
    (rel / "pretrain_a.bin").write_bytes(b"Z" * 1000)  # même taille, autre contenu
    clone = tmp_path / "clone"
    with pytest.raises(L.Refus, match="sha256"):
        L.cmd_telecharger(ns_tel(donnees), REAL, root=clone)
    assert not (clone / "data/stages/x/pretrain/a.bin").exists()  # jamais mis en place
    assert not (clone / "data/stages/x/pretrain/.telechargement_rtx").exists()


def test_telecharger_gh_absent_ou_non_connecte(tmp_path, faux_gh, monkeypatch):
    donnees, _ = petite_release(tmp_path, faux_gh)
    (faux_gh / "connecte").unlink()
    with pytest.raises(L.Refus, match="gh auth login"):
        L.cmd_telecharger(ns_tel(donnees), REAL, root=tmp_path / "clone")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(L.Refus, match="introuvable"):
        L.cmd_telecharger(ns_tel(donnees), REAL, root=tmp_path / "clone")


def test_telecharger_depot_a_renseigner_refuse(tmp_path, faux_gh):
    donnees, _ = petite_release(tmp_path, faux_gh)
    with pytest.raises(L.Refus, match="non renseignés"):
        L.cmd_telecharger(ns_tel(donnees, depot=None, tag=None), REAL, root=tmp_path / "clone")
    assert appels(faux_gh) == []


def test_main_renvoie_2_sur_refus(tmp_path, faux_gh, capsys, monkeypatch):
    donnees, _ = petite_release(tmp_path, faux_gh)
    monkeypatch.setattr(L, "installer_arrets", lambda: None)
    assert L.main(["--sortie", str(tmp_path / "s"), "--donnees", str(donnees), "telecharger"]) == 2
    assert "ERREUR : dépôt ou étiquette non renseignés" in capsys.readouterr().out


def test_creer_dossier_traverse_les_liens_sans_les_remplacer(tmp_path):
    (tmp_path / "vrai.nosync").mkdir()
    (tmp_path / "stages").symlink_to(tmp_path / "vrai.nosync")
    L.creer_dossier(tmp_path / "stages/a/b")
    assert (tmp_path / "stages").is_symlink() and (tmp_path / "vrai.nosync/a/b").is_dir()
    (tmp_path / "casse").symlink_to(tmp_path / "absent")
    with pytest.raises(L.Refus, match="lien cassé"):
        L.creer_dossier(tmp_path / "casse/x")
    assert (tmp_path / "casse").is_symlink()  # pas remplacé
    L.creer_dossier(tmp_path / "neuf/runs/chatbot_rtx")  # clone neuf : vrais dossiers
    assert not (tmp_path / "neuf/runs").is_symlink()


def test_verifier_clone_controle_les_gros_fichiers(tmp_path, faux_gh):
    donnees, _ = petite_release(tmp_path, faux_gh)
    clone = tmp_path / "clone"
    clone.mkdir()
    n, err = L.verifier_donnees(json.loads(donnees.read_text()), clone, progress=False)
    assert n == 2 and len(err) == 2 and all("manquant" in e for e in err)
    L.cmd_telecharger(ns_tel(donnees), REAL, root=clone)
    assert L.verifier_donnees(json.loads(donnees.read_text()), clone, progress=False)[1] == []
    (clone / "data/stages/y/b.jsonl.zst").write_bytes(b"C" * 10)
    assert "contenu modifié" in L.verifier_donnees(json.loads(donnees.read_text()), clone, progress=False)[1][0]
    with pytest.raises(L.Refus, match="chemin refusé"):
        L.chemin_sur("../evil")


def test_donnees_rtx_json_du_depot():
    """La liste suivie par git : noms uniques sans accent, chemins des recettes, < 2 Gio."""
    don = json.loads((ROOT / "chatbot/donnees_rtx.json").read_text(encoding="utf-8"))
    assert L.A_RENSEIGNER in don["depot"] and L.A_RENSEIGNER in don["tag"]
    noms = [f["piece_jointe"] for f in don["fichiers"]]
    assert len(set(noms)) == len(noms) and all(L.NOM_SUR.match(n) for n in noms)
    dests = {f["destination"] for f in don["fichiers"]}
    meta = f"{L.PRETRAIN}/meta.json"
    attendus = {meta, *L.DONNEES}
    attendus |= {f"{L.PRETRAIN}/{n}.bin" for n in L.val(REAL["base"]["parts"])}
    for cb in L.CHATBOTS:
        c = REAL["chatbots"][cb]
        attendus |= set(L.val(c["etape1"]["donnees"])) | {L.val(c["validation"])}
    assert attendus <= dests
    assert all(f["octets"] < 2 * 1024 ** 3 and len(f["sha256"]) == 64 for f in don["fichiers"])
    assert don["total_octets"] == sum(f["octets"] for f in don["fichiers"])
    for f in don["fichiers"]:  # sur le Mac : tailles conformes aux vrais fichiers
        p = ROOT / f["destination"]
        if p.exists():
            assert p.stat().st_size == f["octets"], f["destination"]


def _lancement_fini(tmp_path):
    rec = petite_recette(tmp_path)
    assert lancer(tmp_path, rec, FauxExecutant()) == 0
    args = argparse.Namespace(sortie=str(tmp_path / "sorties"), autotest=False, partiel=False,
                              avec_base=False, archive=None)
    assert L.cmd_resultats(args, rec, root=tmp_path) == 0
    return rec, tmp_path / "sorties/RESULTATS_CHATBOT_RTX.zip"


def ns_pub(tmp_path, publier=False, depot="moi/resultats", tag="r1"):
    return argparse.Namespace(sortie=str(tmp_path / "sorties"), autotest=False, archive=None,
                              depot=depot, tag=tag, publier=publier)


def test_publier_a_blanc_par_defaut_n_envoie_rien(tmp_path, faux_gh, capsys):
    rec, archive = _lancement_fini(tmp_path)  # clone : archive dans le dossier de sortie
    assert archive.exists()
    assert L.cmd_publier(ns_pub(tmp_path), rec, root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "À BLANC" in out and "visibilité : PRIVATE" in out and "gh release upload r1" in out
    assert appels(faux_gh, "release", "create") == [] and appels(faux_gh, "release", "upload") == []
    assert not (faux_gh / "moi__resultats").exists()
    exp = list((tmp_path / "experiments/chatbot_rtx").glob("*_S"))
    assert len(exp) == 1
    copies = {p.relative_to(exp[0]).as_posix() for p in exp[0].rglob("*") if p.is_file()}
    assert {"bilan.json", "etat_lanceur.json", "entrainements/base/report.json",
            "evaluations/moliere/metrics.json"} <= copies
    assert not any(c.startswith("runs/") for c in copies)  # « runs » est ignoré par .gitignore
    assert not any(c.endswith((".pt", ".log", ".zip")) for c in copies)  # ni poids ni journaux
    import sys
    assert not any(sys.executable in p.read_text(encoding="utf-8") for p in exp[0].rglob("*.json"))


def test_publier_decoupe_envoie_reprend_puis_import_depuis_release(tmp_path, faux_gh, monkeypatch, capsys):
    import tools.import_chatbot_results as I
    rec, archive = _lancement_fini(tmp_path)
    monkeypatch.setattr(L, "LIMITE_PIECE", 700)  # force le découpage en morceaux
    with pytest.raises(L.Refus, match="non renseignés"):
        L.cmd_publier(ns_pub(tmp_path, publier=True, depot=None), rec, root=tmp_path)
    assert L.cmd_publier(ns_pub(tmp_path, publier=True), rec, root=tmp_path) == 0
    rel = faux_gh / "moi__resultats/r1"
    noms = sorted(p.name for p in rel.iterdir())
    morceaux = [n for n in noms if ".zip.0" in n]
    assert len(morceaux) == -(-archive.stat().st_size // 700) and any(n.endswith(".manifeste.json") for n in noms)
    assert all((rel / n).stat().st_size <= 700 for n in morceaux)
    assert "--latest=false" in appels(faux_gh, "release", "create")[0]
    n_envois = len(appels(faux_gh, "release", "upload"))
    # Reprise : relancer n'envoie rien de plus (tout est déjà en ligne, même taille)
    assert L.cmd_publier(ns_pub(tmp_path, publier=True), rec, root=tmp_path) == 0
    assert len(appels(faux_gh, "release", "upload")) == n_envois
    assert "déjà en ligne" in capsys.readouterr().out
    # Mac : reconstitution vérifiée, identique à l'archive du PC
    rebuilt = L.reconstituer_depuis_release(L.gh_pret(), "moi/resultats", "r1", tmp_path / "recu")
    assert L.sha256_fichier(rebuilt) == L.sha256_fichier(archive)
    # Import complet depuis la release (sans faire parler les faux modèles)
    monkeypatch.setattr(I, "faire_parler", lambda d, f: [{"message": "m", "reponse": "r"}])
    stages = tmp_path / "stages"
    I.main(["--depuis-release", "moi/resultats", "r1", "--stages-dir", str(stages),
            "--runs-dir", str(tmp_path / "runs_mac"), "--nom", "rtx_test"])
    assert (stages / "rtx_test/preuve_import.json").exists()
    assert [p.name for p in stages.iterdir()] == ["rtx_test"]  # aucun reste de téléchargement
    # Morceau abîmé dans la release : refusé
    (rel / morceaux[0]).write_bytes(b"x" * (rel / morceaux[0]).stat().st_size)
    with pytest.raises(L.Refus, match="sha256"):
        L.reconstituer_depuis_release(L.gh_pret(), "moi/resultats", "r1", tmp_path / "recu2")


def test_publish_release_donnees_a_blanc_puis_envoi_factice(tmp_path, faux_gh, capsys, monkeypatch):
    """Outil Mac des données : à blanc par défaut ; avec --publier (faux gh) : envoi, reprise,
    tailles contrôlées, dépôt et étiquette écrits dans la liste."""
    import tools.publish_chatbot_release as P
    racine = tmp_path / "mac"
    pre = racine / L.PRETRAIN
    pre.mkdir(parents=True)
    (pre / "a.bin").write_bytes(b"\1" * 100)
    (pre / "meta.json").write_text(json.dumps({"sources": {"a": {"file": "a.bin"}}}))
    for d in L.DONNEES:
        (racine / d).parent.mkdir(parents=True, exist_ok=True)
        (racine / d).write_bytes(b"z" * 7)
    liste = tmp_path / "donnees.json"
    assert P.main(["--racine", str(racine), "--json", str(liste)]) == 0
    don = json.loads(liste.read_text())
    assert len(don["fichiers"]) == 2 + len(L.DONNEES) and L.A_RENSEIGNER in don["depot"]
    assert "À BLANC" in capsys.readouterr().out and appels(faux_gh, "release") == []
    with pytest.raises(SystemExit, match="non renseignés"):
        P.main(["--racine", str(racine), "--json", str(liste), "--publier"])
    assert P.main(["--racine", str(racine), "--json", str(liste), "--depot", "moi/d", "--tag", "t",
                   "--publier"]) == 0
    assert len(list((faux_gh / "moi__d/t").iterdir())) == len(don["fichiers"])
    don2 = json.loads(liste.read_text())
    assert (don2["depot"], don2["tag"]) == ("moi/d", "t") and don2["fichiers"] == don["fichiers"]
    assert not (racine / P.ENVOI).exists()
    assert (pre / "a.bin").read_bytes() == b"\1" * 100  # données intactes (liens physiques retirés)
    # Le PC télécharge exactement ces fichiers avec cette liste
    clone = tmp_path / "clone"
    monkeypatch.setattr(L, "assurer_donnees_essai", lambda root: None)  # faux fichiers, pas du zstd
    assert L.cmd_telecharger(ns_tel(liste, depot=None, tag=None), REAL, root=clone) == 0
    assert (clone / L.PRETRAIN / "a.bin").read_bytes() == b"\1" * 100


def test_publish_release_refuse_un_fichier_de_2_gio(tmp_path, monkeypatch):
    import tools.publish_chatbot_release as P
    racine = tmp_path / "mac"
    pre = racine / L.PRETRAIN
    pre.mkdir(parents=True)
    (pre / "a.bin").write_bytes(b"\1" * 100)
    (pre / "meta.json").write_text(json.dumps({"sources": {"a": {"file": "a.bin"}}}))
    for d in L.DONNEES:
        (racine / d).parent.mkdir(parents=True, exist_ok=True)
        (racine / d).write_bytes(b"z")
    monkeypatch.setattr(P, "MAX_OCTETS", 100)
    with pytest.raises(SystemExit, match="2 Gio"):
        P.regenerer(racine)


def test_rapports_copies_ne_sont_pas_ignores_par_git():
    """Les chemins écrits par `publier` dans experiments/ restent visibles pour git (commitables),
    alors que runs/chatbot_rtx et donnees_essai/ ne le sont pas."""
    import shutil
    import subprocess
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("pas de dépôt git")

    def ignore(p):
        return subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode == 0
    base = "experiments/chatbot_rtx/20261114_0312_S/"
    for k in ("bilan.json", "entrainements/base/report.json", "entrainements/base/log.jsonl",
              "evaluations/moliere/metrics.json", "modeles/chatbot_moliere/training_report.json",
              "lanceur/mesures.json"):
        assert not ignore(base + k), k
    assert ignore("runs") and ignore("data/stages")  # (sur le Mac ce sont des liens : on teste le lien)
    assert not ignore("chatbot/donnees_rtx.json") and not ignore("PC_RTX.md")


# ---------------------------------------------------------------------------------------------
# Corrections des vérificateurs (route git)


def test_installer_refuse_une_archive_de_la_recette_d_essai(tmp_path, monkeypatch):
    """Archive complète mais en taille « essai » (modèles minuscules) : jamais installée à la place
    des modèles de la démo ; importée seulement pour examen, rangée sous essai_ (pas rtx_)."""
    import tools.import_chatbot_results as I
    monkeypatch.setattr(I, "faire_parler", lambda d, f: [{"message": "m", "reponse": "r"}])
    rec, archive = _lancement_fini(tmp_path)
    essai = tmp_path / "essai.zip"
    with zipfile.ZipFile(archive) as z, zipfile.ZipFile(essai, "w") as w:
        for n in z.namelist():
            data = z.read(n)
            if n.endswith("MANIFESTE_RESULTATS.json"):
                man = json.loads(data)
                assert man["complet"] and not man["autotest"]
                data = json.dumps(dict(man, taille="essai")).encode()
            w.writestr(n, data)
    models, stages, runs = tmp_path / "models", tmp_path / "stages", tmp_path / "runs_mac"
    (models / "chatbot_moliere").mkdir(parents=True)
    (models / "chatbot_moliere/ckpt.pt").write_bytes(b"demo")
    commun = ["--stages-dir", str(stages), "--runs-dir", str(runs), "--models-dir", str(models)]
    with pytest.raises(SystemExit, match="recette d'essai"):
        I.main([str(essai), "--installer", *commun])
    assert list(stages.iterdir()) == []  # refus avant tout rangement
    assert [p.name for p in models.iterdir()] == ["chatbot_moliere"]
    assert (models / "chatbot_moliere/ckpt.pt").read_bytes() == b"demo"  # démo intacte
    I.main([str(essai), *commun])
    assert [p.name.split("_")[0] for p in stages.iterdir()] == ["essai"]
    # Une archive complète en taille S, elle, s'installe (l'ancien modèle est gardé, daté)
    I.main([str(archive), "--installer", "--nom", "rtx_ok", *commun])
    assert (models / "chatbot_moliere/ckpt.pt").read_bytes() == b"x"
    assert any(p.name.startswith("chatbot_moliere.avant_") for p in models.iterdir())


def test_sortie_acceptee_avant_ou_apres_la_sous_commande(tmp_path, capsys):
    p = L.build_parser()
    assert p.parse_args(["--sortie", "A", "lancer", "--taille", "S"]).sortie == "A"
    assert p.parse_args(["lancer", "--taille", "S", "--sortie", "B"]).sortie == "B"
    assert p.parse_args(["--sortie", "A", "etat"]).sortie == "A"  # non écrasée par la sous-commande
    assert p.parse_args(["etat"]).sortie is None
    # Les conseils donnent la commande complète, --sortie AVANT la sous-commande
    assert "--sortie runs/chatbot_rtx_2 lancer --taille" in L.MESSAGES["divergence"]
    rec = petite_recette(tmp_path)
    lancer(tmp_path, rec, FauxExecutant(codes={("chatbot.pretrain", "base"): [3]}))
    capsys.readouterr()
    assert lancer(tmp_path, rec, FauxExecutant(), taille="M") == 2
    assert "--sortie runs/chatbot_rtx_2 lancer --taille M" in capsys.readouterr().out
    doc = (ROOT / "PC_RTX.md").read_text(encoding="utf-8")
    assert "--sortie runs\\chatbot_rtx_2 lancer --taille" in doc


def test_depot_vide_refuse_avant_tout_envoi(tmp_path, faux_gh, capsys):
    """Dépôt créé sans commit : GitHub refuserait la release. Le plan à blanc prévient, l'envoi
    refuse AVANT de créer la release ; après un premier commit, l'envoi passe."""
    rec, archive = _lancement_fini(tmp_path)
    (faux_gh / "vide").write_text("oui")
    assert L.cmd_publier(ns_pub(tmp_path), rec, root=tmp_path) == 0
    assert "VIDE (aucun commit)" in capsys.readouterr().out
    with pytest.raises(L.Refus, match="--add-readme"):
        L.cmd_publier(ns_pub(tmp_path, publier=True), rec, root=tmp_path)
    assert appels(faux_gh, "release", "create") == [] and appels(faux_gh, "release", "upload") == []
    assert not (tmp_path / "sorties/envoi_release").exists()  # aucune copie de morceau laissée
    (faux_gh / "vide").unlink()  # l'étudiant a mis un premier commit
    assert L.cmd_publier(ns_pub(tmp_path, publier=True), rec, root=tmp_path) == 0
    assert len(appels(faux_gh, "release", "create")) == 1
    guide = (ROOT / "docs/remise/PASSAGE_PC_RTX5090_CHATBOT.md").read_text(encoding="utf-8")
    assert "--private --add-readme" in guide
