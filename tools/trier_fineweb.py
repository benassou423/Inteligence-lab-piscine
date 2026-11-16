"""Tri appris de FineWeb : un modèle à poids ouverts NOTE des documents réels, un petit modèle linéaire
apprend ces notes, et nettoyer_pretrain.py s'en sert pour écarter le spam et le charabia. Deux
apprenants existent : un classifieur de mots (« apprendre ») et une sonde linéaire sur les
activations du modèle de base (« sonde »). C'est la sonde qui a été retenue (règles publiées :
chatbot/regles_pretrain_v4.json ; sonde : chatbot/sonde_fineweb.npz).

Aucun texte n'est généré : seulement des notes. Consigne publiée : chatbot/prompts_tri_fineweb.json.

RÉSULTAT DE L'ESSAI (14/11/2026) : NÉGATIF. Le tri écarte bien le spam (voir le rapport à côté de
chatbot/sonde_fineweb.npz), mais le modèle pré-entraîné sur les données triées ne répond pas mieux
(voir l'en-tête de tools/nettoyer_pretrain.py). Le modèle publié n'utilise PAS ce tri.

  tirer    : N documents de fineweb.bin au prorata des tokens (hors documents déjà tirés)
  noter    : note chaque extrait avec qwen3:8b via Ollama (reprise possible)
  apprendre: classifieur (mots et paires de mots hachés + traits simples), validation croisée,
             puis mesure sur des notes de contrôle indépendantes (non retenu)
  sonde    : sonde linéaire sur les activations d'un bloc du modèle de base (--couche, 5 par défaut ;
             --base : best.pt du modèle de base), validation croisée, puis mesure sur des notes de
             contrôle indépendantes ; c'est elle qui est retenue (--out : chatbot/sonde_fineweb.npz)
"""
import argparse
import json
import random
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import nettoyer_pretrain as NP  # noqa: E402

PROMPTS = ROOT / "chatbot/prompts_tri_fineweb.json"
DIM = 1 << 18
NOTES = {"propre": 0, "defaut_leger": 1, "defaut_grave": 2}


def extrait(t, debut=900, milieu=600):
    if len(t) <= debut + milieu:
        return t
    m = len(t) // 2
    return t[:debut] + "\n[…coupé…]\n" + t[m:m + milieu]


def indices(texte):
    """Cases (hachage stable crc32) des mots et des paires de mots de l'extrait, sans doublon."""
    w = [x.lower() for x in NP.MOT.findall(texte)]
    cases = {zlib.crc32(x.encode()) % DIM for x in w}
    cases.update(zlib.crc32((a + " " + b).encode()) % DIM for a, b in zip(w, w[1:]))
    return np.fromiter(cases, dtype=np.int64, count=len(cases))


class Classifieur:
    """logit = b + somme(w[cases]) / racine(nombre de cases) + v · traits normalisés."""

    def __init__(self, chemin):
        z = np.load(chemin)
        assert list(z["noms"]) == NP.NOMS and int(z["dim"]) == DIM, "classifieur appris avec d'autres traits"
        self.w, self.v, self.b = z["w"], z["v"], float(z["b"])
        self.moy, self.ect = z["moy"], z["ect"]

    def score(self, texte, traits):
        idx = indices(extrait(texte))
        x = (np.asarray(traits, dtype=np.float32) - self.moy) / self.ect
        return float(self.b + self.w[idx].sum() / max(1.0, np.sqrt(len(idx))) + float(self.v @ x))


def cmd_tirer(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(NP.TOKENIZER))
    arr = np.memmap(NP.SRC / "fineweb.bin", dtype=np.uint16, mode="r")
    u = NP.unites(arr, 0)
    deja = set()
    for f in a.hors or []:
        deja.update(x["debut"] for x in json.loads(Path(f).read_text(encoding="utf-8")))
    rng = random.Random(a.graine)
    pris, fiches = set(), []
    while len(fiches) < a.n:
        p = rng.randrange(len(arr))
        i = int(np.searchsorted(u[:, 0], p, side="right") - 1)
        d, f = int(u[i, 0]), int(u[i, 1])
        if i in pris or d in deja or f - d < 60:
            continue
        pris.add(i)
        fiches.append({"numero": len(fiches) + 1, "debut": d, "fin": f, "tokens": f - d - 1,
                       "texte": tok.decode(arr[d + 1:f].tolist())})
    Path(a.out).write_text(json.dumps(fiches, ensure_ascii=False), encoding="utf-8")
    print(len(fiches), "documents ->", a.out)


def noter_un(texte, cfg, port):
    g = cfg["generateur"]
    corps = json.dumps({"model": g["model"], "prompt": cfg["consigne"] + texte, "stream": False, "think": g["think"],
                        "format": cfg["format"],
                        "options": {"temperature": g["temperature"], "num_predict": g["num_predict"], "seed": g["seed"]}}).encode()
    for essai in range(4):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate", data=corps, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                rep = json.loads(r.read())["response"]
            return json.loads(rep)["note"]
        except Exception as e:  # réponse mal formée ou serveur occupé : on réessaie
            err = repr(e)
            time.sleep(1 + essai)
    return "erreur:" + err[:80]


def cmd_noter(a):
    cfg = json.loads(PROMPTS.read_text(encoding="utf-8"))
    fiches = json.loads(Path(a.fiches).read_text(encoding="utf-8"))
    sortie = Path(a.out)
    notes = {x["numero"]: x["note"] for x in json.loads(sortie.read_text(encoding="utf-8"))} if sortie.exists() else {}
    a_faire = [f for f in fiches if notes.get(f["numero"]) not in NOTES][:a.limite or None]
    print(f"{len(fiches)} fiches, {len(a_faire)} à noter, ports {a.ports}, {a.par_port} appels par port", flush=True)
    t0 = time.time()

    def un(k_f):
        k, f = k_f
        return f["numero"], noter_un(extrait(f["texte"], cfg["extrait"]["debut"], cfg["extrait"]["milieu"]), cfg, a.ports[k % len(a.ports)])

    def sauver():
        tmp = Path(str(sortie) + ".tmp")
        tmp.write_text(json.dumps([{"numero": n, "note": v} for n, v in sorted(notes.items())], ensure_ascii=False), encoding="utf-8")
        tmp.replace(sortie)
    with ThreadPoolExecutor(len(a.ports) * a.par_port) as ex:
        for k, (n, v) in enumerate(ex.map(un, enumerate(a_faire)), 1):
            notes[n] = v
            if k % 200 == 0:
                sauver()
                print(f"  {k} / {len(a_faire)} ({time.time() - t0:.0f} s, {k / (time.time() - t0):.2f} doc/s)", flush=True)
    sauver()
    compte = {}
    for v in notes.values():
        compte[v if v in NOTES else "erreur"] = compte.get(v if v in NOTES else "erreur", 0) + 1
    print("notes :", compte, f"({time.time() - t0:.0f} s)")


def matrice(fiches):
    idx = [indices(extrait(f["texte"])) for f in fiches]
    T = np.array([NP.traits(f["texte"]) for f in fiches], dtype=np.float32)
    return idx, T


def entrainer(idx, T, y, moy, ect, pas=300, l2=3e-4, device="cuda"):
    import torch
    torch.manual_seed(0)
    lignes = np.concatenate([np.full(len(i), r) for r, i in enumerate(idx)])
    cols = np.concatenate(idx)
    vals = np.concatenate([np.full(len(i), 1.0 / max(1.0, np.sqrt(len(i))), dtype=np.float32) for i in idx])
    X = torch.sparse_coo_tensor(np.stack([lignes, cols]), vals, (len(idx), DIM)).coalesce().to(device)
    Z = torch.tensor((T - moy) / ect, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(DIM, device=device, requires_grad=True)
    v = torch.zeros(T.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, v, b], lr=0.05)
    pos = yt.mean().clamp(0.02, 0.98)
    poids = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))  # classes équilibrées
    for _ in range(pas):
        opt.zero_grad()
        logit = torch.sparse.mm(X, w[:, None])[:, 0] + Z @ v + b
        perte = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, weight=poids)
                 + l2 * (w.square().sum() + v.square().sum()))
        perte.backward()
        opt.step()
    return w.detach().cpu().numpy(), v.detach().cpu().numpy(), float(b.item())


def scores(idx, T, w, v, b, moy, ect):
    Z = (T - moy) / ect
    return np.array([b + w[i].sum() / max(1.0, np.sqrt(len(i))) for i in idx], dtype=np.float32) + Z @ v


def auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def cmd_apprendre(a):
    fiches = {f["numero"]: f for f in json.loads(Path(a.fiches).read_text(encoding="utf-8"))}
    notes = {x["numero"]: x["note"] for x in json.loads(Path(a.notes).read_text(encoding="utf-8")) if x["note"] in NOTES}
    nums = sorted(notes)
    F = [fiches[n] for n in nums]
    y = np.array([notes[n] == "defaut_grave" for n in nums])
    print(f"{len(nums)} documents notés, {int(y.sum())} « defaut_grave » ({100 * y.mean():.1f} %)")
    idx, T = matrice(F)
    moy, ect = T.mean(0), T.std(0) + 1e-6
    rng = np.random.default_rng(0)
    pli = rng.integers(0, 5, len(nums))
    s_cv = np.zeros(len(nums), dtype=np.float32)
    for k in range(5):
        tr, te = pli != k, pli == k
        w, v, b = entrainer([i for i, m in zip(idx, tr) if m], T[tr], y[tr], moy, ect, device=a.device)
        s_cv[te] = scores([i for i, m in zip(idx, te) if m], T[te], w, v, b, moy, ect)
    rapport = {"documents": len(nums), "graves": int(y.sum()), "auc_validation_croisee": round(auc(s_cv, y), 4)}
    w, v, b = entrainer(idx, T, y, moy, ect, device=a.device)
    np.savez(a.out, w=w.astype(np.float32), v=v.astype(np.float32), b=np.float32(b), moy=moy, ect=ect,
             noms=np.array(NP.NOMS), dim=DIM)
    rapport["notes_sha256"] = __import__("hashlib").sha256(Path(a.notes).read_bytes()).hexdigest()
    if a.controle_fiches:
        cf = {f["numero"]: f for f in json.loads(Path(a.controle_fiches).read_text(encoding="utf-8"))}
        cn = {d["numero"]: d["qualite"] for d in json.loads(Path(a.controle_notes).read_text(encoding="utf-8"))["fineweb"]}
        cnums = sorted(cn)
        ci, cT = matrice([cf[n] for n in cnums])
        cs = scores(ci, cT, w, v, b, moy, ect)
        cy = np.array([cn[n] == "defaut_grave" for n in cnums])
        leger = np.array([cn[n] == "defaut_leger" for n in cnums])
        rapport["controle"] = {"documents": len(cnums), "graves": int(cy.sum()), "auc": round(auc(cs, cy), 4), "seuils": []}
        for q in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            seuil = float(np.quantile(cs, 1 - q))
            ecarte = cs > seuil
            rapport["controle"]["seuils"].append({
                "part_ecartee": q, "seuil": round(seuil, 4), "ecartes": int(ecarte.sum()),
                "graves_ecartes": int((ecarte & cy).sum()), "legers_ecartes": int((ecarte & leger).sum()),
                "propres_ecartes": int((ecarte & ~cy & ~leger).sum()),
                "graves_restants_parmi_gardes": round(float((cy & ~ecarte).sum() / max(1, (~ecarte).sum())), 4)})
        np.save(str(a.out) + ".scores_controle.npy", cs)
    Path(str(a.out) + ".rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rapport, ensure_ascii=False, indent=1))


def table_controle(cs, cn, cnums):
    cy = np.array([cn[n] == "defaut_grave" for n in cnums])
    leger = np.array([cn[n] == "defaut_leger" for n in cnums])
    out = {"documents": len(cnums), "graves": int(cy.sum()), "legers": int(leger.sum()), "auc": round(auc(cs, cy), 4), "seuils": []}
    for q in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        seuil = float(np.quantile(cs, 1 - q))
        e = cs > seuil
        out["seuils"].append({"part_ecartee": q, "seuil": round(seuil, 4), "ecartes": int(e.sum()), "graves_ecartes": int((e & cy).sum()),
                              "legers_ecartes": int((e & leger).sum()), "propres_ecartes": int((e & ~cy & ~leger).sum()),
                              "graves_restants_parmi_gardes": round(float((cy & ~e).sum() / max(1, (~e).sum())), 4)})
    return out


def cmd_sonde(a):
    """Sonde linéaire : « defaut_grave » prédit depuis la moyenne des activations d'un bloc du modèle de base."""
    import hashlib
    arr = np.memmap(NP.SRC / "fineweb.bin", dtype=np.uint16, mode="r")
    fiches = json.loads(Path(a.fiches).read_text(encoding="utf-8"))
    notes = {x["numero"]: x["note"] for x in json.loads(Path(a.notes).read_text(encoding="utf-8")) if x["note"] in NOTES}
    F = [f for f in fiches if f["numero"] in notes]
    y = np.array([notes[f["numero"]] == "defaut_grave" for f in F])
    X = NP.activations(arr, np.array([(f["debut"], f["fin"]) for f in F], dtype=np.int64), a.base, a.couche, a.device)
    moy, ect = X.mean(0), X.std(0) + 1e-6
    vide = lambda n: [np.zeros(0, dtype=np.int64)] * n
    pli = np.random.default_rng(0).integers(0, 5, len(F))
    s_cv = np.zeros(len(F), dtype=np.float32)
    for k in range(5):
        tr, te = pli != k, pli == k
        _, v, b = entrainer(vide(int(tr.sum())), X[tr], y[tr], moy, ect, device=a.device)
        s_cv[te] = ((X[te] - moy) / ect) @ v + b
    _, v, b = entrainer(vide(len(F)), X, y, moy, ect, device=a.device)
    rapport = {"documents": len(F), "graves": int(y.sum()), "couche": a.couche, "auc_validation_croisee": round(auc(s_cv, y), 4),
               "notes_sha256": hashlib.sha256(Path(a.notes).read_bytes()).hexdigest(),
               "base_sha256": hashlib.sha256(Path(a.base).read_bytes()).hexdigest()}
    np.savez(a.out, v=v.astype(np.float32), b=np.float32(b), moy=moy, ect=ect, couche=a.couche, base_sha256=rapport["base_sha256"])
    if a.controle_fiches:
        cf = {f["numero"]: f for f in json.loads(Path(a.controle_fiches).read_text(encoding="utf-8"))}
        cn = {d["numero"]: d["qualite"] for d in json.loads(Path(a.controle_notes).read_text(encoding="utf-8"))["fineweb"]}
        cnums = sorted(cn)
        Xc = NP.activations(arr, np.array([(cf[n]["debut"], cf[n]["fin"]) for n in cnums], dtype=np.int64), a.base, a.couche, a.device)
        rapport["controle"] = table_controle(((Xc - moy) / ect) @ v + b, cn, cnums)
    Path(str(a.out) + ".rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    c = rapport.get("controle", {})
    print({k: (v if "sha" not in k else v[:12]) for k, v in rapport.items() if k != "controle"})
    print("contrôle : AUC", c.get("auc"), "| graves", c.get("graves"), "| légers", c.get("legers"))
    for x in c.get("seuils", []):
        print("  ", x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    t = s.add_parser("tirer")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--graine", type=int, default=20261116)
    t.add_argument("--hors", nargs="*", help="fiches déjà tirées, à ne pas reprendre")
    n = s.add_parser("noter")
    n.add_argument("--fiches", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--ports", type=int, nargs="+", default=[11434])
    n.add_argument("--par-port", type=int, default=3)
    n.add_argument("--limite", type=int, default=0)
    c = s.add_parser("apprendre")
    c.add_argument("--fiches", required=True)
    c.add_argument("--notes", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--controle-fiches", default=None)
    c.add_argument("--controle-notes", default=None)
    c.add_argument("--device", default="cuda")
    so = s.add_parser("sonde")
    so.add_argument("--fiches", required=True)
    so.add_argument("--notes", required=True)
    so.add_argument("--out", required=True)
    so.add_argument("--couche", type=int, default=5, help="bloc du modèle de base dont on lit les activations (0 = premier)")
    so.add_argument("--base", default=str(ROOT / "runs/chatbot_rtx/runs/base/best.pt"))
    so.add_argument("--controle-fiches", default=None)
    so.add_argument("--controle-notes", default=None)
    so.add_argument("--device", default="cuda")
    a = p.parse_args()
    {"tirer": cmd_tirer, "noter": cmd_noter, "apprendre": cmd_apprendre, "sonde": cmd_sonde}[a.cmd](a)


if __name__ == "__main__":
    main()
