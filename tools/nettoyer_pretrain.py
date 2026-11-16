"""Filtre les tokens de pré-entraînement déjà préparés (.bin uint16) et écrit de NOUVEAUX fichiers.

RÉSULTAT DE L'ESSAI (14/11/2026) : NÉGATIF. Le modèle publié n'est PAS entraîné sur ces données
filtrées. Un modèle pré-entraîné sur les données filtrées puis affiné comme le modèle de référence a
été jugé à l'aveugle sur 72 réponses : 38 sans défaut, contre 44 pour la référence ; son BPB sur la
validation Molière ne bouge pas de façon mesurable (1,0572 contre 1,0560). Des données plus propres
n'ont donc pas donné de meilleures réponses. L'outil est gardé pour que l'essai puisse être refait.

Les .bin d'origine (data/stages/chatbot_20261113/pretrain) ne sont jamais modifiés. Chaque source est
découpée en unités (un document, ou un morceau de --morceaux tokens pour les documents longs), chaque
unité est décodée, mesurée par des traits simples (apostrophes perdues, part de mots-outils,
ponctuation, répétition…) et, si --nll est donné, par la surprise moyenne du modèle de base (nats par
token). Une unité est écartée si elle vérifie UNE des règles du fichier --regles ; les seuils de ce
fichier sont choisis sur un échantillon noté par des lecteurs, puis contrôlés sur un tirage neuf.

  traits   : traits des documents d'un échantillon (fiches JSON : numero, debut, fin, texte ;
             « tools/trier_fineweb.py tirer » en écrit pour FineWeb)
  filtrer  : source -> nouveau .bin + rapport + échantillons « gardé » / « écarté » à faire relire
  finaliser: meta.json du nouveau dossier (sources non filtrées : liens physiques vers l'origine)
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/stages/chatbot_20261113/pretrain"
TOKENIZER = ROOT / "submissions/2026-11-11_153839/tokenizer.json"
EOT = 0

MOT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ]+")
CHIFFRE = re.compile(r"\d")
FIN_PHRASE = re.compile(r"[.!?…]")
# Apostrophe remplacée par une espace : « l entretien », « qu il », « c est ».
ESPACE_APOS = re.compile(r"(?<![\w'’-])(?<!\d )(?:[ldjnsmtcLDJNSMTC]|[Qq]u) (?=[aeiouyhéèêàâîôû]\w)")
STOP = frozenset("de la le et les des en un une du que est pour qui dans a par plus pas au sur se ne ce il sont avec son "
                 "on à ou mais comme vous nous je elle ils être fait cette aux y été ont sa ses leur tout d l n s qu c j m t".split())
# Mots soudés qui ne sont pas des mots français : signe d'apostrophes supprimées.
SOUDES = frozenset("cest jai nest quil quils sest daccord aujourdhui quon jen cétait nétait dêtre davoir lhomme dautres "
                   "lensemble lutilisation lentreprise sil lorsquil puisquil jusquà nont lon dun nai jétais létat lannée "
                   "lécole linformation lhistoire leau lâge lamour lenfant lautre lidée dautre dabord dailleurs dici "
                   "quelquun nen jy cétaient quau quaux quen quune quun navait navez nétaient sagit sétait "
                   "lobjet lusage lorigine loccasion lexpérience lintérieur lextérieur lespace lordre lheure".split())
NOMS = ["n_mots", "apos_100", "soudes_1000", "espace_1000", "stop", "alpha", "chiffres", "ponct_100", "maj", "rep3",
        "top_mot", "long_mot"]


def traits(t):
    """Traits d'un texte, dans l'ordre de NOMS."""
    w = MOT.findall(t)
    n = len(w)
    if n == 0:
        return [0.0] * len(NOMS)
    wl = [x.lower() for x in w]
    non_blanc = max(1, len(t) - t.count(" ") - t.count("\n"))
    lettres = sum(len(x) for x in w)
    tri = set(zip(wl, wl[1:], wl[2:]))
    rep3 = 1.0 - len(tri) / (n - 2) if n >= 32 else 0.0
    pleins = collections.Counter(x for x in wl if len(x) > 3 and x not in STOP)
    return [n,
            100.0 * (t.count("'") + t.count("’")) / n,
            1000.0 * sum(1 for x in wl if x in SOUDES) / n,
            1000.0 * len(ESPACE_APOS.findall(t)) / n,
            sum(1 for x in wl if x in STOP) / n,
            lettres / non_blanc,
            len(CHIFFRE.findall(t)) / non_blanc,
            100.0 * len(FIN_PHRASE.findall(t)) / n,
            sum(1 for x in w if x[0].isupper()) / n,
            rep3,
            (pleins.most_common(1)[0][1] / n) if pleins else 0.0,
            lettres / n]


OPS = {"<": lambda a, b: a < b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


def raison(v, regles):
    """Première règle vérifiée (toutes ses conditions vraies), sinon None. v : dict trait -> valeur."""
    for r in regles:
        if all(OPS[op](v[nom], seuil) for nom, op, seuil in r["si"]):
            return r["raison"]
    return None


def unites(arr, morceaux):
    """(début, fin) de chaque unité : un document (EOT en tête), coupé tous les `morceaux` tokens s'il est long."""
    pos = np.flatnonzero(np.asarray(arr) == EOT)
    if len(pos) == 0 or pos[0] != 0:
        pos = np.concatenate([[0], pos])
    fins = np.concatenate([pos[1:], [len(arr)]])
    if not morceaux:
        return np.stack([pos, fins], axis=1)
    out = []
    for a, b in zip(pos.tolist(), fins.tolist()):
        if b - a <= morceaux * 3 // 2:
            out.append((a, b))
        else:
            c = list(range(a, b, morceaux))
            if b - c[-1] < morceaux // 2 and len(c) > 1:  # petit reste : collé au morceau d'avant
                c.pop()
            out.extend(zip(c, c[1:] + [b]))
    return np.array(out, dtype=np.int64)


_tok = None
_clf = None


def _travail(tache):
    """Décode un bloc d'unités et renvoie leurs traits (un processus par bloc)."""
    global _tok, _clf
    chemin, bornes, classifieur = tache
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(TOKENIZER))
    arr = np.memmap(chemin, dtype=np.uint16, mode="r")
    textes = _tok.decode_batch([arr[a:b].tolist() for a, b in bornes])
    T = [traits(t) for t in textes]
    if classifieur:  # score du tri appris (tools/trier_fineweb.py), sur le même texte
        if _clf is None:
            sys.path.insert(0, str(Path(__file__).parent))
            import trier_fineweb
            _clf = trier_fineweb.Classifieur(classifieur)
        return np.array([x + [_clf.score(t, x)] for t, x in zip(textes, T)], dtype=np.float32)
    return np.array([x + [0.0] for x in T], dtype=np.float32)


def tous_les_traits(chemin, u, procs, classifieur=None, bloc=2000):
    """Traits (colonnes de NOMS) puis score du tri appris (dernière colonne, 0 sans classifieur)."""
    taches = [(str(chemin), u[i:i + bloc].tolist(), classifieur) for i in range(0, len(u), bloc)]
    t0, fait, morceaux = time.time(), 0, []
    with ProcessPoolExecutor(procs) as ex:
        for k, m in enumerate(ex.map(_travail, taches, chunksize=4)):
            morceaux.append(m)
            fait += len(m)
            if k % 50 == 0:
                print(f"  traits : {fait:,} / {len(u):,} unités ({time.time() - t0:.0f} s)", flush=True)
    return np.concatenate(morceaux)


def charger_base(ckpt, device):
    sys.path.insert(0, str(ROOT))
    import torch
    from chatbot.generate import load_checkpoint
    model, _ = load_checkpoint(ckpt)
    return torch, model.to(device).eval()


def surprise(arr, u, ckpt, device="cuda", lot=384, fenetre=512):
    """Surprise moyenne (nats par token) des `fenetre` premiers tokens de chaque unité, par le modèle de base."""
    torch, model = charger_base(ckpt, device)
    ordre = np.argsort(u[:, 1] - u[:, 0], kind="stable")
    out = np.zeros(len(u), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ordre), lot):
            idx = ordre[i:i + lot]
            longs = np.minimum(u[idx, 1] - u[idx, 0], fenetre)
            L = int(longs.max())
            if L < 2:
                continue
            x = np.zeros((len(idx), L), dtype=np.int64)
            for r, (j, n) in enumerate(zip(idx, longs)):
                x[r, :n] = arr[u[j, 0]:u[j, 0] + n]
            xt = torch.from_numpy(x).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(xt[:, :-1])[0]
            nll = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), xt[:, 1:], reduction="none")
            masque = torch.arange(L - 1, device=device)[None, :] < torch.from_numpy(longs - 1).to(device)[:, None]
            out[idx] = ((nll * masque).sum(1) / masque.sum(1).clamp(min=1)).cpu().numpy()
            if (i // lot) % 200 == 0:
                print(f"  surprise : {i:,} / {len(u):,} unités ({time.time() - t0:.0f} s)", flush=True)
    return out


def activations(arr, u, ckpt, couche, device="cuda", lot=384, fenetre=512, sonde=None):
    """Moyenne des activations du bloc `couche` du modèle de base sur les `fenetre` premiers tokens de chaque unité.

    Sans `sonde` : matrice (unités, d_model). Avec sonde = (v, b, moy, ect) : le score de la sonde linéaire,
    un nombre par unité (b + v · activations normalisées), calculé sur la carte sans garder la matrice."""
    torch, model = charger_base(ckpt, device)
    vu = {}

    class Assez(Exception):
        """Le bloc voulu est calculé : inutile de finir le modèle."""

    def crochet(m, i, o):
        vu["a"] = o
        raise Assez
    model.blocks[couche].register_forward_hook(crochet)
    ordre = np.argsort(u[:, 1] - u[:, 0], kind="stable")
    d = model.wte.weight.shape[1]
    out = np.zeros(len(u) if sonde else (len(u), d), dtype=np.float32)
    if sonde:
        v, b, moy, ect = (torch.tensor(np.asarray(x, dtype=np.float32), device=device) for x in sonde)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ordre), lot):
            idx = ordre[i:i + lot]
            longs = np.minimum(u[idx, 1] - u[idx, 0], fenetre)
            L = min(fenetre, -(-int(longs.max()) // 64) * 64)  # longueurs arrondies : peu de tailles de lots différentes
            x = np.zeros((len(idx), L), dtype=np.int64)
            for r, (j, n) in enumerate(zip(idx, longs)):
                x[r, :n] = arr[u[j, 0]:u[j, 0] + n]
            xt = torch.from_numpy(x).to(device)
            if (i // lot) % 50 == 0 and device == "cuda":
                torch.cuda.empty_cache()  # sinon le cache de PyTorch sature la carte
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    model(xt)
            except Assez:
                pass
            masque = (torch.arange(L, device=device)[None, :] < torch.from_numpy(longs).to(device)[:, None]).float()[..., None]
            moyenne = (vu["a"].float() * masque).sum(1) / masque.sum(1).clamp(min=1)
            out[idx] = (((moyenne - moy) / ect) @ v + b).cpu().numpy() if sonde else moyenne.cpu().numpy()
            if (i // lot) % 200 == 0:
                print(f"  activations : {i:,} / {len(u):,} unités ({time.time() - t0:.0f} s)", flush=True)
    return out


def cmd_traits(a):
    fiches = json.loads(Path(a.fiches).read_text(encoding="utf-8"))
    arr = np.memmap(SRC / f"{a.source}.bin", dtype=np.uint16, mode="r")
    u = np.array([(f["debut"], f["fin"]) for f in fiches], dtype=np.int64)
    T = np.array([traits(f["texte"]) for f in fiches], dtype=np.float32)
    s = surprise(arr, u, a.base, a.device) if a.nll else np.zeros(len(u), dtype=np.float32)
    out = [{"numero": f["numero"], **{n: float(v) for n, v in zip(NOMS, t)}, "nll": float(x)} for f, t, x in zip(fiches, T, s)]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(len(out), "documents ->", a.out)


def cmd_filtrer(a):
    regles = json.loads(Path(a.regles).read_text(encoding="utf-8"))
    r_source = regles["sources"][a.source]
    besoin_nll = any(nom == "nll" for r in r_source["regles"] for nom, _, _ in r["si"])
    besoin_tri = any(nom == "score_tri" for r in r_source["regles"] for nom, _, _ in r["si"])
    if Path(a.dst).resolve() == SRC.resolve():
        sys.exit("ERREUR : --dst doit être un autre dossier que l'origine")
    if besoin_tri and not (r_source.get("classifieur") or r_source.get("sonde")):
        sys.exit("ERREUR : une règle utilise score_tri mais la source n'a ni « sonde » ni « classifieur »")
    if r_source.get("classifieur") and r_source.get("morceaux"):
        sys.exit("ERREUR : le classifieur note des documents entiers, pas des morceaux")
    if besoin_nll and not r_source.get("morceaux"):
        sys.exit("ERREUR : une règle nll exige des morceaux (la surprise ne lit que 512 tokens)")
    chemin = SRC / f"{a.source}.bin"
    arr = np.memmap(chemin, dtype=np.uint16, mode="r")
    t0 = time.time()
    u = unites(arr, r_source.get("morceaux", 0))
    if a.limite:
        u = u[:a.limite]
    print(f"{a.source} : {len(arr):,} tokens, {len(u):,} unités (morceaux = {r_source.get('morceaux', 0)})", flush=True)
    classifieur = r_source.get("classifieur")
    TS = tous_les_traits(chemin, u, a.procs, str(ROOT / classifieur) if classifieur else None)
    T, score_tri = TS[:, :len(NOMS)], TS[:, len(NOMS)]
    sonde_sha = base_sha = None
    if r_source.get("sonde"):  # sonde linéaire sur les activations du modèle de base (tools/trier_fineweb.py sonde)
        z = np.load(ROOT / r_source["sonde"])
        base_sha = hashlib.sha256(Path(a.base).read_bytes()).hexdigest()
        if str(z["base_sha256"]) != base_sha:
            sys.exit("ERREUR : la sonde a été apprise sur un autre modèle de base que --base")
        sonde_sha = hashlib.sha256((ROOT / r_source["sonde"]).read_bytes()).hexdigest()
        score_tri = activations(arr, u, a.base, int(z["couche"]), a.device, sonde=(z["v"], z["b"], z["moy"], z["ect"]))
    s = surprise(arr, u, a.base, a.device) if besoin_nll else np.zeros(len(u), dtype=np.float32)
    garde = np.ones(len(u), dtype=bool)
    compte = collections.Counter()
    tokens_ecartes = collections.Counter()
    pourquoi = [None] * len(u)
    for i in range(len(u)):
        v = dict(zip(NOMS, T[i].tolist()))
        v["nll"] = float(s[i])
        v["score_tri"] = float(score_tri[i])
        v["tokens"] = int(u[i, 1] - u[i, 0])
        r = None if v["tokens"] <= 1 else raison(v, r_source["regles"])
        if r:
            garde[i] = False
            pourquoi[i] = r
            compte[r] += 1
            tokens_ecartes[r] += v["tokens"]
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n_out = separateurs = 0
    if not a.sans_ecrire:
        tmp = dst / f"{a.source}.bin.tmp"
        with open(tmp, "wb") as f:
            fin = -1
            for i in np.flatnonzero(garde):
                morceau = np.asarray(arr[u[i, 0]:u[i, 1]])
                if morceau[0] != EOT and u[i, 0] != fin:  # trou dans un document : nouvelle frontière
                    f.write(np.array([EOT], dtype=np.uint16).tobytes())
                    n_out += 1
                    separateurs += 1
                f.write(morceau.tobytes())
                n_out += len(morceau)
                fin = u[i, 1]
        os.replace(tmp, dst / f"{a.source}.bin")
    else:
        n_out = int((u[garde, 1] - u[garde, 0]).sum())
    total = int((u[:, 1] - u[:, 0]).sum())
    rapport = {"source": a.source, "origine": str(chemin.relative_to(ROOT)).replace("\\", "/"), "regles": r_source,
               "version_regles": regles.get("version"), "unites": int(len(u)), "unites_gardees": int(garde.sum()),
               "unites_ecartees": dict(compte), "tokens_lus": total, "tokens_gardes": n_out,
               "tokens_ecartes": dict(tokens_ecartes), "part_tokens_ecartes": round(1 - n_out / max(1, total), 5),
               "limite": a.limite, "ecrit": not a.sans_ecrire, "separateurs_ajoutes": separateurs,
               "classifieur_sha256": hashlib.sha256((ROOT / classifieur).read_bytes()).hexdigest() if classifieur else None,
               "sonde_sha256": sonde_sha, "base_sha256": base_sha,
               "secondes": round(time.time() - t0, 1)}
    (dst / (f"rapport_{a.source}.json" if not a.sans_ecrire else f"rapport_essai_{a.source}.json")).write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rapport[k] for k in ("unites", "unites_gardees", "unites_ecartees", "tokens_lus", "tokens_gardes",
                                              "part_tokens_ecartes", "secondes")}, ensure_ascii=False))
    if a.controle:  # tirage neuf, au prorata des tokens, pour relecture à l'aveugle
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER))
        rng = random.Random(a.graine)
        poids = (u[:, 1] - u[:, 0]).astype(np.float64)
        fiches = []
        for etat, masque in (("garde", garde), ("ecarte", ~garde)):
            idx = np.flatnonzero(masque)
            if len(idx) == 0:
                continue
            p = poids[idx] / poids[idx].sum()
            tires = np.random.default_rng(rng.randrange(2 ** 31)).choice(idx, size=min(a.controle, len(idx)), replace=False, p=p)
            for i in tires.tolist():
                fiches.append({"etat": etat, "raison": pourquoi[i], "unite": i, "tokens": int(u[i, 1] - u[i, 0]),
                               "traits": {n: round(float(x), 4) for n, x in zip(NOMS, T[i])} | {"nll": round(float(s[i]), 4), "score_tri": round(float(score_tri[i]), 4)},
                               "texte": tok.decode(arr[u[i, 0]:u[i, 1]].tolist())})
        rng.shuffle(fiches)
        for k, f in enumerate(fiches, 1):
            f["numero"] = k
        cdir = Path(a.controle_dst)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"controle_{a.source}_cle.json").write_text(json.dumps(fiches, ensure_ascii=False), encoding="utf-8")
        for lot in range(0, len(fiches), 40):
            with open(cdir / f"controle_{a.source}_lot{lot // 40 + 1:02d}.txt", "w", encoding="utf-8") as f:
                for x in fiches[lot:lot + 40]:
                    t = x["texte"]
                    if len(t) > 1500:
                        m = len(t) // 2
                        t = t[:900] + "\n[…coupé…]\n" + t[m:m + 600]
                    f.write(f"===== DOCUMENT {x['numero']} ({x['tokens']} tokens) =====\n{t}\n\n")
        print(f"contrôle : {len(fiches)} unités mélangées -> {cdir}")


def cmd_finaliser(a):
    dst = Path(a.dst)
    if dst.resolve() == SRC.resolve():
        sys.exit("ERREUR : --dst doit être un autre dossier que l'origine")
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    for nom, info in meta["sources"].items():
        cible = dst / info["file"]
        rap = dst / f"rapport_{nom}.json"
        if rap.exists():
            r = json.loads(rap.read_text(encoding="utf-8"))
            if r["limite"]:
                sys.exit(f"ERREUR : {nom} a été filtrée avec --limite (essai), pas en entier")
            n = cible.stat().st_size // 2
            if r["tokens_lus"] != info["tokens"]:
                sys.exit(f"ERREUR : {nom} : {r['tokens_lus']} tokens lus, l'origine en compte {info['tokens']}")
            if n != r["tokens_gardes"]:
                sys.exit(f"ERREUR : {cible} a {n} tokens, le rapport en annonce {r['tokens_gardes']}")
            info["tokens_origine"] = info["tokens"]
            info["tokens"] = n
            info["key"] = dict(info["key"], nettoyage={"version": r["version_regles"], "regles": r["regles"],
                                                       "classifieur_sha256": r.get("classifieur_sha256"),
                                                       "sonde_sha256": r.get("sonde_sha256"), "base_sha256": r.get("base_sha256")})
            info["nettoyage"] = {k: r[k] for k in ("unites", "unites_gardees", "unites_ecartees", "tokens_ecartes", "part_tokens_ecartes")}
        elif not cible.exists():
            try:
                os.link(SRC / info["file"], cible)  # même fichier, sans recopier
            except OSError:
                shutil.copy2(SRC / info["file"], cible)
        print(f"{nom}: {cible.stat().st_size // 2:,} tokens" + (" (filtrée)" if rap.exists() else " (inchangée)"))
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("meta.json écrit dans", dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    base = str(ROOT / "runs/chatbot_rtx/runs/base/best.pt")
    t = s.add_parser("traits")
    t.add_argument("--fiches", required=True)
    t.add_argument("--source", default="fineweb")
    t.add_argument("--out", required=True)
    t.add_argument("--nll", action="store_true")
    f = s.add_parser("filtrer")
    f.add_argument("--source", required=True)
    f.add_argument("--regles", required=True)
    f.add_argument("--dst", required=True)
    f.add_argument("--limite", type=int, default=0, help="essai : seulement les N premières unités")
    f.add_argument("--sans-ecrire", action="store_true", help="compter sans écrire le .bin")
    f.add_argument("--procs", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 4)))
    f.add_argument("--controle", type=int, default=0, help="N unités gardées + N écartées à faire relire")
    f.add_argument("--controle-dst", default=str(ROOT / "runs/evals_manuelles/nettoyage"))
    f.add_argument("--graine", type=int, default=20261115)
    z = s.add_parser("finaliser")
    z.add_argument("--dst", required=True)
    for q in (t, f):
        q.add_argument("--base", default=base)
        q.add_argument("--device", default="cuda")
    a = p.parse_args()
    {"traits": cmd_traits, "filtrer": cmd_filtrer, "finaliser": cmd_finaliser}[a.cmd](a)


if __name__ == "__main__":
    main()
