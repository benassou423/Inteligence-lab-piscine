"""Écarte les conversations de synthèse dont un message d'utilisateur est identique (lettres seules, sans casse)
à un message des jeux de test de tools/nuit_moliere.py : le modèle ne doit pas avoir vu les messages qui le jugent."""
import importlib.util, io, json, re, sys, collections, hashlib
from pathlib import Path
import zstandard
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("nm", "tools/nuit_moliere.py"); nm = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(nm)
norm = lambda s: "".join(re.findall(r"[a-zà-ÿœ]+", s.lower()))
proto = {norm(m): jeu for jeu, (isoles, suivies) in nm.JEUX.items() for m in isoles + [x for c in suivies for x in c]}
gardes, ecart = [], collections.Counter()
with open(src, "rb") as f:
    for l in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
        if not l.strip(): continue
        d = json.loads(l)
        touche = [proto[norm(t["text"])] for t in d["turns"][0::2] if norm(t["text"]) in proto]
        if touche: ecart[touche[0]] += 1
        else: gardes.append(d)
dst.parent.mkdir(parents=True, exist_ok=True)
with open(dst, "wb") as f, zstandard.ZstdCompressor(level=10).stream_writer(f) as z:
    for d in gardes: z.write((json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8"))
r = {"source": str(src).replace("\\", "/"), "lues": len(gardes) + sum(ecart.values()), "gardees": len(gardes), "ecartees_par_jeu": dict(ecart),
     "messages_de_test": len(proto), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
     "tours": dict(sorted(collections.Counter(len(d["turns"]) for d in gardes).items())),
     "mots": sum(len(t["text"].split()) for d in gardes for t in d["turns"])}
Path(str(dst).replace(".jsonl.zst", ".garde.json")).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(r, ensure_ascii=False))
