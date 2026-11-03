"""Étape decontaminate : définition du kit, copies retirées, formules gardées, lignes masquées."""
import json
import random
from pathlib import Path

import pytest

ftb_contamination = pytest.importorskip("ftb.contamination")

from pipeline.dedup.decontaminate import DecontaminateStage, ValIndex  # noqa: E402
from pipeline.run import Context  # noqa: E402

FORMULE = "Le scrutin est annoncé dans l'enceinte de l'Assemblée nationale et la parole est au rapporteur"
VAL = [
    {"id": "v1", "genre": "dialogue", "text": FORMULE + ". Mme Durand : Le budget des écoles rurales baisse "
     "encore cette année alors que les classes sont pleines et que les maires ne savent plus comment faire."},
    {"id": "v2", "genre": "presse", "text": "Un violent orage a frappé la ville de Tours dans la nuit de mardi à "
     "mercredi, privant d'électricité plus de dix mille foyers selon le gestionnaire du réseau électrique."},
]
SPECIFIC = "Mme Durand : Le budget des écoles rurales baisse encore cette année alors que les classes sont pleines"


def test_identique_au_kit():
    """Le pré-filtre (premier, dernier mot) ne change rien : même résultat que tout hacher."""
    idx = ValIndex(VAL)
    rng = random.Random(0)
    words = (VAL[0]["text"] + " " + VAL[1]["text"] + " le la de et un orage").split()
    for _ in range(300):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(5, 60)))
        if rng.random() < 0.3:  # injecte un vrai passage de la val
            text += " " + VAL[1]["text"][: rng.randint(60, 200)]
        brute = {h for h in ftb_contamination.ngram_hashes(text) if h in idx.owner}
        assert idx.shared(text) == brute


def test_lignes_retirees():
    idx = ValIndex(VAL)
    text = "Première ligne sans rapport.\n" + SPECIFIC + " depuis des années.\nDernière ligne sans rapport."
    targets = idx.shared(text)
    out, n = idx.drop_lines(text, targets)
    assert out == "Première ligne sans rapport.\nDernière ligne sans rapport." and n == 1
    assert not idx.shared(out)


def _stage(tmp_path, **params):
    p = tmp_path / "val.jsonl"
    p.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in VAL), encoding="utf-8")
    st = DecontaminateStage({"against": str(p), **params}, Context(root=Path("."), config_name="t", sources=["s"]))
    st.setup()
    return st


def test_copie_formule_et_masquage(tmp_path):
    st = _stage(tmp_path, formula_min_docs=3, doc_min_coverage=0.2)
    train = [
        {"id": "s:copie", "text": "Dépêche.\n" + VAL[1]["text"]},  # copie d'un doc de val
        {"id": "s:f1", "text": "Séance 1.\n" + FORMULE + "."},  # formule seule, dans 3 séances
        {"id": "s:f2", "text": "Séance 2.\n" + FORMULE + "."},
        {"id": "s:f3", "text": "Séance 3.\n" + FORMULE + ".\nLigne à garder."},
        {"id": "s:spec", "text": "Intro gardée.\n" + SPECIFIC + ".\nConclusion gardée."},  # recouvrement résiduel
        {"id": "s:rien", "text": "Aucun rapport avec la validation."},
    ]
    records = [[i, st.scan_doc(d)] for i, d in enumerate(train) if st.scan_doc(d)]
    st.plan([("s/part-00000.jsonl.zst", "s", {"n": len(train), "records": records})])
    out = [st.apply_at(dict(d), "s/part-00000.jsonl.zst", i) for i, d in enumerate(train)]
    assert not out[0].keep and out[0].reason == "val_copy" and out[0].doc["meta"]["decontaminate_val_id"] == "v2"
    assert all(o.keep for o in out[1:4]) and out[3].doc["text"].endswith("Ligne à garder.")
    assert out[4].keep and out[4].doc["text"] == "Intro gardée.\nConclusion gardée."
    assert out[5].keep and out[5].doc["text"] == train[5]["text"]
    # le recouvrement résiduel avec v1 se limite aux formules (< 10 % du document de val)
    kept = [o.doc["text"] for o in out if o.keep]
    union = set().union(*(st.index.shared(t) for t in kept))
    assert len(union & set(st.index.owner)) / st.index.sizes[0] < 0.5
    report, md = st.finalize([], {})
    assert report["actions"] == {"copy": 1, "formula_only": 3, "mask": 1} and "Formule" in md


def test_couverture_part_la_plus_grande_et_ordre_indifferent():
    """Le document de val « le plus recouvert » est celui dont la PART recouverte est la plus grande
    (un petit document copié en entier), et le choix ne dépend pas de l'ordre des 13-grammes
    (un set de bytes : son ordre change d'un processus à l'autre avec PYTHONHASHSEED)."""
    gros = " ".join(f"mot{i}" for i in range(200))  # 188 13-grammes
    petit = "le petit texte de la validation tient en tout juste quatorze mots bien comptés"  # 2 13-grammes
    idx = ValIndex([{"id": "gros", "text": gros}, {"id": "petit", "text": petit}])
    hits = sorted(idx.shared(" ".join(f"mot{i}" for i in range(15)) + "\n" + petit))
    assert len(hits) == 5  # 3 avec le gros, 2 avec le petit
    for order in (hits, hits[::-1]):
        i, k, cov = idx.coverage(order)
        assert (idx.docs[i]["id"], k, cov) == ("petit", 2, 1.0)
    # Égalité de part : le plus petit index, quel que soit l'ordre
    jumeaux = ValIndex([{"id": "a", "text": petit}, {"id": "b", "text": petit}])
    h = sorted(jumeaux.shared(petit))
    assert jumeaux.coverage(h)[0] == jumeaux.coverage(h[::-1])[0] == 0
