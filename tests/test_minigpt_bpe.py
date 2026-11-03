"""Tests du tokenizer BPE (minigpt/bpe.py) : uv run pytest -q tests/test_minigpt_bpe.py"""
import json
from collections import Counter

import pytest
import regex

from minigpt.bpe import SPLIT_PATTERN, BPETokenizer, merge

# Texte d'entraînement : d'après Molière, L'Avare (IV, 7) et Le Misanthrope (I, 1).
TEXT = (
    "Au voleur ! au voleur ! à l'assassin ! au meurtrier ! Justice, juste Ciel ! je suis perdu, "
    "je suis assassiné, on m'a coupé la gorge, on m'a dérobé mon argent. Qui peut-ce être ? "
    "Qu'est-il devenu ? Où est-il ? Où se cache-t-il ? Que ferai-je pour le trouver ? Où courir ? "
    "Où ne pas courir ? N'est-il point là ? N'est-il point ici ? Qui est-ce ? Arrête. Rends-moi "
    "mon argent, coquin… Ah ! c'est moi. Mon esprit est troublé, et j'ignore où je suis, qui je "
    "suis, et ce que je fais. Hélas ! mon pauvre argent, mon pauvre argent, mon cher ami ! on m'a "
    "privé de toi ; et puisque tu m'es enlevé, j'ai perdu mon support, ma consolation, ma joie ; "
    "tout est fini pour moi, et je n'ai plus que faire au monde : sans toi, il m'est impossible "
    "de vivre. C'en est fait, je n'en puis plus ; je me meurs, je suis mort, je suis enterré.\n\n"
    "PHILINTE. — Qu'est-ce donc ? Qu'avez-vous ?\n"
    "ALCESTE. — Laissez-moi, je vous prie.\n"
    "PHILINTE. — Mais encor, dites-moi, quelle bizarrerie…\n"
    "ALCESTE. — Laissez-moi là, vous dis-je, et courez vous cacher.\n"
)

# Du français difficile, avec des caractères absents de TEXT : ils passent par les octets.
HARD = (
    "« Hélas\u00a0! » s’écria-t-elle — l’œuvre de Molière, 1673…\n"  # \u00a0 = espace insécable
    "Cœur, Œdipe, ÇA, où, naïf, Noël ; prix\u202f: 3,50\u00a0€ – fin.\r\n"  # \u202f = insécable fine
    "\n\n   Emoji : 😀 🇫🇷 👨‍👩‍👧 \t tabulation  \n"  # famille = 3 emoji reliés par U+200D (invisible)
)

VOCAB_SIZE = 400  # 143 fusions : TEXT (1 Ko) en permet 308 au plus (vérifié plus bas)


@pytest.fixture(scope="module")
def tok():
    """Un tokenizer entraîné une fois pour tout le fichier."""
    t = BPETokenizer()
    t.train(TEXT, VOCAB_SIZE)
    return t


def test_toy_example_by_hand():
    """« aaabdaaabac » (l'exemple de Wikipédia), calculé à la main :
    - paires : aa ×4 (« aaa » contient 2 fois aa), ab ×2, bd, da, ba, ac ×1 → aa = 256 ;
    - « ZabdZabac » : Za ×2 et ab ×2 à égalité → la plus petite paire, (97, 98) = ab → 257 ;
    - « ZYdZYac » : ZY ×2 → 258 = « aaab », et le texte devient « XdXac »."""
    t = BPETokenizer()
    t.train("aaabdaaabac", 256 + 3 + 1)  # 256 octets + 3 fusions + <|endoftext|>
    assert list(t.merges.items()) == [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]
    assert t.vocab[256] == b"aa" and t.vocab[257] == b"ab" and t.vocab[258] == b"aaab"
    assert t.encode("aaabdaaabac") == [258, ord("d"), 258, ord("a"), ord("c")]
    assert t.special_tokens == {"<|endoftext|>": 259}


def test_vocab_size_is_exact(tok):
    assert len(tok.vocab) == VOCAB_SIZE
    assert len(tok.merges) == VOCAB_SIZE - 256 - 1
    assert max(tok.encode(TEXT + HARD)) < VOCAB_SIZE
    # Aucun hasard : un 2e entraînement donne exactement les mêmes fusions.
    again = BPETokenizer()
    again.train(TEXT, VOCAB_SIZE)
    assert again.merges == tok.merges
    with pytest.raises(ValueError):  # pas de place pour <|endoftext|>
        BPETokenizer().train(TEXT, 256)
    with pytest.raises(ValueError):  # « ab » n'offre qu'une seule fusion
        BPETokenizer().train("ab", 256 + 2 + 1)
    # Corpus juste assez long : 308 fusions passent avec la taille exacte, la 309e est impossible.
    full = BPETokenizer()
    full.train(TEXT, 256 + 308 + 1)
    assert len(full.vocab) == 256 + 308 + 1
    with pytest.raises(ValueError):
        BPETokenizer().train(TEXT, 256 + 309 + 1)


@pytest.mark.parametrize("s", [
    TEXT, HARD, "", " ", "\n", "aujourd'hui",
    "e\u0301",  # « é » décomposé : un e suivi d'un accent aigu combinant
    "<|endoftext|>",  # sans allow_special=True, c'est du texte ordinaire
])
def test_round_trip(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_decode_invalid_bytes(tok):
    # 195 = 0xC3, 1er octet de « é » : seul, ce n'est pas de l'UTF-8 valide → « � ».
    assert tok.decode([195]) == "\ufffd"
    assert tok.decode(list("é".encode("utf-8"))) == "é"


def test_merges_applied_in_learned_order():
    """Fusions apprises : (b, c) = 256, puis (a, b) = 257. Sur « abc », les deux paires sont
    présentes : on applique (b, c), apprise en premier → [a, 256]. C'est ce que donnerait
    l'entraînement : après la 1re fusion, il ne reste plus de paire (a, b). Fusionner de
    gauche à droite donnerait [257, c], un résultat différent."""
    t = BPETokenizer()
    t.train("bc\nbc\nbc\nab\nab\n", 256 + 2 + 1)  # bc ×3, ab ×2
    assert list(t.merges) == [(98, 99), (97, 98)]
    assert t.encode("abc") == [97, 256]


def test_encode_replays_merges_in_order(tok):
    """encode() donne le même résultat que rejouer toutes les fusions, une par une, dans
    l'ordre de l'entraînement, sur chaque morceau du texte."""
    expected = []
    for chunk in regex.findall(SPLIT_PATTERN, TEXT + HARD):
        ids = list(chunk.encode("utf-8"))
        for pair, new_id in tok.merges.items():  # dict : ordre d'insertion = ordre appris
            ids = merge(ids, pair, new_id)
        expected += ids
    assert tok.encode(TEXT + HARD) == expected


def test_train_matches_naive_bpe(tok):
    """L'entraînement optimisé (morceaux uniques, mise à jour des seuls mots touchés) apprend
    les mêmes fusions qu'un BPE naïf qui recompte tout le corpus à chaque fusion."""
    words = [list(chunk.encode("utf-8")) for chunk in regex.findall(SPLIT_PATTERN, TEXT)]
    naive = []
    for k in range(len(tok.merges)):
        counts = Counter(pair for w in words for pair in zip(w, w[1:]))
        pair = min(counts, key=lambda p: (-counts[p], p))  # même règle d'égalité
        naive.append(pair)
        words = [merge(w, pair, 256 + k) for w in words]
    assert list(tok.merges) == naive


def test_pre_split_keeps_words_apart(tok):
    assert regex.findall(SPLIT_PATTERN, "Je t’aime, dit-il.") == [
        "Je", " t", "’aime", ",", " dit", "-il", "."]
    # Le prix du motif de GPT-4 : sa contraction anglaise « 't » coupe « p'tite » en trois.
    assert regex.findall(SPLIT_PATTERN, "p'tite") == ["p", "'t", "ite"]
    # Aucun token appris ne colle une lettre à un blanc ou à une ponctuation qui la suit.
    for i in range(256, 256 + len(tok.merges)):
        assert not regex.search(r"\p{L}[\s.,;:!?]", tok.vocab[i].decode("utf-8", errors="replace"))


def test_special_tokens(tok):
    eot = tok.special_tokens["<|endoftext|>"]
    assert eot == VOCAB_SIZE - 1  # le dernier id, comme 50256 dans GPT-2
    s = "Acte I<|endoftext|>Acte II"
    ids = tok.encode(s, allow_special=True)
    assert ids.count(eot) == 1
    assert tok.decode(ids) == s
    plain = tok.encode(s)  # par défaut : simple texte, découpé en octets et fusions
    assert eot not in plain
    assert tok.decode(plain) == s
    assert tok.decode([eot]) == "<|endoftext|>"


def test_save_load_gives_same_encoding(tok, tmp_path):
    path = tmp_path / "bpe.json"
    tok.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))  # c'est du JSON lisible
    assert data["special_tokens"] == {"<|endoftext|>": VOCAB_SIZE - 1}
    assert data["merges"][0] == list(next(iter(tok.merges)))
    loaded = BPETokenizer().load(path)
    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab
    assert loaded.special_tokens == tok.special_tokens
    for s in [TEXT, HARD, "Acte I<|endoftext|>Acte II"]:
        assert loaded.encode(s, allow_special=True) == tok.encode(s, allow_special=True)


# --- Pont vers la compétition : le kit doit encoder exactement comme notre BPE ---------------

def test_pattern_is_the_kit_pattern():
    """Caractère pour caractère, le motif du kit : une fois converti, le découpage ne change pas."""
    from ftb.tokenizer import GPT4_SPLIT_PATTERN
    assert SPLIT_PATTERN == GPT4_SPLIT_PATTERN


def test_minbpe_file_converted_by_kit_encodes_like_us(tmp_path):
    """save_minbpe -> lecture par le kit -> tokenizer Hugging Face : mêmes ids que notre encode."""
    from ftb.tokenizer import from_merges, read_minbpe_model

    with open("corpus/moliere_train.txt", encoding="utf-8") as f:
        train_text = f.read(50_000)  # 50 000 premiers caractères : de quoi apprendre 300 fusions
    t = BPETokenizer()
    t.train(train_text, 256 + 300 + 1)
    path = tmp_path / "bpe.model"
    t.save_minbpe(path)

    pattern, merges, special = read_minbpe_model(path)  # le lecteur du kit, tel quel
    assert pattern == SPLIT_PATTERN
    assert merges == list(t.merges)  # mêmes fusions, dans le même ordre
    assert special == {"<|endoftext|>": 256 + 300}
    kit = from_merges(merges, pattern)

    with open("data/validation.jsonl", encoding="utf-8") as f:  # validation du kit, lecture seule
        val_docs = [json.loads(line)["text"] for _, line in zip(range(20), f)]
    texts = [TEXT, HARD, "p'tite", "Acte I<|endoftext|>Acte II", *val_docs]
    for s in texts:
        # add_special_tokens=False : le kit n'ajoute rien ; il reconnaît « <|endoftext|> » écrit
        # dans le texte, comme notre encode avec allow_special=True.
        assert kit.encode(s, add_special_tokens=False).ids == t.encode(s, allow_special=True)
