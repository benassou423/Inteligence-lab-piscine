"""Une reproduction ne doit remplacer ni un dossier existant ni le kit."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.reproduce import LINKS, check_kit, ensure_links, repair_cache


def test_links_nosync_et_refus_de_remplacement(tmp_path):
    path = tmp_path / 'data/raw'
    path.mkdir(parents=True)
    (path / 'preuve.txt').write_text('à conserver')
    with pytest.raises(RuntimeError, match='dossier ordinaire'):
        ensure_links(tmp_path)
    assert (path / 'preuve.txt').read_text() == 'à conserver'
    assert not (tmp_path / 'runs').exists()
    (path / 'preuve.txt').unlink()
    path.rmdir()
    ensure_links(tmp_path)
    ensure_links(tmp_path)
    assert all((tmp_path / name).is_symlink() for name in LINKS)
    assert all((tmp_path / name).resolve().name.endswith('.nosync') for name in LINKS)


def test_kit_modifie_detecte(tmp_path):
    manifest = tmp_path / 'docs/remise/KIT_SHA256.json'
    manifest.parent.mkdir(parents=True)
    sample = tmp_path / 'ftb/model.py'
    sample.parent.mkdir()
    sample.write_text('référence')
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'files': {'ftb/model.py': digest}}))
    assert check_kit(tmp_path) == 1
    sample.write_text('modifié')
    with pytest.raises(RuntimeError, match='Kit différent'):
        check_kit(tmp_path)
    sample.write_text('référence')
    (sample.parent / 'ajout.py').write_text('inattendu')
    with pytest.raises(RuntimeError, match='Fichier ajouté au kit'):
        check_kit(tmp_path)


def test_lien_casse_conserve_et_cible_creee(tmp_path):
    (tmp_path / 'data').mkdir()
    link = tmp_path / 'data/raw'
    link.symlink_to('raw.nosync', target_is_directory=True)
    ensure_links(tmp_path)
    assert link.is_symlink() and link.readlink().name == 'raw.nosync'
    assert link.resolve().is_dir()


def test_cache_indisponible_deplace_sans_toucher_aux_donnees(tmp_path):
    cache = tmp_path / 'data/stages/nettoye/01_exclude_heldout/_units/source/record.json'
    raw = tmp_path / 'data/raw/source/part-00000.jsonl.zst'
    corpus = tmp_path / 'data/stages/nettoye/01_exclude_heldout/source/part-00000.jsonl.zst'
    for path, value in [(cache, 'cache à régénérer'), (raw, 'brut original'), (corpus, 'corpus conservé')]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    moved = repair_cache(tmp_path, unavailable=lambda p: p == cache)
    assert len(moved) == 1 and not cache.exists()
    assert Path(moved[0]['preserved']).read_text() == 'cache à régénérer'
    assert raw.read_text() == 'brut original' and corpus.read_text() == 'corpus conservé'
    assert repair_cache(tmp_path, unavailable=lambda p: True) == []
