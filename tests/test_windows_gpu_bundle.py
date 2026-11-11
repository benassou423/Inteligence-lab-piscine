"""Le passage PC conserve les données et refuse les fichiers altérés/dossiers ordinaires."""
import hashlib
import json
from pathlib import Path
import pytest
from tools.windows_gpu import init_links, verify, LINKS, target_names


def test_manifest_detects_real_content_tampering(tmp_path):
    p = tmp_path / 'config.json';p.write_text('un fichier original')
    (tmp_path / 'TRANSFER_MANIFEST.json').write_text(json.dumps({'files_sha256': {'config.json': hashlib.sha256(p.read_bytes()).hexdigest()}}))
    verify(tmp_path)
    p.write_text('un fichier modifié')
    with pytest.raises(ValueError, match='modifié'):
        verify(tmp_path)


def test_links_are_idempotent_and_keep_existing_contents(tmp_path):
    init_links(tmp_path)
    for name, target in LINKS.items():
        p = tmp_path / name
        assert p.resolve() == (p.parent / target).resolve()
    (tmp_path / 'data/raw/un_document.txt').write_text('conservé')
    init_links(tmp_path)
    assert (tmp_path / 'data/raw.nosync/un_document.txt').read_text() == 'conservé'


def test_normal_heavy_directory_is_never_replaced(tmp_path):
    (tmp_path / 'data/raw').mkdir(parents=True)
    (tmp_path / 'data/raw/test.txt').write_text('à préserver')
    with pytest.raises(ValueError, match='dossier normal'):
        init_links(tmp_path)
    assert (tmp_path / 'data/raw/test.txt').read_text() == 'à préserver'


def test_two_transfer_targets_cannot_be_mixed():
    assert target_names({}) == ('full_reference_20261110', 'full_reference_rtx5090_20261110')
    assert target_names({'mix_name': 'full_reference_masked_20261110',
                         'run_name': 'full_reference_masked_rtx5090_20261110'})[0] == 'full_reference_masked_20261110'
    with pytest.raises(ValueError, match='inattendu'):
        target_names({'mix_name': 'full_reference_masked_20261110', 'run_name': 'full_reference_rtx5090_20261110'})
