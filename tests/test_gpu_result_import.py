"""Le retour GPU ne peut pas changer les fichiers hors de son périmètre ni son protocole."""
import zipfile
import pytest
from tools.import_windows_results import extract_result, validate_meta


@pytest.mark.parametrize('name', ['../kit.py', '/tmp/file', 'ftb/model.py', 'runs\\evil.py'])
def test_foreign_and_traversal_members_rejected(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, 'ne doit pas être extrait')
    with pytest.raises(ValueError):
        extract_result(archive, tmp_path / 'out')
    assert not (tmp_path / 'kit.py').exists()


@pytest.mark.parametrize('changes', [{'official': False}, {'tokens_seen': 25001984},
                                    {'precision': 'torch.float32'}, {'device': 'mps'}, {'dataset_sha256': 'autre'}])
def test_proxy_or_wrong_device_cannot_pass_for_official_gpu(changes):
    expected = {'dataset_sha256': 'dataset', 'tokenizer_sha256': 'tokenizer'}
    config = {'train': {'seed': 1337}}
    meta = {**expected, 'official': True, 'tokens_seen': 100007936, 'precision': 'torch.float16',
            'device': 'NVIDIA GeForce RTX 5090', 'train_config': {'seed': 1337}, 'val_bpb': 1.5}
    validate_meta(meta, expected, config)
    with pytest.raises(ValueError):
        validate_meta({**meta, **changes}, expected, config)
