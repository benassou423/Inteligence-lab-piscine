"""Le masquage final doit préserver les faits et la provenance du flux retenu."""
import gzip
import hashlib
import json

import pytest

from tools.prepare_masked_reference import gz_write, sha, transform, verify_provenance


def inputs(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    texts = ['La population était de 1 234 habitants en 2020. Contact : a@example.org.',
             'La population était de 1 234 habitants en 2020. Contact : b@example.org.',
             'Le téléphone de la mairie est le 01 23 45 67 89.']
    with gz_write(source / 'train.jsonl.gz') as f, gz_write(source / 'provenance.jsonl.gz') as p:
        for index, text in enumerate(texts):
            f.write(json.dumps({'text': text}) + '\n')
            p.write(json.dumps({'row_zero_based': index, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'id': f'wikipedia_fr:{index}', 'source': 'wikipedia_fr', 'domain': 'encyclopedique',
                'url': f'https://example.org/{index}', 'date': '2026-01-01', 'license': 'fixture'}) + '\n')
    return source


def test_faits_provenance_et_nouveaux_doublons(tmp_path):
    source = inputs(tmp_path)
    before = (source / 'train.jsonl.gz').read_bytes()
    output = tmp_path / 'out'
    result = transform(source, output)
    assert result['counts']['documents_out'] == 2
    assert result['counts']['new_exact_duplicates'] == 1
    with gzip.open(output / 'input.jsonl.gz', 'rt') as f, gzip.open(output / 'provenance.jsonl.gz', 'rt') as p:
        rows, refs = [json.loads(s) for s in f], [json.loads(s) for s in p]
    assert rows[0]['text'] == 'La population était de 1 234 habitants en 2020. Contact : [EMAIL].'
    assert rows[1]['text'] == 'Le téléphone de la mairie est le [TELEPHONE].'
    assert [r['row_zero_based'] for r in refs] == [0, 1]
    assert [r['original_row_zero_based'] for r in refs] == [0, 2]
    assert all(hashlib.sha256(r['text'].encode()).hexdigest() == ref['text_sha256'] for r, ref in zip(rows, refs))
    assert (source / 'train.jsonl.gz').read_bytes() == before
    digest = (output / 'input.jsonl.gz').read_bytes()
    transform(source, output)
    assert (output / 'input.jsonl.gz').read_bytes() == digest


def test_provenance_fausse_refusee(tmp_path):
    source = inputs(tmp_path)
    with gz_write(source / 'provenance.jsonl.gz') as f:
        f.write(json.dumps({'row_zero_based': 0, 'text_sha256': 'faux'}) + '\n')
    with pytest.raises(ValueError, match='Provenance non alignée'):
        transform(source, tmp_path / 'out')


def test_index_final_detecte_un_texte_remplace(tmp_path):
    source, output = inputs(tmp_path), tmp_path / 'mix'
    result = transform(source, output)
    (output / 'input.jsonl.gz').replace(output / 'train.jsonl.gz')
    manifest = {'documents': 2, 'by_source': {'wikipedia_fr': {'documents': 2}},
                'train_sha256': sha(output / 'train.jsonl.gz')}
    (output / 'manifest.json').write_text(json.dumps(manifest))
    assert verify_provenance(output)['documents'] == 2
    with gz_write(output / 'train.jsonl.gz') as f:
        f.write(json.dumps({'text': 'un texte remplacé'}) + '\n')
    with pytest.raises(ValueError, match='divergente'):
        verify_provenance(output)
