"""Les bornes portent sur les choix de labels, pas une confiance statistique."""
import pytest

from tools.finalize_langid_ai_80 import sensitivity


def test_bornes_ponderees_avec_ambiguite_acceptee_et_rejetee():
    items = [{'id': 'a', 'weight': 1, 'keep': True}, {'id': 'b', 'weight': 2, 'keep': False},
             {'id': 'c', 'weight': 4, 'keep': True}, {'id': 'd', 'weight': 3, 'keep': True}]
    labels = {'a': 'f', 'b': 'f', 'c': 'f', 'd': 'n'}
    limits = sensitivity(items, labels, {'a': ['f', 'n'], 'b': ['f', 'n']}, lambda x: x == 'f')
    assert limits['precision_min'] == .5
    assert limits['precision_max'] == .625
    assert limits['recall_min'] == pytest.approx(2 / 3)
    assert limits['recall_max'] == 1.
