"""Tests de micrograd : chaque gradient doit être égal à celui de PyTorch (float64, à 1e-6 près), plus des cas limites."""
import random

import pytest
import torch

from minigpt.micrograd import EXPRESSIONS, MLP, XS, YS, Value, demo_training

torch.set_num_threads(4)    # règle du projet : au plus 4 cœurs CPU pour le bloc 1


def run_micrograd(f, a, b):
    """Valeur de f(a, b) et ses deux gradients, calculés par micrograd."""
    va, vb = Value(a), Value(b)
    out = f(va, vb)
    out.backward()
    return [out.data, va.grad, vb.grad]


def run_torch(f, a, b):
    """La même chose calculée par PyTorch, qui sert de référence."""
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    out = f(ta, tb)
    out.backward()
    return [out.item(), ta.grad.item(), tb.grad.item()]


@pytest.mark.parametrize("f", EXPRESSIONS)
@pytest.mark.parametrize("a, b", [(-1.5, 2.0), (0.5, 3.0)])    # deux points, pour ne pas tomber juste par hasard
def test_expression_matches_torch(f, a, b):
    assert run_micrograd(f, a, b) == pytest.approx(run_torch(f, a, b), abs=1e-6)


def test_reused_node_accumulates():
    a = Value(3.0)
    b = a + a        # a sert deux fois : db/da = 2. Avec = au lieu de += dans _backward, on trouverait 1.
    b.backward()
    assert a.grad == 2.0


def test_deep_graph_backward():
    """sum() de 5 000 Value enchaîne 5 000 additions : backward() ne doit pas buter sur la limite de récursion."""
    xs = [Value(float(i)) for i in range(5000)]
    sum(xs).backward()
    assert all(x.grad == 1.0 for x in xs)


def test_gradients_do_not_depend_on_memory_layout():
    """Mêmes poids, mêmes données : mêmes gradients au bit près, même si les Value sont ailleurs en mémoire."""
    grads, padding = set(), []
    for k in range(20):
        padding.append([object() for _ in range(k)])    # décale les adresses mémoire des Value créées ensuite
        random.seed(0)
        model = MLP(3, [4, 4, 1])
        sum((model(x)[0] - y) ** 2 for x, y in zip(XS, YS)).backward()
        grads.add(tuple(p.grad for p in model.parameters()))
    assert len(grads) == 1    # avec un set pour _parents, les derniers chiffres changeaient d'un essai à l'autre


def test_pow_outside_real_numbers_raises():
    with pytest.raises(ValueError):    # (-2)^0.5 n'est pas un réel : erreur, comme log(-1) (** donnait un complexe)
        Value(-2.0) ** 0.5


def test_mlp_matches_torch():
    """Même réseau, mêmes poids, mêmes données : même loss et mêmes gradients que PyTorch."""
    random.seed(0)    # poids de départ fixés : le test donne toujours le même résultat
    model = MLP(3, [4, 4, 1])
    loss = sum((model(x)[0] - y) ** 2 for x, y in zip(XS, YS))
    loss.backward()

    # Le même réseau en PyTorch : par couche, une matrice W (une ligne par neurone) et un vecteur de biais.
    Ws = [torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True)
          for layer in model.layers]
    bs = [torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True)
          for layer in model.layers]
    h = torch.tensor(XS, dtype=torch.float64)
    for W, bias in zip(Ws, bs):
        # h : une ligne par exemple ; W.T : une colonne par neurone. h @ W.T + bias = w · x + b partout.
        h = torch.tanh(h @ W.T + bias)    # tanh(w · x + b) pour chaque neurone et chacun des 4 exemples
    # h[:, 0] : la sortie (un seul neurone) de chacun des 4 exemples
    torch_loss = ((h[:, 0] - torch.tensor(YS, dtype=torch.float64)) ** 2).sum()
    torch_loss.backward()

    assert loss.data == pytest.approx(torch_loss.item(), abs=1e-6)
    for layer, W, bias in zip(model.layers, Ws, bs):
        for i, n in enumerate(layer.neurons):    # ligne i de W et case i du biais = neurone i
            assert [w.grad for w in n.w] == pytest.approx(W.grad[i].tolist(), abs=1e-6)
            assert n.b.grad == pytest.approx(bias.grad[i].item(), abs=1e-6)


def test_training_reduces_loss():
    losses = demo_training()
    assert losses[-1] < losses[0] / 10    # la démo apprend vraiment : loss divisée par plus de 10 en 50 pas
