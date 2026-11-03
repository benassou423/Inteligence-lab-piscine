"""Mini-autograd « micrograd » (d'après Karpathy) : des nombres qui calculent tout seuls leur gradient.

Chaque Value retient de quels nombres elle vient et par quelle opération.
backward() parcourt ce graphe de la sortie vers les entrées en appliquant la règle de la chaîne.

Démo : uv run python -m minigpt.micrograd
"""
import math
import random


class Value:
    """Un nombre du graphe de calcul : sa valeur (data) et la dérivée de la sortie finale par rapport à lui (grad)."""

    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self.grad = 0.0                  # rempli par backward() ; 0 tant que rien n'a été propagé
        self._backward = lambda: None    # transmet grad aux parents ; rien à faire par défaut (Value sans parents)
        # Les parents : les Value qui ont servi à calculer celle-ci. Un tuple et pas un set : l'ordre d'un set
        # dépend des adresses mémoire, or en virgule flottante l'ordre des additions change les derniers chiffres.
        self._parents = _parents
        self._op = _op                   # l'opération qui a créé ce nœud, pour l'affichage

    def __repr__(self):
        """Ce que print(v) affiche, ex. Value(data=6, grad=1, op='*') (4 chiffres significatifs)."""
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g}, op='{self._op}')"

    # --- Opérations de base : chacune calcule le résultat ET range sa dérivée locale dans _backward ---
    # Règle de la chaîne : grad d'un parent += (dérivée locale) × out.grad, out étant le résultat de l'opération.
    # += et pas = : un nœud utilisé plusieurs fois reçoit la somme de toutes ses contributions.

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)    # un nombre devient une Value : a + 2 marche
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():    # appelée plus tard par backward() ; elle se souvient de self, other et out (closure)
            self.grad += out.grad     # d(a+b)/da = 1 : le gradient passe tel quel
            other.grad += out.grad    # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad    # d(a·b)/da = b
            other.grad += self.data * out.grad    # d(a·b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "exposant constant seulement (int ou float)"
        # math.pow et pas ** : (-2) ** 0.5 donnerait un nombre complexe ; math.pow lève une erreur, comme log(-1).
        out = Value(math.pow(self.data, k), (self,), f'**{k}')

        def _backward():
            self.grad += k * math.pow(self.data, k - 1) * out.grad    # d(a^k)/da = k·a^(k-1)
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad    # d(e^a)/da = e^a, déjà calculé : c'est out.data
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data    # d(ln a)/da = 1/a
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad    # d(tanh a)/da = 1 - tanh(a)²
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            slope = 1.0 if self.data > 0 else 0.0    # pente 1 si a > 0, sinon 0 (en 0 aussi, comme PyTorch)
            self.grad += slope * out.grad
        out._backward = _backward
        return out

    # --- Opérations composées : écrites avec celles du dessus, donc aucune dérivée à écrire en plus ---

    def __neg__(self):              # -a = a × (-1)
        return self * -1

    def __sub__(self, other):       # a - b = a + (-b)
        return self + (-other)

    def __truediv__(self, other):   # a / b = a × b^(-1)
        return self * (other ** -1)

    # Versions inversées, pour 2 + a, 2 * a, 2 - a, 2 / a : le nombre 2 ne sait pas calculer avec
    # une Value, alors Python demande à la Value. Exemple : 2 - a appelle a.__rsub__(2).

    def __radd__(self, other):      # 2 + a = a + 2
        return self + other

    def __rmul__(self, other):      # 2 * a = a × 2
        return self * other

    def __rsub__(self, other):      # 2 - a = (-a) + 2
        return (-self) + other

    def __rtruediv__(self, other):  # 2 / a = a^(-1) × 2
        return (self ** -1) * other

    def backward(self):
        """Rétropropagation : ajoute d(self)/dv au grad de chaque Value v dont self dépend.

        Un appel par graphe (on le reconstruit à chaque pas) : un 2e appel partirait des grad déjà
        remplis des nœuds intermédiaires et donnerait des gradients faux.
        """
        # 1) Tri topologique : chaque nœud est rangé APRÈS tous ses parents. Avec une pile et pas une
        #    fonction récursive : Python l'arrête vers 1 000 appels imbriqués (RecursionError).
        #    Ex. L = a * b : on dépile L, on rempile (L, True) puis a et b ; b puis a sont rangés, enfin L.
        order, visited = [], set()                 # visited : on y teste « déjà vu ? », son ordre ne joue pas
        stack = [(self, False)]                   # (nœud, parents déjà rangés ?) ; pop() sort le dernier ajouté
        while stack:
            v, parents_done = stack.pop()
            if parents_done:
                order.append(v)                    # ses parents ont été dépilés et rangés avant lui : on le range
            elif v not in visited:                 # un nœud partagé n'est traité qu'une fois
                visited.add(v)
                stack.append((v, True))            # on reviendra le ranger...
                for parent in v._parents:
                    stack.append((parent, False))  # ... après ses parents, empilés au-dessus de lui

        # 2) d(self)/d(self) = 1, puis on remonte de la sortie vers les entrées. À l'envers, tous les nœuds
        #    qui utilisent v passent avant v : son grad est donc complet au moment où il le transmet.
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


# --- Un petit réseau de neurones fait de Value ---

class Neuron:
    """Un neurone : tanh(w · x + b)."""

    def __init__(self, n_inputs):
        # Poids au hasard : s'ils partaient tous égaux, les neurones d'une couche resteraient identiques.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)    # on part de b et on ajoute chaque wi·xi
        return act.tanh()    # non-linéarité : sans elle, empiler des couches ne donnerait qu'une fonction affine

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Une couche : n_outputs neurones qui lisent tous la même entrée."""

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]    # les paramètres de chaque neurone, bout à bout


class MLP:
    """Perceptron multicouche : la sortie de chaque couche est l'entrée de la suivante."""

    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes    # ex. 3 entrées et [4, 4, 1] → couches 3→4, 4→4, 4→1
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # backward() ACCUMULE (+=) : sans remise à zéro, les gradients du pas précédent s'ajouteraient.
        for p in self.parameters():
            p.grad = 0.0


# --- Démo : uv run python -m minigpt.micrograd ---

def reuse(a, b):
    """Nœuds réutilisés : a sert 2 fois et c 3 fois, leurs gradients doivent s'additionner."""
    c = a * b + a
    return c * c + c


def arithmetic(a, b):
    """Puissances (entière et réelle), division, soustraction, moins unaire, et nombre + v, * v, - v, / v."""
    c = a ** 3 / b - (b - a) ** 0.5
    return 2 - 3 * c + 1 / (2 + a) - (-b)


def functions(a, b):
    """exp, log, tanh, et relu des deux côtés de 0."""
    return (a.exp() + b.log()).tanh() + (a * b).relu() + (b - a).relu()


# Ces formules marchent telles quelles sur des Value ET sur des tenseurs PyTorch :
# on compare donc la même formule, écrite une seule fois, dans les deux moteurs (démo et tests).
EXPRESSIONS = [reuse, arithmetic, functions]

# Le jeu jouet de la vidéo de Karpathy : 4 exemples de 3 nombres, chacun avec sa cible +1 ou -1.
XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]


def demo_gradients():
    """Affiche côte à côte les gradients de micrograd et de PyTorch sur les 3 expressions."""
    import torch                    # seulement pour vérifier : Value n'utilise que math et random
    torch.set_num_threads(4)        # règle du projet : au plus 4 cœurs CPU pour le bloc 1

    a, b = -1.5, 2.0
    print(f"Gradients en a = {a}, b = {b} : micrograd vs PyTorch (float64)")
    print(f"{'expression':<12}{'var':<5}{'micrograd':>13}{'PyTorch':>13}{'écart':>10}")
    for f in EXPRESSIONS:
        va, vb = Value(a), Value(b)
        f(va, vb).backward()
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        f(ta, tb).backward()
        for name, v, t in [("a", va, ta), ("b", vb, tb)]:
            gap = abs(v.grad - t.grad.item())
            print(f"{f.__name__:<12}{name:<5}{v.grad:>13.6f}{t.grad.item():>13.6f}{gap:>10.1e}")


def demo_training():
    """Entraîne un MLP 3 → 4 → 4 → 1 sur les 4 points et renvoie la loss de chaque pas."""
    random.seed(42)                 # mêmes poids de départ à chaque exécution
    model = MLP(3, [4, 4, 1])
    steps, lr = 50, 0.05
    print(f"\nEntraînement : MLP 3 → 4 → 4 → 1 ({len(model.parameters())} paramètres), {steps} pas, lr = {lr}")
    losses = []
    for step in range(steps):
        preds = [model(x)[0] for x in XS]                      # [0] : la couche de sortie a un seul neurone
        loss = sum((yp - y) ** 2 for yp, y in zip(preds, YS))  # erreur quadratique ; sum part de 0, d'où __radd__
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad                              # descente de gradient : un petit pas contre la pente
        losses.append(loss.data)
        if step % 10 == 0 or step == steps - 1:
            print(f"pas {step:2d}   loss {loss.data:.4f}")
    print("prédictions :", [round(yp.data, 2) for yp in preds], "  cibles :", YS)
    return losses


if __name__ == "__main__":
    demo_gradients()
    demo_training()
