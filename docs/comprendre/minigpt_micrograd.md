# micrograd : l'autograd fait main

D'après Karpathy (2022) · code : `minigpt/micrograd.py` · démo : `uv run python -m minigpt.micrograd`

## 1. À quoi ça sert

Calculer **tout seul les gradients** (de combien la sortie bouge si on pousse un nombre) : le `loss.backward()` de PyTorch refait en **Python pur**, **sans boîte noire**.

## 2. Comment ça marche

- Chaque calcul crée une **Value** : `data`, ses **parents** (`_parents`, un tuple : ordre fixe) et `_backward` (applique la **dérivée locale**) → un **graphe de calcul**.
- `backward()` : **tri topologique** (chaque nœud rangé après ses parents) avec une **pile** (dernier entré, premier sorti), `grad = 1` sur la sortie, puis les `_backward` **à l'envers**.
- `_backward` = **règle de la chaîne**, avec **+=** : un nœud réutilisé **cumule** ses contributions.
- `-`, `/`, `2 + v`… sont écrits avec `+`, `*` et `**` (a / b = a × b⁻¹) : **aucune dérivée en plus**.
- Un **MLP** (couches de neurones) fait de Value : `backward()` donne **tous** ses gradients.

## 3. Équations et chiffres clés

- **Dérivées locales** : a + b → 1 et 1 · a·b → b et a · aᵏ → k·aᵏ⁻¹ · eᵃ → eᵃ · ln a → 1/a · tanh a → 1 − tanh²a · relu a → 1 si a > 0, sinon 0.
- **Chaîne** : dL/dx = dL/dy × dy/dx. **Descente de gradient** : p ← p − lr × dL/dp (**lr** = taille du pas).
- **À la main** (`reuse`, a = −1,5, b = 2) : c = ab + a = −4,5, L = c² + c → dL/da = (2c + 1)(b + 1) = **−24**, dL/db = (2c + 1)·a = **12**, comme PyTorch.
- Écart avec PyTorch (**float64**) : **0** sur les expressions, **1,1·10⁻¹⁶** au pire sur le MLP (seuil 10⁻⁶). **Identique au bit près** à chaque exécution.
- MLP 3 → 4 → 4 → 1 (**41 paramètres**), 50 pas, lr 0,05, **seed** 42 (hasard fixé) : **loss 5,23 → 0,027**.

## 4. « Explique cette ligne »

1. `self.grad += other.data * out.grad` → b (dérivée locale) × `out.grad` ; **+=** cumule.
2. `stack.append((v, True))` → ranger v **plus tard** : ses parents, empilés au-dessus, sont rangés **avant** lui (L = a·b → b, a, L).
3. `for v in reversed(order):` → puis `v._backward()`, **sortie → entrées** : chaque grad est **complet** quand il part.
4. `act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)` → b + Σ wᵢ·xᵢ (`sum` part de b), puis `tanh` (non linéaire).
5. `p.data -= lr * p.grad` → petit pas (**lr**) **contre** le gradient : la loss **baisse**.

## 5. Questions probables du jury

1. **Pourquoi += et pas = ?** Pour b = a + a, db/da = 2 ; avec =, on trouverait 1. Un test le vérifie.
2. **Pourquoi un tri topologique ?** Un nœud doit recevoir **tout** son gradient avant de le transmettre. Une **pile** : en récursif, Python plante vers 1 000 nœuds à la suite (**RecursionError**).
3. **Comment savez-vous que c'est juste ?** 12 tests : mêmes formules dans micrograd et PyTorch, un MLP aux poids copiés, un graphe de 5 000 nœuds.
4. **Pourquoi `zero_grad()` ?** `backward()` cumule : sinon le gradient du pas d'avant s'ajoute au nouveau.
5. **Différence avec PyTorch ?** Même principe, mais sur des **tenseurs** (tableaux de nombres, en C++). micrograd : lent mais lisible.
