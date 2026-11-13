"""Tout ce qui dépend du matériel, au même endroit (carte NVIDIA, Mac, processeur, Windows).

Pourquoi un module à part ? pretrain.py et finetune.py ne doivent pas se remplir de « si CUDA
alors… ». Ici : choix du device, précision du calcul, transfert des lots, compilation, description
de la machine pour les rapports, et deux soucis propres à Windows (console, remplacement de
fichier). Sur Mac et sur CPU, le réglage par défaut (« auto ») donne EXACTEMENT le calcul d'avant :
float32 partout, aucun contexte d'autocast, aucun GradScaler.

LA PRÉCISION MIXTE, en bref
- Un nombre float32 occupe 4 octets ; bfloat16 et float16 en occupent 2. Les cœurs « Tensor » d'une
  carte NVIDIA multiplient des matrices de 16 bits bien plus vite qu'en 32 bits, et la mémoire
  transférée est divisée par deux : c'est le gros du gain sur la RTX 5090.
- « Mixte » : les POIDS restent en float32 (la copie de référence, mise à jour par AdamW) ; seul le
  calcul de la passe avant se fait en 16 bits, à l'intérieur d'un contexte torch.autocast qui
  convertit au vol les entrées des multiplications de matrices.
- La loss reste calculée en float32 : ftb.model fait cross_entropy(logits.float(), …). Pourquoi ?
  La loss somme le logarithme de 8 192 probabilités (softmax) : en 16 bits, les petites
  probabilités s'arrondissent à 0 (log = -inf) et la moyenne perd ses derniers chiffres. Le
  softmax/log en float32 coûte peu et évite ces erreurs ; c'est aussi ce que fait le kit.
- bfloat16 a la même PLAGE que float32 (8 bits d'exposant), avec moins de chiffres : pas de
  débordement, donc pas besoin de GradScaler. C'est le choix par défaut sur une carte récente.
- float16 a plus de chiffres mais une plage étroite (max 65 504, rien sous ~6e-8) : de petits
  gradients deviendraient 0. Le GradScaler multiplie la loss par un grand facteur avant
  backward(), puis divise les gradients d'autant avant le pas d'AdamW ; s'il voit un gradient
  infini, il saute ce pas et baisse le facteur. Seulement pour les cartes sans bfloat16 natif.
- TF32 (cartes NVIDIA récentes) : les multiplications float32 restantes passent par les cœurs
  Tensor avec une mantisse de 10 bits. Activé seulement sur CUDA.
"""

import contextlib
import os
import platform
import signal
import sys
import time

import numpy as np
import torch

from ftb.evaluate import default_device

PRECISIONS = ("auto", "bf16", "fp16", "fp32")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def pick_device(requested="auto"):
    """auto : cuda > mps > cpu (même ordre que le kit, ftb.evaluate.default_device)."""
    return default_device() if requested in (None, "auto") else requested


def device_kind(device):
    """« cuda:1 » -> « cuda » : le type attendu par torch.autocast et GradScaler."""
    return str(device).split(":")[0]


def bf16_native():
    """Vrai si la carte calcule en bfloat16 en matériel (NVIDIA Ampere et plus récentes).

    including_emulation=False : sinon une T4 répondrait « oui » par émulation, lente.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # ancienne version de PyTorch, sans ce paramètre
        return torch.cuda.is_bf16_supported()


def resolve_precision(requested, device):
    """Précision effective : auto -> bf16 (ou fp16) sur CUDA, fp32 sur mps et cpu.

    Un choix explicite (bf16/fp16 sur cpu, par exemple pour un test) est respecté.
    """
    if requested not in PRECISIONS:
        sys.exit(f"ERREUR : --precision {requested!r} : choisir parmi {', '.join(PRECISIONS)}")
    kind = device_kind(device)
    if requested == "auto":
        if kind == "cuda":
            return "bf16" if bf16_native() else "fp16"
        return "fp32"  # Mac et processeur : le calcul d'avant, à l'identique
    if requested == "bf16" and kind == "cuda" and not bf16_native():
        sys.exit("ERREUR : cette carte ne calcule pas en bfloat16 : utiliser --precision fp16")
    return requested


def enable_tf32():
    """CUDA seulement : multiplications float32 restantes en TF32 (plus rapides, quasi exactes).

    Jamais appelé sur cpu ou mps : sur processeur ce réglage global pourrait changer le calcul.
    """
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True


class SafeForward:
    """Appelle le modèle compilé ; s'il échoue, message clair puis repli sur le modèle normal.

    torch.compile est paresseux : la vraie compilation n'a lieu qu'au premier appel (et à chaque
    nouvelle forme de tenseur). Une erreur peut donc arriver tard : on l'attrape ici aussi.
    """

    def __init__(self, compiled, eager, accel):
        self.fn, self.eager, self.accel = compiled, eager, accel

    def __call__(self, *args, **kwargs):
        if self.fn is self.eager:
            return self.eager(*args, **kwargs)
        try:
            return self.fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - toute erreur de compilation -> repli
            self.accel.compile_failed(e)
            self.fn = self.eager
            return self.eager(*args, **kwargs)


class Accel:
    """Réglages matériels d'un entraînement : device, précision, GradScaler, compilation.

    Accel("cpu") ou Accel("mps") avec precision="auto" : float32, aucun autocast, aucun
    GradScaler, aucune copie en mémoire épinglée -> exactement le calcul d'avant ce module.
    """

    def __init__(self, device="cpu", precision="auto", compile=False):
        self.device = pick_device(device)
        self.kind = device_kind(self.device)
        self.precision = resolve_precision(precision, self.device)
        self.dtype = DTYPES[self.precision]
        self.scaler = None
        if self.precision == "fp16":
            try:
                self.scaler = torch.amp.GradScaler(self.kind)
            except Exception as e:  # noqa: BLE001
                sys.exit(f"ERREUR : GradScaler indisponible sur {self.kind} ({e}) : "
                         f"utiliser --precision bf16 ou fp32")
        self.pin = self.kind == "cuda"  # mémoire épinglée : copie CPU -> GPU asynchrone
        # TF32 sur CUDA, sauf --precision fp32 demandé explicitement : « fp32 » veut dire du vrai
        # float32 partout (calcul de référence, comparable au Mac). Jamais sur cpu ni mps.
        self.tf32 = self.kind == "cuda" and precision != "fp32"
        if self.tf32:
            enable_tf32()
        self.compile_requested = bool(compile)
        self.compile_status = "demandé" if compile else "désactivé"
        self.compile_error = None

    # --- passe avant et transfert des lots ---------------------------------------------------

    def autocast(self):
        """Contexte de la passe avant (entraînement ET évaluation).

        fp32 : contexte vide (nullcontext), pas même un autocast désactivé : rien ne change.
        """
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.kind, dtype=self.dtype)

    def host(self, batch):
        """Lot encore en mémoire centrale : épinglé sur CUDA (le GPU peut le copier seul)."""
        return batch.pin_memory() if self.pin else batch

    def to_device(self, batch):
        """Envoi au device. non_blocking : la copie se fait pendant que le GPU calcule encore."""
        if self.pin:
            return batch.to(self.device, non_blocking=True)
        return batch.to(self.device)

    # --- rétropropagation et pas d'optimisation ---------------------------------------------

    def backward(self, loss):
        (self.scaler.scale(loss) if self.scaler else loss).backward()

    def unscale(self, opt):
        """À faire AVANT de couper la norme du gradient : sinon on couperait des gradients ×65 536."""
        if self.scaler:
            self.scaler.unscale_(opt)

    def step(self, opt):
        """Pas d'AdamW. Avec GradScaler : sauté si un gradient est infini, puis facteur ajusté."""
        if self.scaler:
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.step()

    def scaler_state(self):
        return self.scaler.state_dict() if self.scaler else None

    def load_scaler_state(self, state):
        if self.scaler is not None and state:
            self.scaler.load_state_dict(state)

    # --- compilation -------------------------------------------------------------------------

    def compile_failed(self, error):
        self.compile_status = "échec, repli sans compilation"
        self.compile_error = f"{type(error).__name__}: {str(error).splitlines()[0][:300] if str(error) else ''}"
        print(f"--compile : torch.compile a échoué ({self.compile_error}). On continue SANS "
              f"compilation : même calcul, seulement plus lent.", file=sys.stderr, flush=True)

    def compile(self, model, sample_shape=None):
        """Fonction à appeler pour la passe avant d'entraînement (le modèle lui-même par défaut).

        Avec --compile : torch.compile(model), puis un essai (passe avant + arrière sur un lot de
        zéros de la forme réelle, gradients jetés, générateurs aléatoires remis en l'état) pour
        compiler TOUT DE SUITE et attraper une erreur avant le premier vrai pas. En cas d'échec :
        message, puis le modèle non compilé. Les poids restent ceux de `model` (partagés) :
        sauvegarde, évaluation et checkpoints utilisent toujours `model`.
        """
        if not self.compile_requested:
            return model
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if self.kind == "cuda" else None
        try:
            compiled = torch.compile(model)
            if sample_shape is not None:
                x = torch.zeros(sample_shape, dtype=torch.long, device=self.device)
                with self.autocast():
                    _, loss = compiled(x, x)
                loss.backward()
        except Exception as e:  # noqa: BLE001
            self.compile_failed(e)
            return model
        finally:
            for p in model.parameters():
                p.grad = None  # gradients de l'essai jetés : le vrai pas 1 part de zéro
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.compile_status = "actif"
        return SafeForward(compiled, model, self)

    # --- mesures et description --------------------------------------------------------------

    def sync(self):
        """Attendre la fin des calculs du GPU (sinon un chronomètre mesure seulement l'envoi)."""
        if self.kind == "cuda":
            torch.cuda.synchronize()
        elif self.kind == "mps":
            torch.mps.synchronize()

    def peak_memory_gb(self):
        """Mémoire GPU maximale occupée depuis le début (CUDA seulement), en Go."""
        if self.kind != "cuda":
            return None
        return round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2)

    def describe(self):
        """Machine et réglages, recopiés dans report.json et la méta des checkpoints.

        Volontairement sans nom de machine ni chemin personnel.
        """
        info = {"device": self.device, "precision": self.precision,
                "autocast_dtype": None if self.precision == "fp32" else str(self.dtype),
                "grad_scaler": self.scaler is not None, "tf32": self.tf32,
                "pin_memory": self.pin,
                "compile": {"requested": self.compile_requested, "status": self.compile_status,
                            "error": self.compile_error},
                # str() : torch.__version__ est un objet TorchVersion, que torch.load(...,
                # weights_only=True) refuserait de relire dans la méta d'un checkpoint.
                "torch": str(torch.__version__),
                "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
                "python": platform.python_version(), "numpy": np.__version__,
                "os": f"{platform.system()} {platform.release()}", "machine": platform.machine()}
        if self.kind == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            info["gpu"] = {"name": str(props.name), "memory_gb": round(props.total_memory / 1e9, 1),
                           "capability": f"{props.major}.{props.minor}",
                           "count": torch.cuda.device_count(), "bf16_native": bf16_native(),
                           "cudnn": torch.backends.cudnn.version()}  # int ou None
        elif self.kind == "mps":
            info["gpu"] = {"name": "Apple GPU (MPS)"}
        return info

    def summary(self):
        """Une ligne pour la console."""
        gpu = self.describe().get("gpu", {})
        name = f" ({gpu['name']}, {gpu.get('memory_gb', '?')} Go)" if gpu.get("memory_gb") else ""
        extra = " + GradScaler" if self.scaler else ""
        return (f"device {self.device}{name} | précision {self.precision}{extra} | "
                f"compile {self.compile_status}")


# ---------------------------------------------------------------------------------------------
# Windows


def setup_console():
    """Sorties console en UTF-8, sans jamais planter.

    Sous Windows, quand la sortie est redirigée (fichier, tube, session Claude Code), Python écrit
    dans l'encodage local (souvent cp1252), qui n'a pas « ≈ », « ≠ », « ∝ » : print() lèverait
    UnicodeEncodeError en plein entraînement. On repasse stdout et stderr en UTF-8 ; un caractère
    impossible devient un code \\uXXXX au lieu d'une erreur. Déjà en UTF-8 (Mac, Linux) : rien.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if enc == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError, AttributeError):
            pass  # flux exotique : on le laisse tel quel


class StopRequest:
    """Demande d'arrêt (Ctrl-C, SIGTERM, Ctrl-Break) : on finit le pas en cours, on sauve state.pt.

    Couper au milieu de opt.step() laisserait des poids à moitié mis à jour : on attend la fin
    du pas. Un deuxième Ctrl-C coupe tout de suite (comportement normal de Python).
    Windows : SIGTERM existe dans Python mais aucun autre programme ne peut l'envoyer (« tuer »
    un processus Windows l'arrête sans prévenir). On écoute donc aussi SIGBREAK = Ctrl-Break
    (ou Ctrl-Pause) dans la console. Pour arrêter proprement sur le PC : Ctrl-C, ou --stop-at-step.
    """

    SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGBREAK")
                    if hasattr(signal, name))

    def __init__(self):
        self.requested = None
        self.previous = {}

    def __enter__(self):
        for sig in self.SIGNALS:
            self.previous[sig] = signal.signal(sig, self.handler)
        return self

    def handler(self, signum, frame):
        self.requested = signal.Signals(signum).name
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            print(f"\n{self.requested} reçu : fin du pas en cours, puis sauvegarde de state.pt",
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            # La sortie est un tube vers un lanceur déjà arrêté (tube cassé) : écrire lèverait une
            # erreur en pleine boucle et le processus mourrait SANS sauver. On jette donc tous les
            # messages suivants : la sauvegarde de state.pt passe avant l'affichage.
            sink = open(os.devnull, "w", encoding="utf-8")
            sys.stdout = sys.stderr = sink

    def __exit__(self, *exc):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def replace_file(tmp, dst, tries=20, wait=0.25):
    """os.replace(tmp, dst) avec quelques nouvelles tentatives.

    Sous Windows, remplacer un fichier OUVERT ailleurs (antivirus qui scanne le checkpoint tout
    juste écrit, démo qui lit best.pt, entraînement qui lit un .bin par np.memmap) lève
    PermissionError, alors que macOS et Linux l'acceptent. Un verrou d'antivirus dure quelques
    dixièmes de seconde : on réessaie ~5 s, puis on explique. Sur Mac/Linux : un seul essai.
    """
    for attempt in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise PermissionError(
                    f"impossible de remplacer {dst} : fichier ouvert par un autre programme "
                    f"(démo, autre entraînement, antivirus ?). Le nouveau contenu est dans {tmp}.")
            time.sleep(wait)
