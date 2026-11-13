"""Formats de conversation : chacun est le même pour l'affinage et pour le chat.

Deux formats, choisis par la clé "format" de chat_config.json (table FORMATS en bas du fichier) :
- "repliques" (défaut, chatbot Molière) : « NOM.\nréplique\n\n », décrit ci-dessous ;
- "tirets" (chatbot en français moderne) : « — texte\n\n », décrit avant la table FORMATS.

Format « repliques » : les répliques de théâtre.

Pourquoi un format unique ? Un modèle de langue ne fait que continuer un texte qui ressemble à ce
qu'il a lu. Si on l'affine sur des scènes écrites

    HARPAGON.
    Au voleur! au voleur!

    CLÉANTE.
    Qu'avez-vous, mon père?

alors, pour le faire parler, on lui présente exactement la même chose : l'utilisateur devient un
personnage (l'interlocuteur), le modèle écrit la réplique du personnage suivant (le persona).

Règle qui rend le format sans ambiguïté : une réplique ne contient JAMAIS de ligne vide. Une
ligne vide (« \\n\\n ») signifie donc toujours « fin de réplique » : les lignes vides tapées par
l'utilisateur sont retirées. De plus, dans un message de l'utilisateur, une ligne qui ressemble à
un nom de rôle (« HARPAGON. ») est remise en minuscules (« Harpagon. ») : en sortie on coupe sur
une ligne « NOM. », donc en entrée on empêche l'utilisateur d'en écrire une.

Typographie : on impose partout (entraînement ET chat) la convention du texte Molière du kit, celle
de la validation : apostrophe droite ' et pas d'espace avant ! ? ; :. Les pièces de Wikisource
écrivent ’ et « mot ? » ; sans cette normalisation, le modèle copierait la typographie de son
interlocuteur et glisserait vers le style de ces autres sources.
"""

import re
import unicodedata

EOT = "<|endoftext|>"
# Étiquette lisible du format, recopiée dans les rapports. Le cache des tokens dépend en plus de
# l'empreinte de CE fichier (chatbot/format.py) : tout changement ici refait la tokenisation.
FORMAT_VERSION = "repliques-v2"  # v2 : typographie normalisée, espaces Unicode, contrôles retirés

# Un nom de rôle : une capitale, puis capitales, espaces, apostrophes, tirets ou points, et un
# point final (« CLÉANTE. », « MONSIEUR JOURDAIN. », « M. LOYAL. »). Une seule définition, utilisée
# pour couper la sortie du modèle (chatbot.generate) et pour neutraliser la saisie de l'utilisateur.
NAME_CHARS = r"[A-ZÀ-ÖØ-ÞŒ'’ \-.]"
SPEAKER_LINE = re.compile(r"[A-ZÀ-ÖØ-ÞŒ]" + NAME_CHARS + r"{1,60}\.")

# Personnages proposés au chat. Graphies et interlocuteurs vérifiés le 12/11/2026 dans
# data/stages/chatbot_20261112/theatre/moliere_train.jsonl.zst (nom de rôle le plus fréquent et
# personnage qui lui donne le plus souvent la réplique), SAUF HARPAGON et ALCESTE : L'Avare et Le
# Misanthrope (avec Le Sicilien) sont les pièces de validation, absentes de moliere_train. On les
# garde quand même : la recette d'affinage est choisie avec ces 3 pièces tenues à l'écart, puis le
# modèle final de démo est ré-entraîné avec la même recette sur TOUT Molière, ces pièces comprises.
# Pour un modèle affiné SANS ces pièces, HARPAGON et ALCESTE sont des rôles jamais vus (test de
# généralisation, pas de fidélité au personnage).
# MOLIÈRE (persona par défaut, demande du référent) n'a que 99 répliques (L'Impromptu de
# Versailles) : chatbot/finetune.py --persona-augment en fabrique des copies d'entraînement.
PERSONAS = [
    {"name": "MOLIÈRE", "label": "Molière (L'Impromptu de Versailles)", "interlocutor": "LA GRANGE",
     "about": "Molière en personne, chef de troupe, qui dirige la répétition de ses comédiens."},
    {"name": "SGANARELLE", "label": "Sganarelle (Dom Juan)", "interlocutor": "DON JUAN",
     "about": "Le valet bavard et peureux, qui sermonne son maître sans jamais oser lui désobéir."},
    {"name": "HARPAGON", "label": "Harpagon (L'Avare)", "interlocutor": "CLÉANTE",
     "about": "L'avare obsédé par sa cassette, qui soupçonne tout le monde de le voler."},
    {"name": "ALCESTE", "label": "Alceste (Le Misanthrope)", "interlocutor": "PHILINTE",
     "about": "L'homme intransigeant qui hait la flatterie et veut qu'on dise toujours la vérité."},
    {"name": "DON JUAN", "label": "Don Juan (Dom Juan)", "interlocutor": "SGANARELLE",
     "about": "Le séducteur libertin, élégant et cynique, qui ne croit qu'en « deux et deux font quatre »."},
    {"name": "SCAPIN", "label": "Scapin (Les Fourberies de Scapin)", "interlocutor": "ARGANTE",
     "about": "Le valet rusé qui invente des stratagèmes pour soutirer de l'argent aux pères."},
    {"name": "TOINETTE", "label": "Toinette (Le Malade imaginaire)", "interlocutor": "ARGAN",
     "about": "La servante insolente et lucide qui se moque des médecins et de son maître."},
    {"name": "MONSIEUR JOURDAIN", "label": "Monsieur Jourdain (Le Bourgeois gentilhomme)",
     "interlocutor": "MADAME JOURDAIN",
     "about": "Le bourgeois naïf qui veut à tout prix devenir un homme de qualité."},
    {"name": "ARGAN", "label": "Argan (Le Malade imaginaire)", "interlocutor": "TOINETTE",
     "about": "Le malade imaginaire, persuadé d'être mourant, qui ne jure que par ses médecins."},
    {"name": "DORINE", "label": "Dorine (Le Tartuffe)", "interlocutor": "ORGON",
     "about": "La suivante franche et moqueuse qui tient tête à son maître aveuglé par Tartuffe."},
    {"name": "TARTUFFE", "label": "Tartuffe (Le Tartuffe)", "interlocutor": "ELMIRE",
     "about": "Le faux dévot, hypocrite et mielleux, qui cache ses désirs sous de grands mots pieux."},
]
DEFAULT_PERSONA = "MOLIÈRE"


def normalize_speaker(name):
    """Nom de personnage canonique : majuscules, espaces simples, apostrophe droite, sans point final."""
    name = re.sub(r"\s+", " ", str(name).replace("’", "'")).strip().rstrip(".").strip()
    if not name:
        raise ValueError("nom de personnage vide")
    return name.upper()


def clean_text(text):
    """Nettoyage d'un texte de réplique, sans l'interpréter (entraînement ET chat).

    - `<|endoftext|>` tapé à la main serait lu comme le vrai séparateur de documents : on le retire ;
    - toute espace Unicode (tabulation, insécable, fine insécable, U+2028...) -> une espace simple ;
    - caractères invisibles (contrôle « Cc » et format « Cf », p. ex. l'espace sans largeur) retirés :
      les données d'affinage n'en contiennent pas, le modèle ne les a jamais vus ;
    - typographie du kit Molière : ’ -> ', « ſ » (s long) -> s, pas d'espace avant ! ? ; : ;
    - lignes vides retirées : c'est ce qui garantit qu'aucune réplique ne peut en fabriquer une autre.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # \s couvre toutes les espaces Unicode ; [^\S\n] = « une espace qui n'est pas un saut de ligne ».
    # Avant le retrait des contrôles : \t ou U+0085 sont des « Cc » mais séparent des mots.
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "".join(c for c in text if c == "\n" or unicodedata.category(c) not in ("Cc", "Cf"))
    # <|endoftext|> APRÈS le retrait des invisibles : sinon « <|endo\u200bftext|> » passerait le
    # test, puis le retrait du \u200b reformerait le vrai séparateur.
    text = text.replace(EOT, " ").replace("’", "'").replace("ſ", "s")
    # Les retraits ci-dessus peuvent laisser deux espaces côte à côte : on les fusionne.
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([!?;:])", r"\1", text)
    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def render_turn(speaker, text):
    """Une réplique : « NOM.\\n texte \\n\\n ». La ligne vide finale marque la fin de la réplique."""
    return f"{normalize_speaker(speaker)}.\n{clean_text(text)}\n\n"


def render_scene(turns):
    """Une scène = la suite de ses répliques (liste de {"speaker", "text"})."""
    return "".join(render_turn(t["speaker"], t["text"]) for t in turns)


def parse_scene(text):
    """Inverse de render_scene (utile aux tests) : découpe sur les lignes vides."""
    turns = []
    for block in text.split("\n\n"):
        if block.strip():
            head, _, body = block.partition("\n")
            turns.append({"speaker": head.rstrip("."), "text": body})
    return turns


def persona_info(persona):
    """Fiche du persona demandé. Un nom hors liste est refusé (ValueError).

    On ne fait parler que des personnages de la liste : sinon une faute (« MOLIERE », « DOM JUAN »)
    ferait improviser un rôle jamais vu, sans aucun message.
    """
    name = normalize_speaker(persona)
    for p in PERSONAS:
        if p["name"] == name:
            return p
    noms = ", ".join(p["name"] for p in PERSONAS)
    raise ValueError(f"personnage inconnu : {name} (choisir parmi : {noms})")


def check_messages(messages):
    """Liste alternée de messages user/assistant qui finit par un message user non vide."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("la conversation doit contenir au moins un message")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise ValueError("chaque message doit avoir role = 'user' ou 'assistant'")
        if not isinstance(m.get("content"), str):
            raise ValueError("le contenu d'un message doit être un texte")
        if i and m["role"] == messages[i - 1]["role"]:
            raise ValueError("les messages doivent alterner user et assistant")
    if messages[-1]["role"] != "user" or not clean_text(messages[-1]["content"]):
        raise ValueError("la conversation doit finir par un message utilisateur non vide")


def neutralize_names(text):
    """Une ligne de l'utilisateur qui ressemble à un nom de rôle passe en minuscules.

    « HARPAGON. » -> « Harpagon. », « MERCI. » -> « Merci. » : le modèle ne la prend plus pour un
    changement de personnage. Seulement côté chat ; le texte d'entraînement n'est pas touché.
    """
    return "\n".join(_neutral_line(line) for line in text.split("\n"))


def _neutral_line(line):
    """« HARPAGON. » -> « Harpagon. » ; si capitalize() ne change rien (« A -. », une seule lettre),
    toute la ligne passe en minuscules : un nom de rôle commence forcément par une capitale."""
    if not SPEAKER_LINE.fullmatch(line):
        return line
    line = line.capitalize()
    return line.lower() if SPEAKER_LINE.fullmatch(line) else line


def render_messages(messages, persona, interlocutor):
    """Une réplique rendue par message : user -> interlocuteur, assistant -> persona."""
    check_messages(messages)
    out = []
    for m in messages:
        speaker = interlocutor if m["role"] == "user" else persona
        # Aucun message (même une ancienne réponse renvoyée par le client) ne doit contenir une
        # ligne « NOM. » : elle fabriquerait un faux changement de personnage.
        text = neutralize_names(clean_text(m["content"]))
        # Une réplique vide casserait le format (ligne vide juste après le nom) : on met « … ».
        out.append(render_turn(speaker, text or "…"))
    return out


def persona_header(persona):
    """Début de la réplique que le modèle doit écrire : « PERSONA.\\n »."""
    return f"{normalize_speaker(persona)}.\n"


def build_prompt(messages, persona, interlocutor):
    """Prompt complet : la conversation mise en scène, puis le nom du persona (le modèle continue)."""
    return "".join(render_messages(messages, persona, interlocutor)) + persona_header(persona)


# ---------------------------------------------------------------------------------------------
# Format « tirets » : le dialogue de roman, pour le chatbot en français moderne.
#
#     <|endoftext|>— Vous allez vous battre?
#
#     — Non, je vais dîner.
#
# Chaque tour de parole est UN paragraphe qui commence par « — » + espace, suivi d'une ligne vide.
# Deux interlocuteurs sans nom (A puis B) alternent. Pourquoi sans nom ? Dans un roman, on sait qui
# parle par l'alternance des tirets, pas par une étiquette : c'est la forme la plus courante des
# dialogues en français moderne. Au chat : l'utilisateur et le bot alternent de la même façon, le
# prompt finit par « — » + espace, le modèle écrit la suite et s'arrête à la fin du paragraphe.
#
# Règle qui rend ce format sans ambiguïté : un tour ne contient AUCUN retour à la ligne. Un message
# de plusieurs lignes devient un seul paragraphe (lignes jointes par une espace), et un tiret en tête
# de ligne y est retiré : l'utilisateur ne peut pas fabriquer un faux tour.

FORMAT_VERSION_TIRETS = "tirets-v1"
DASH = "—"
DASH_HEADER = DASH + " "  # début de chaque tour, donc fin du prompt du chat
DASH_SPEAKERS = ("A", "B")  # contrat des fichiers de conversations : A commence, puis alternance
# Tiret de dialogue en tête de ligne, éventuellement répété : « — », « – », « ― », « ‒ », « -- »
# (vieux romans : « --Vous allez vous battre? ») ou « - » suivi d'une espace. Un « -5 » est gardé.
LEADING_DASH = re.compile(r"^(?:(?:[—–―‒]|-{2,}|-(?=\s|$))\s*)+")


def dash_text(text):
    """Texte d'un tour de parole au format tirets (entraînement ET chat).

    - même nettoyage que les répliques (clean_text : <|endoftext|>, espaces Unicode, typographie) ;
    - un tiret de dialogue en tête de ligne est retiré (sinon « — — texte » ou un faux tour) ;
    - les lignes sont jointes par une espace : un tour = un paragraphe, sans retour à la ligne.
    Le second clean_text recolle la ponctuation d'une ligne qui commençait par « ? » ou « ! ».
    """
    lines = (LEADING_DASH.sub("", line).strip() for line in clean_text(text).split("\n"))
    return clean_text(" ".join(line for line in lines if line))


def render_dash_turn(text):
    """Un tour : « — texte\n\n ». Un tour vide devient « … », comme une réplique vide au chat."""
    return f"{DASH_HEADER}{dash_text(text) or '…'}\n\n"


def render_dash_exchange(turns):
    """Un échange (contrat : speaker "A"/"B" en alternance stricte, A commence) -> texte.

    On vérifie l'alternance : un fichier mal construit est refusé, pas appris de travers.
    """
    for i, t in enumerate(turns):
        if t.get("speaker") != DASH_SPEAKERS[i % 2]:
            raise ValueError(f"tour {i} : speaker {t.get('speaker')!r}, attendu "
                             f"{DASH_SPEAKERS[i % 2]!r} (A et B alternent, A commence)")
    return "".join(render_dash_turn(t["text"]) for t in turns)


def render_dash_messages(messages, persona=None, interlocutor=None):
    """Un tour rendu par message (utilisateur et bot alternent). persona/interlocutor : ignorés,
    présents seulement pour avoir la même signature que render_messages (table FORMATS)."""
    check_messages(messages)
    return [render_dash_turn(m["content"]) for m in messages]


def build_dash_prompt(messages):
    """Prompt complet du chat en tirets : les tours, puis « — » + espace (le modèle continue)."""
    return "".join(render_dash_messages(messages)) + DASH_HEADER


# ---------------------------------------------------------------------------------------------
# Table des formats : finetune, generate et evaluate lisent ici ce qui change d'un format à
# l'autre, au lieu d'écrire des « if format == ... » partout.
#
# - version : étiquette recopiée dans report.json, la méta des checkpoints et la clé du cache ;
# - render_exchange(turns) : texte d'entraînement d'une scène / d'un échange ;
# - render_messages(messages, persona, interlocutor) : un texte par message du chat ;
# - header(persona) : fin du prompt, début du tour que le modèle doit écrire ;
# - reply_lead : partie de header que le modèle écrit lui-même, en tokens. Le tokenizer colle
#   l'espace au mot suivant (« — Bonjour » = « — » + « ␣Bon » + « jour ») : un prompt qui finirait
#   par le token « ␣ » seul ne ressemblerait à rien de vu à l'entraînement. Le texte du prompt
#   finit bien par « — » + espace, mais on le tokenise sans cette espace finale ;
# - user_frame(interlocutor) : un tour utilisateur sans son texte (tokens fixes d'une coupe) ;
# - personas : le format a-t-il des personnages (noms de rôle) ?

FORMATS = {
    "repliques": {"version": FORMAT_VERSION, "render_exchange": render_scene,
                  "render_messages": render_messages, "header": persona_header, "reply_lead": "",
                  "user_frame": lambda interlocutor: f"{interlocutor}.\n\n\n", "personas": True},
    "tirets": {"version": FORMAT_VERSION_TIRETS, "render_exchange": render_dash_exchange,
               "render_messages": render_dash_messages, "header": lambda persona=None: DASH_HEADER,
               "reply_lead": " ", "user_frame": lambda interlocutor=None: f"{DASH_HEADER}\n\n",
               "personas": False},
}
DEFAULT_FORMAT = "repliques"


def get_format(name):
    """Fiche du format demandé ; un nom inconnu est refusé (ValueError)."""
    if name not in FORMATS:
        raise ValueError(f"format inconnu : {name!r} (choisir parmi : {', '.join(FORMATS)})")
    return FORMATS[name]


def format_of_version(version):
    """Nom du format d'après l'étiquette d'un checkpoint (« tirets-v1 » -> « tirets »), ou None."""
    for name, f in FORMATS.items():
        if version and str(version).split("-")[0] == f["version"].split("-")[0]:
            return name
    return None


def token_header(fmt, persona=None):
    """Fin du prompt telle qu'on la tokenise : header sans la partie écrite par le modèle."""
    head = fmt["header"](persona)
    return head[:len(head) - len(fmt["reply_lead"])]
