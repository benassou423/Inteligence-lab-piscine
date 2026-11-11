"""Rapport de deux pages et datasheet, à partir des résultats réellement disponibles."""
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
DOC = ROOT / 'docs/remise'
BLUE, TEAL, ORANGE = '#102A43', '#087E8B', '#BD4F28'
LABELS = {'wikipedia_fr': 'Wikipédia', 'wikisource_fr': 'Wikisource', 'gutenberg_fr': 'Gutenberg',
          'wikinews_fr': 'Wikinews', 'assemblee_nationale': 'Assemblée nationale',
          'fineweb2_fra': 'FineWeb-2', 'commoncrawl': 'Common Crawl'}


def read(path):
    return json.loads((ROOT / path).read_text())


def number(x, digits=0):
    return f'{x:,.{digits}f}'.replace(',', ' ').replace('.', ',')


FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
BOLD = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
pdfmetrics.registerFont(TTFont('Body', str(FONT)))
pdfmetrics.registerFont(TTFont('Strong', str(BOLD)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Strong')
STYLES = {
    'title': ParagraphStyle('Title', fontName='Strong', fontSize=23, leading=27, textColor=colors.HexColor(BLUE), spaceAfter=10),
    'sub': ParagraphStyle('Sub', fontName='Body', fontSize=10, leading=14, textColor=colors.HexColor(TEAL), spaceAfter=12),
    'h': ParagraphStyle('Heading', fontName='Strong', fontSize=12, leading=16, textColor=colors.HexColor(BLUE), spaceBefore=11, spaceAfter=5, keepWithNext=True),
    'p': ParagraphStyle('Body', fontName='Body', fontSize=9.4, leading=13.1, spaceAfter=7),
    'small': ParagraphStyle('Small', fontName='Body', fontSize=7.5, leading=10.2, spaceAfter=5, textColor=colors.HexColor('#425466')),
    'cell': ParagraphStyle('Cell', fontName='Body', fontSize=8.8, leading=12),
}


def para(text, kind='p'):
    return Paragraph(text, STYLES[kind])


def table(rows, widths):
    cells = [[para(escape(str(v)), 'cell') for v in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7F1F2')),
                          ('FONTNAME', (0, 0), (-1, 0), 'Strong'), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                          ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                          ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                          ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor(TEAL)),
                          ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#D9E2EC'))]))
    return t


def footer(canvas, doc):
    canvas.setFont('Body', 7.5)
    canvas.setFillColor(colors.HexColor('#526475'))
    canvas.drawString(42, 24, 'Benjamin Assouline - Feed the Beast - état documenté du 10/11/2026')
    canvas.drawRightString(A4[0]-42, 24, str(doc.page))


def build(path, elements):
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=40, bottomMargin=42, title=path.stem, author='Benjamin Assouline')
    doc.build(elements, onFirstPage=footer, onLaterPages=footer)


def main():
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    comparison = read('experiments/targeted_clean_v1_20261104.json')
    masked_prep = ROOT / 'data/stages/finalisation_20261110/masked_reference/preparation.json'
    final_run = 'full_reference_masked_20261110' if masked_prep.exists() else 'full_reference_20261110'
    mix = read(f'data/mixes/{final_run}/manifest.json')
    prepared = read(f'data/mixes/{final_run}/prepared/data_meta.json')
    contamination = read('data/stages/finalisation_20261110/masked_reference/contamination.json'
                         if masked_prep.exists() else 'data/stages/finalisation_20261110/mix_contamination.json')
    cleaned = read('data/stages/nettoye/08_exact_dedup/stats.json')
    a = comparison['average']
    run_eval = ROOT / 'runs/full_reference_20261110/eval.json'
    full_state = ('Le run Mac de référence est terminé : BPB ' + number(json.loads(run_eval.read_text())['bpb'], 4) +
                  '. Il utilise float32 ; le run CUDA float16 reste à valider sur le PC.') if run_eval.exists() else (
                  'Le run complet Mac est en cours. Aucun résultat final à 100 M tokens n’est revendiqué ici. Le run CUDA float16 est préparé pour la RTX 5090 de Ben.')
    if masked_prep.exists():
        state_file = ROOT / 'data/stages/finalisation_20261110/masked_training_status.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get('complete'):
            full_state = ('Le modèle final avec coordonnées masquées a vu 100 007 936 tokens : BPB '
                          + number(state['bpb'], 4) + ', MPS float32. La vérification CUDA float16 reste distincte.')
        elif state.get('state') == 'training':
            full_state = ('La référence 100M avant masquage obtient 1,2874 BPB. Le modèle final avec coordonnées '
                          'masquées est en entraînement ; son score final n’est pas encore revendiqué.')
        else:
            full_state = ('Le mélange final avec coordonnées masquées est préparé. Aucun score final à 100M '
                          'n’est encore attesté pour ce mélange ; la référence avant masquage obtenait 1,2874 BPB.')
    source_rows = [['Source', 'Documents sélectionnés', 'Mo de texte', 'Part du texte']]
    for source, r in sorted(mix['by_source'].items(), key=lambda kv:-kv[1]['bytes']):
        source_rows.append([LABELS[source], number(r['documents']), number(r['bytes']/1e6, 2), number(r['bytes']/mix['bytes']*100, 2)+' %'])
    report = [para('Feed the Beast', 'title'), para('Rapport expérimental - version de travail, résultats vérifiés au 10 novembre 2026', 'sub'),
        para('1. Objectif et protocole', 'h'),
        para('Nous étudions la qualité des données pour un modèle identique à celui des autres participants : 10 166 400 paramètres, contexte de 512 tokens et vocabulaire de 8 192 unités. Le BPB mesure les bits nécessaires par octet UTF-8 de validation. Une baisse indique une meilleure prédiction. Le kit, sa configuration et les 230 documents de validation restent intacts.'),
        para('2. Corpus et recette de référence', 'h'),
        para('Le corpus actif contient 3 119 274 documents et 17 352 Mo de texte issus de sept sources. Les données sont gelées avant le 16 octobre 2026. Le mélange complet est un tirage déterministe de '+number(mix['documents'])+' documents ('+number(mix['bytes']/1e6,2)+' Mo), mélangés avec la graine 1337. Sa composition est la suivante :'),
        table(source_rows, [157,120,105,129]),
        para('La littérature et Wikipédia dominent le mélange. Wikinews représente 0,14 % du texte ; la quantité de presse présente dans le web général n’est pas mesurée. La validation comporte environ 25 % de presse. Les proportions naturelles constituent la référence ; leur optimisation reste une piste supplémentaire.'),
        para('3. Préparation et contrôle de contamination', 'h'),
        para('Le nettoyage de référence exclut les IDs réservés, répare l’encodage, extrait le contenu principal, filtre le français à 0,65, remet les livres en paragraphes, décontamine et retire les doublons exacts. Le mélange final masque ensuite les coordonnées structurées : 548 textes modifiés, 72 612 documents conservés dans leur ordre initial. Le contrôle officiel par 13-grammes sur les 419 Mo sélectionnés signale '+str(contamination['n_signales'])+' document sur 230. Zéro signalement ne signifie pas zéro phrase commune.'),
        para('Sources : configs/pipeline/nettoye.yaml ; manifest et prepared/data_meta du mélange '+final_run+' ; masked_reference/preparation.json et contamination.json.', 'small'),
        PageBreak(), para('Expériences et décision', 'title'),
        para('4. Réparer les données sans inventer les valeurs', 'h'),
        para('La version Wikipédia HF 2023 perdait de nombreuses dates et valeurs issues des modèles wiki. Nous l’avons archivée et remplacée par le dump officiel frwiki-20261001, rendu localement. Sur l’essai de 2 000 articles : 15 trous détectés par regex (0,75 %), aucun premier mot collé et 20 relectures tracées. Sur 2 641 265 articles collectés : 0,804 % de trous regex. Les valeurs dynamiques absentes ne sont pas inventées. Ces mesures de qualité ne constituent pas une ablation d’entraînement HF 2023 contre dump 2026.'),
        para('5. Ablation contrôlée du nettoyage supplémentaire', 'h'),
        para('Hypothèse : supprimer des restes de tableaux, infobox, identifiants isolés et répétitions améliore le modèle. La variante modifie 36 475 documents et retire 11,69 Mo, soit 0,067 % du texte. Nous comparons deux tirages d’entrée, 1337 et 2027. L’initialisation du modèle reste 1337. Les deux recettes utilisent le même tokenizer, le même modèle MPS float32 et exactement 25 001 984 tokens vus.'),
        table([['Tirage','Référence','Variante','Variante - référence']]+[[str(p['seed']),number(p['baseline_bpb'],6),number(p['candidate_bpb'],6),'+'+number(p['delta_bpb'],6)] for p in comparison['pairs']]+[['Moyenne',number(a['baseline_bpb'],6),number(a['candidate_bpb'],6),'+'+number(a['delta_bpb'],6)]], [75,125,125,186]),
        para('La variante se dégrade sur les deux tirages et dans les quatre genres. La hausse moyenne vaut '+number(a['delta_bpb'],6)+' BPB. La presse subit la plus forte hausse moyenne (+0,029469). Le critère défini avant les essais exigeait un gain sur les deux tirages, au moins 0,01 BPB moyen, et aucune hausse moyenne par genre supérieure à 0,02. Il n’est pas satisfait. Nous conservons le corpus de référence.'),
        para('6. Limites et passage au budget officiel', 'h'),
        para('Le contrôle automatique de langue porte sur 300 premiers extraits, puis 80 extraits couvrant les sept sources. Un audit de 888 documents complets et 75 relectures ciblées révèle du français rejeté, des éditions bilingues et des catalogues. Les résultats dépendent des classes attribuées automatiquement et des extraits disponibles. Le seuil reste 0,65. La variante complète et la comparaison brut/nettoyé/complet sont en cours.'),
        para(full_state+' Le flux contient '+number(prepared['n_tokens'])+' tokens, assez pour le budget sans retour au début. Les versions et le hash sont conservés. Le passage de 25M à 100M change aussi le budget et l’échantillon : il ne constitue pas une ablation isolée du nettoyage.'),
        para('Preuves : experiments/targeted_clean_v1_20261104.json et _verification.json ; audit de langue ; data/stages/finalisation_20261110/. Les expériences A1/A2/A4-A7 du plan restent à compléter. Rédaction assistée par IA, résultats issus des exécutions du projet.', 'small')]
    complete_path = ROOT / 'experiments/ablation_complete_20261110.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete['contract']['tokens_seen'] == 25001984 and len(complete['results']) == 6
        start = next(i for i, item in enumerate(report) if isinstance(item, Paragraph) and item.getPlainText().startswith('5.'))
        end = next(i for i, item in enumerate(report) if isinstance(item, Paragraph) and item.getPlainText().startswith('6.'))
        rows = [['Données', 'Tirage 1337', 'Tirage 2027', 'Moyenne BPB']]
        for recipe, label in [('brut', 'Brutes protégées'), ('nettoye', 'Nettoyées'), ('complet', 'Complètes')]:
            scores = [next(r['bpb'] for r in complete['results'] if r['recipe'] == recipe and r['seed'] == seed)
                      for seed in [1337, 2027]]
            rows.append([label, number(scores[0], 6), number(scores[1], 6), number(complete['average'][recipe]['bpb'], 6)])
        basic_gain = complete['average']['nettoye']['bpb'] - complete['average']['brut']['bpb']
        extra_delta = complete['average']['complet']['bpb'] - complete['average']['nettoye']['bpb']
        recommendation = ('Le complet améliore les deux tirages et ne dégrade aucun genre en moyenne : il est candidat à une confirmation à 100M.'
                          if complete['proxy_adoption_gate_passed'] else
                          'Le complet ne satisfait pas le critère conservateur de confirmation (deux tirages meilleurs et aucun genre dégradé en moyenne). La référence reste conservée.')
        report[start:end] = [
            para('5. Ablation : brut, nettoyé et complet', 'h'),
            para('Six runs comparent le même modèle, le même tokenizer et les mêmes hyperparamètres : 25 001 984 tokens vus, MPS float32, initialisation 1337. Les nombres 1337 et 2027 ci-dessous sont les graines de tirage des données. Les trois recettes utilisent des proportions naturelles et un plafond de 100 Mo. Le brut exclut les IDs réservés avant sélection, puis est décontaminé ; il ne reçoit aucun autre nettoyage. Le complet ajoute masquage structuré, qualité web et MinHash/LSH avec Jaccard vérifié à 0,80. Tous les flux sont assez longs pour éviter une répétition involontaire.'),
            table(rows, [157,118,118,118]),
            para('Nettoyé moins brut : '+number(basic_gain,6)+' BPB en moyenne. Complet moins nettoyé : '+number(extra_delta,6)+'. Une différence négative est meilleure. '+recommendation),
            para('Le contrôle de 50 paires confirme le seuil numérique, mais 38 décrivent des entités différentes au moyen d’un même gabarit : la déduplication peut réduire la variété et les faits disponibles. L’ordre des sources peut aussi conserver une édition moins bien présentée. Dans une expérience séparée, un nettoyage structurel supplémentaire dégradait déjà le score de +0,017365 BPB : son apparence plus propre ne suffisait pas à l’adopter.')]
        for i, item in enumerate(report):
            if isinstance(item, Paragraph) and item.getPlainText().startswith('Le contrôle automatique de langue'):
                report[i] = para('Le contrôle automatique de langue couvre 300 premiers extraits puis 80 extraits des sept sources, dont 46 tronqués et 17 cas limites. Il reste distinct d’une validation manuelle et ne caractérise pas tous les livres entiers. Le seuil fr 0,65 est conservé. Deux tirages et une initialisation commune ne prouvent pas une significativité statistique ; les différences incluent les proportions de sources après filtrage.')
            if isinstance(item, Paragraph) and item.getPlainText().startswith('Preuves :'):
                report[i] = para('Preuves : experiments/ablation_complete_20261110.json ; audit des paires ; contrôles de contamination ; expériences de nettoyage ciblé. Les mélanges, répétitions et ordres de données restent des pistes supplémentaires. Résultats issus des exécutions du projet.', 'small')
    report_md = []
    for item in report:
        if isinstance(item, Paragraph):
            prefix = '# ' if item.style.name == 'Title' else '## ' if item.style.name == 'Heading' else ''
            report_md.extend([prefix + item.getPlainText(), ''])
        elif isinstance(item, Table):
            for index, row in enumerate(item._cellvalues):
                report_md.append('| ' + ' | '.join(v.getPlainText() if isinstance(v, Paragraph) else str(v) for v in row) + ' |')
                if index == 0:
                    report_md.append('| ' + ' | '.join('---' for _ in row) + ' |')
            report_md.append('')
    report_path = OUT / 'rapport_ablation.pdf'; build(report_path, report)
    assert len(PdfReader(report_path).pages) == 2, 'Le rapport dépasse deux pages'
    sections = [
      ('1. Motivation et responsabilité', [
       'Ce jeu sert au projet étudiant Feed the Beast à ECE Paris. Il permet d’entraîner depuis une initialisation aléatoire le modèle du kit et de comparer des recettes de données à budget fixe. Responsable du projet : Benjamin Assouline. Le code et la rédaction bénéficient de l’assistance de Codex ; cette assistance ne vaut pas validation humaine des annotations.',
       'La recette de référence combine des textes encyclopédiques, littéraires, de presse, de dialogue parlementaire et de web général. Elle ne vise pas une représentation équilibrée de toute la langue française. Le test caché et le classement final sont gérés par l’organisateur.']),
      ('2. Composition et unités', [
       'Trois niveaux sont distingués : 3 223 847 documents bruts actifs ; 3 119 274 documents après le nettoyage de référence ; '+number(mix['documents'])+' documents dans le mélange sélectionné. Une ligne du fichier d’entraînement est un objet JSON avec le champ text. Chaque document reçoit ensuite un marqueur de fin pour la tokenisation.',
       'Le mélange contient '+number(mix['bytes'])+' octets UTF-8 de texte et '+number(prepared['n_tokens'])+' tokens disponibles. Il est assez long pour le budget de 100 007 936 tokens vus. Une partie de sa fin ne sera pas vue. Les parts ci-dessous sont des parts de texte, pas des parts garanties de tokens ni des parts du préfixe effectivement entraîné.']),
      ('3. Origine, dates et licences déclarées', [
       'Wikipédia provient du dump frwiki-20261001, sous CC BY-SA 4.0 déclarée. Wikinews fournit des articles sous CC BY 2.5. Wikisource regroupe des transcriptions et œuvres avec statuts de droits variables, décrits dans les fiches de source. Gutenberg déclare les textes de domaine public aux États-Unis ; les situations dans d’autres pays restent à vérifier. L’Assemblée nationale utilise la Licence Ouverte.',
       'FineWeb-2 français est un sous-ensemble déterministe d’un dépôt figé (af9c13333eb981300149d5ca60a8e9d659b276b9), déclaré ODC-By 1.0. Common Crawl utilise CC-MAIN-2026-39, avec droits des pages détenus par leurs auteurs. Les déclarations des collecteurs ne constituent pas une vérification juridique indépendante de chaque œuvre ou page. Aucune licence uniforme n’est attribuée à l’ensemble.',
       'Le gel porte sur le 16/10/2026. La date d’une capture web, celle d’une révision wiki et celle de publication d’une œuvre ont des sens différents. Les dates des sources sont conservées et leur sémantique est décrite par source. La fiche des sources et les URLs de provenance doivent accompagner une diffusion autorisée.']),
      ('4. Collecte et sélection', [
       'Les collecteurs lisent les dumps ou fichiers en flux et conservent les documents sous JSONL compressé zstd. Ils enregistrent des identifiants stables, une source, un domaine, une URL, une date et une licence déclarée. Les archives, manifestes et dossiers de travail sont conservés sous des liens .nosync sur le Mac.',
       'L’ancienne Wikipédia HF 2023 reste archivée pour ablation. Elle n’entre pas dans le mélange officiel préparé. Les autres limites de collecte comprennent le faible échantillon de Common Crawl, les livres très longs de Wikisource et les dates anciennes du web général. Aucune campagne de consentement individuel n’a été menée auprès des personnes mentionnées dans ces sources publiques.']),
      ('5. Prétraitement, langue et qualité', [
       'La configuration nettoye applique huit étapes : exclusion des IDs de référence, réparation d’encodage, extraction du contenu principal, score de langue, filtre français, remise en paragraphes des livres, décontamination puis déduplication exacte. Le seuil actif est fr >= 0,65. GlotLID donne un diagnostic probabiliste ; les langues proches ne sont pas toutes assimilées au français.',
       'Le contrôle automatique comprend 300 premiers extraits, puis 80 extraits couvrant les sept sources : 63 français, 10 sans français rédigé exploitable, 7 mélangés ; 17 cas limites. Pour ces 80 extraits, la précision pondérée du filtre est de 99,69 % relativement aux classes automatiques ; la sensibilité aux cas limites donne 99,56–99,80 %. Les 46 extraits tronqués ne permettent pas de juger leurs livres entiers. L’audit complémentaire utilise deux détecteurs sur 888 documents entiers, avec 75 relectures automatisées (64 intégrales, 11 par passages). Ce protocole reste distinct d’une validation manuelle indépendante.',
       'La variante complète ajoute qualité web, masquage et MinHash/LSH avec Jaccard vérifié à 0,80. Elle a été exécutée, contrôlée à zéro contamination et évaluée sur deux tirages : 1,632570 BPB moyen, contre 1,622266 pour la référence nettoyée. Les paires proches peuvent décrire des entités différentes ; cette variante n’est pas retenue. Le mélange final conserve la référence avec masquage structuré de ses coordonnées, dans le même ordre. Le contrôle final trouve zéro document de validation signalé.']),
      ('6. Validation, fuite et reproductibilité', [
       'La validation publique contient 230 documents et reste réservée à la mesure. Le contrôle officiel recherche les 13-grammes de mots partagés, signale les documents dépassant 10 % de recouvrement et a signalé '+str(contamination['n_signales'])+' document sur le mélange complet. Zéro signalement n’implique pas l’absence de toute phrase commune. Le contrôle du test caché relève de l’organisateur.',
       'Le tokenizer a 8 192 tokens, sans normalisation destructive, padding, troncature ou BPE-dropout. Il est repris de la référence des essais pour isoler les effets du nettoyage. Tokenizers 0.23.3 est figé ; PyTorch 2.15.0 est la version du projet. Le PC utilisera le binaire CUDA adapté, avec une vraie vérification de calcul GPU. Le kit et sa configuration restent intacts.',
       'Hash du flux préparé : '+prepared['dataset_sha256']+'. Hash du tokenizer : '+prepared['tokenizer_sha256']+'. Le hash du gzip de texte vaut '+mix['train_sha256']+'. Les scripts de transfert refusent les écarts sur les données et les fichiers critiques. Une reconstruction complète depuis une copie neuve et la comparaison T4 à ±0,02 BPB ne sont pas encore attestées.']),
      ('7. Données personnelles, biais et usages', [
       'Le mélange final masque 423 occurrences d’emails, 486 de téléphones et 4 d’adresses IP, dans 548 textes. Aucun article n’est retiré par cette transformation. Les numéros explicitement contextuels sont distingués des tableaux et décimales. Ce masquage ne constitue pas une anonymisation exhaustive : noms, adresses postales, formats obfusqués et de nombreux téléphones internationaux ne sont pas couverts. Les métadonnées d’attribution restent également à considérer avant diffusion.',
       'Les textes anciens, les traductions, les opinions politiques, le commerce, les stéréotypes et des erreurs de numérisation peuvent influencer le modèle. La provenance publique ne garantit pas la factualité. Le modèle reste petit et son entraînement expérimental ; il ne doit pas servir à des décisions médicales, juridiques, financières ou individuelles importantes.',
       'Usages prévus : recherche pédagogique, comparaison de données, reproduction du protocole et analyse de générations. Aucun effet sur des populations particulières n’a été mesuré. Les scores locaux ne prouvent pas une performance générale sur toute la langue française.']),
      ('8. Diffusion et maintenance', [
       'Le mélange et les paquets restent locaux ; aucune URL publique définitive n’est encore validée. L’index provenance.jsonl.gz relie les 72 612 lignes au document d’origine et au texte transformé, grâce à leurs deux SHA-256 et aux masques appliqués. Le texte et l’ordre sont vérifiés après préparation du banc. Les URLs et licences déclarées d’origine sont conservées ; cet index et les fiches des sources doivent accompagner une diffusion autorisée.',
       'Une diffusion doit garder les textes, tokenizer, manifeste, versions, empreintes et informations de provenance nécessaires. Les conditions de chaque source s’appliquent ; les auteurs et liens d’origine doivent rester consultables. Les demandes de correction ou de retrait devront être traitées par le responsable du projet et donner lieu à une nouvelle version identifiée, jamais au remplacement silencieux du flux soumis.',
       'Cette fiche suit les thèmes de motivation, composition, collecte, préparation, usages, distribution et maintenance proposés par Gebru et al., Datasheets for Datasets, arXiv:1803.09010 (2018, version publiée 2021). Elle décrit l’état constaté du projet et sera mise à jour après les vérifications CUDA et la préparation de la publication.'])]
    sheet = [para('Fiche du jeu de données', 'title'), para('Feed the Beast - référence nettoye - état constaté au 10/11/2026', 'sub')]
    md = ['# Fiche du jeu de données - Feed the Beast', '', 'État du 10/11/2026. La publication et la vérification GPU restent à finaliser.', '']
    for idx, (title, paragraphs) in enumerate(sections):
        if idx in [2, 5]:
            sheet.append(PageBreak())
        sheet.append(para(title, 'h'));md.extend(['## '+title,''])
        for p in paragraphs:
            sheet.append(para(escape(p)));md.extend([p,''])
        if idx == 1:
            sheet.append(table(source_rows, [157,120,105,129]));sheet.append(Spacer(1,10))
            md.extend(['| Source | Documents | Mo de texte | Part du texte |','|---|---:|---:|---:|']+['| '+' | '.join(row)+' |' for row in source_rows[1:]]+[''])
    sheet.append(para('Sources : docs/sources/*.md ; configs/pipeline/nettoye.yaml ; manifestes du corpus et du mélange ; rapports d’audit de langue et de nettoyage ; docs/KIT_README.md. Cadre de documentation : https://arxiv.org/abs/1803.09010.', 'small'))
    datasheet_path = OUT / 'datasheet.pdf';build(datasheet_path,sheet)
    (DOC/'DATASHEET.md').write_text('\n'.join(md).rstrip()+'\n')
    (DOC/'RAPPORT.md').write_text('\n'.join(report_md).rstrip()+'\n')
    proof = {'generated': datetime.now().astimezone().isoformat(), 'report_pages': len(PdfReader(report_path).pages),
             'datasheet_pages': len(PdfReader(datasheet_path).pages),
             'report': str(report_path), 'datasheet': str(datasheet_path), 'full_cuda_result_available': False}
    (ROOT/'data/stages/finalisation_20261110/documents.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__ == '__main__':
    main()
