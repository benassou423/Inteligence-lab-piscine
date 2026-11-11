// Mise à jour ciblée du support sauvegardé ; aucun ajout ou retrait de slide.
export default async function ({presentation, input}) {
  const set = (id, value) => { presentation.resolve(id).text = value; };
  const cells = (id, rows) => {
    const table = presentation.resolve(id);
    rows.forEach((row,r)=>row.forEach((value,c)=>table.cells.set(r,c,value)));
  };
  const fmt = value => value.toFixed(6).replace('.', ',');
  set('sh/bi1ojqls', 'Composition du mélange final');
  presentation.resolve('ch/pk36xgfa').series.getItemAt(0).values = input.sourceFractions.map(value=>100*value);
  presentation.slides.getItem(2).speakerNotes.text = 'Source : manifest du mélange full_reference_masked_20261110. Parts des octets de texte sélectionnés, pas des tokens effectivement vus. 72 612 documents, 419 295 632 octets ; ordre original conservé. 548 textes ont leurs coordonnées structurées masquées. Le web général peut contenir de la presse ; sa quantité précise n’est pas mesurée.';

  set('sh/o76lczax', 'Préparation du mélange final');
  cells('tb/tcj6t4fi', [
    ['Étape','But'], ['1. Exclusion','Retirer les IDs réservés'],
    ['2. Encodage','Réparer le texte et les espaces'], ['3. Extraction','Garder le contenu éditorial'],
    ['4-6. Langue et prose','Filtrer le français, reformer les paragraphes'],
    ['7. Décontamination','Retirer les recouvrements problématiques'],
    ['8. Doublons exacts','Conserver une occurrence de chaque texte'],
    ['Mélange et masquage','Remplacer les coordonnées après sélection']]);
  presentation.slides.getItem(4).speakerNotes.text = 'Sources : configs/pipeline/nettoye.yaml et tools/prepare_masked_reference.py. Les huit étapes de nettoyage sont regroupées pour la lecture. Après sélection du mélange, le masquage est appliqué et les doublons exacts éventuels revérifiés. Ici, zéro nouveau doublon : les 72 612 articles sont conservés. Le contrôle officiel du mélange final signale zéro document de validation.';

  presentation.resolve('tb/u5kj2t4n').cells.set(0,1,'Identique dans les trois recettes');
  set('sh/258vm5w3','Complet : masquage, qualité web et MinHash ajoutés au nettoyage.');
  presentation.slides.getItem(6).speakerNotes.text = 'Source : experiments/ablation_complete_20261110.json. Six runs : brut protégé contre la validation, nettoyé, complet ; deux graines de tirage 1337 et 2027. L’initialisation du modèle reste 1337, tokenizer et contrat identiques, 25 001 984 tokens vus sans répétition involontaire. Les proportions naturelles changent après filtrage : l’effet mesuré porte sur la recette complète. Deux tirages ne prouvent pas une significativité statistique.';

  set('sh/mxcvul0j','Comparaison brut, nettoyé et complet');
  const rows = [['Données','Tirage 1337','Tirage 2027','Moyenne BPB']];
  for (const [recipe,label] of [['brut','Brut protégé'],['nettoye','Nettoyé'],['complet','Complet']]) {
    const values=[1337,2027].map(seed=>input.results.find(r=>r.recipe===recipe&&r.seed===seed).bpb);
    rows.push([label,fmt(values[0]),fmt(values[1]),fmt(input.average[recipe].bpb)]);
  }
  cells('tb/bq9wjil8',rows);
  set('sh/ytcvq107','Décision : nettoyage de référence, puis coordonnées masquées.');
  presentation.slides.getItem(7).speakerNotes.text = 'Source : experiments/ablation_complete_20261110.json. Moyennes : brut 1,657747 ; nettoyé 1,622266 ; complet 1,632570. Nettoyé moins brut : -0,035481. Complet moins nettoyé : +0,010304. Les deux tirages et les quatre genres sont moins bons pour la variante complète. Elle n’est pas retenue. Le mélange final conserve les mêmes documents et leur ordre avec les coordonnées structurées masquées. Une expérience structurelle séparée avait également régressé de +0,017365 BPB.';

  set('sh/ds7mxsza','Effet du traitement complet par genre');
  const chart = presentation.resolve('ch/8fmlozyd');
  chart.series.getItemAt(0).values = ['dialogue','encyclopedique','litteraire','presse'].map(g=>input.deltas[g]);
  chart.yAxis = {min:0,max:0.018,majorUnit:0.003,numberFormatCode:'0.000'};
  set('sh/h8fep0bi','Moyenne des deux tirages : complet moins nettoyé. Plus haut est moins bon.');
  presentation.slides.getItem(8).speakerNotes.text = 'Source : complete_genre_deltas dans experiments/ablation_complete_20261110.json. Deltas moyens : dialogue +0,005854 ; encyclopédique +0,010545 ; littéraire +0,010806 ; presse +0,014122. Les 50 paires contrôlées atteignent bien le seuil de Jaccard, mais 38 décrivent des entités différentes au moyen d’un gabarit partagé. La similarité de mots ne garantit pas l’équivalence des faits. Les composants du complet sont regroupés : cette expérience ne permet pas d’isoler chaque composant.';

  set('sh/l4re1k7i','Le modèle final à 100 millions de tokens');
  cells('tb/wzux4b25',[
    ['Préparation','Mesure'],['Texte disponible','419,3 Mo, 72 612 documents'],
    ['Flux préparé','114 701 016 tokens'],['Tokens réellement vus','100 007 936 tokens'],
    ['Score Mac (float32)',fmt(input.final.bpb)+' BPB']]);
  presentation.slides.getItem(9).speakerNotes.text = 'Sources : experiments/full_reference_masked_20261110.json et run correspondant. Budget officiel, 3 052 pas, 100 007 936 tokens vus ; modèle 10 166 400 paramètres, hyperparamètres du kit identiques. BPB final 1,288292 sur 230 documents. MPS float32, temps de calcul 2 039,9 s. Référence avant masquage : 1,287391 BPB. Le résultat CUDA float16 reste à mesurer sur le PC ; aucune équivalence numérique T4 n’est revendiquée avant vérification.';

  set('sh/tsfatgbe','Flux préparé : b88033ad2fba…\nTokenizer : 35c0f184dade…');
  presentation.slides.getItem(10).speakerNotes.text = 'Sources : provenance_report.json du mélange final et transformation.json. Les 72 612 lignes sont contrôlées dans le même ordre. La provenance conserve deux hashes : texte d’origine et texte après masquage. Zéro URL ou licence déclarée manquante ; la déclaration de licence ne constitue pas une validation juridique indépendante. Le contrôle officiel signale zéro document. SHA-256 est une empreinte de fichier/flux permettant de détecter une différence.';

  set('sh/lo3mhoje','Recette et résultats retenus');
  set('sh/bi14be1w','Recette finale');
  set('sh/ahs329kr','Dump Wikipédia 20261001\nFrançais au seuil de 0,65\nTokenizer 8 192 sans perte\n548 textes : coordonnées masquées\nAucun article retiré');
  set('sh/xkjmdoj2','Résultats vérifiés');
  set('sh/wjal4j2h','100 007 936 tokens vus\nBPB validation : 1,288292\n0 document de validation signalé\nPipeline et provenance documentés');
  presentation.slides.getItem(11).speakerNotes.text = 'La référence nettoyée est conservée après les ablations. Le masquage structuré est appliqué au mélange final en préservant les articles et leur ordre. Les noms, adresses postales et formats non reconnus peuvent rester présents : aucune anonymisation exhaustive n’est revendiquée. À la date de cette version, CUDA, la diffusion du dataset et le dépôt distant ne sont pas attestés ; ces étapes restent distinctes des résultats locaux présentés.';
  await presentation.render({slide:[2,4,6,7,8,9,10,11]});
  return {slides:presentation.slides.items.length,changedSlides:[3,5,7,8,9,10,11,12],finalBpb:input.final.bpb};
}
