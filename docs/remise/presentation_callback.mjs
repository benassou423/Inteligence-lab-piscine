// Callback de présentation : chaque objet reste modifiable dans le support sauvegardé.
export default async function ({presentation,input}) {
  const C={ink:'#102A43',teal:'#087E8B',muted:'#526475',orange:'#BD4F28',light:'#E7F1F2'};
  const text=(s,value,x,y,w,h,size=26,bold=false,color=C.ink)=>{
    const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    box.text=value;box.text.style={fontFamily:'Arial',fontSize:size,color,bold};return box;
  };
  const table=(s,values,y=190,widths=null)=>{
    const columns=values[0].length;
    const t=s.tables.add({rows:values.length,columns,left:72,top:y,width:1136,height:Math.min(430,values.length*53),
      columnWidths:widths||Array(columns).fill(1136/columns),values});
    for(let r=0;r<values.length;r++)for(let c=0;c<columns;c++){
      const cell=t.getCell(r,c);cell.fill=r===0?C.light:'#FFFFFF';
      cell.text.style={fontFamily:'Arial',fontSize:24,bold:r===0,color:C.ink};
    }
    t.borders.assign({style:'solid',fill:'#D9E2EC',width:1});return t;
  };
  const body=(s,paras,y=190)=>paras.forEach((p,i)=>text(s,p,76,y+i*112,1120,96,29));
  const chart=(s,cats,values,title,color=C.teal,format='0.00')=>s.charts.add('bar',{
    position:{left:80,top:185,width:1100,height:400},categories:cats,
    series:[{name:title,values,fill:color}],hasLegend:false,
    xAxis:{textStyle:{fontSize:22,fill:C.muted}},
    yAxis:{min:0,numberFormatCode:format,textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:'#D9E2EC',width:1,style:'solid'}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:24,fill:C.ink,bold:true}}});
  const descriptors=[
    null,
    {title:'Les contraintes de l’expérience',type:'constraints',note:'Source : docs/KIT_README.md, configuration officielle. Expliquer que le modèle, le budget et l’optimiseur sont figés. Le travail porte sur les données et leur tokenisation. BPB signifie bits par octet : moins de bits pour prédire les mêmes textes est meilleur.'},
    {title:'Composition du mélange complet',type:'sources',note:'Source : data/mixes/full_reference_20261110/manifest.json. Les parts portent sur les 419,3 Mo préparés, pas sur toute la collecte ni exactement sur les 100 M tokens vus. Le mélange reprend les proportions naturelles. Sa forte dominante encyclopédique et littéraire est une limite à discuter.'},
    {title:'Le rendu des modèles Wikipédia',type:'wiki',note:'Sources : pipeline/collect/wikipedia.py, tests et fixtures Wikipédia, rapports de qualité du 04/11. Les trois exemples sont des exemples de tests et non des citations alignées d’un même article avant/après. Sur 2 000 articles, 15 trous regex et 0 premier mot collé. Le dump complet conserve 0,804 % de trous regex : pas de promesse de zéro défaut.'},
    {title:'Les huit étapes du nettoyage',type:'pipeline',note:'Source : configs/pipeline/nettoye.yaml. La validation ne sert pas à l’entraînement. Les scores de langue sont calculés séparément du seuil, ce qui permet de comparer des règles sans refaire ce calcul. La décontamination utilise le contrôle des 13-grammes du kit.'},
    {title:'Le contrôle de langue',type:'language',note:'Sources : audit_langid_full.py et rapport langid_full_audit_20261104.md. Les annotations viennent de Codex, sans validation humaine indépendante. Les relectures sont ciblées, pas un tirage représentatif de taux d’erreur. Certains livres bilingues trompent les deux détecteurs. Conserver fr 0,65 est une décision prudente étayée, pas une garantie absolue.'},
    {title:'Le protocole de comparaison',type:'protocol',note:'Source : experiments/targeted_clean_v1_20261104.json. Les graines 1337 et 2027 concernent le tirage et l’ordre des textes. La graine du modèle reste celle du kit, 1337. Tokenizer et configuration d’entraînement sont identiques. Une seule initialisation commune et deux tirages ne prouvent pas une significativité statistique.'},
    {title:'Le nettoyage ciblé dégrade les deux essais',type:'results',note:'Source : experiments/targeted_clean_v1_20261104.json. Référence : 1,615490 et 1,629042 ; variante : 1,630004 et 1,649257. Les différences sont +0,014514 et +0,020216. Positive signifie moins bon. Moyennes 1,622266 et 1,639631. Le critère pratique était fixé avant les résultats et n’est pas satisfait.'},
    {title:'La presse se dégrade le plus',type:'genres',note:'Source : moyenne des différences par genre des deux paires dans experiments/targeted_clean_v1_20261104.json. Delta = variante moins référence. Tous les deltas sont positifs. Le résultat porte sur le nettoyage testé à 25 M tokens, pas sur tous les nettoyages possibles.'},
    {title:'La recette complète à 100 millions de tokens',type:'full',note:'Sources : manifest.json et prepared/data_meta.json du mélange full_reference_20261110. 114 705 877 tokens disponibles, 100 007 936 réellement demandés au kit. Le mode full ne passe pas --debug-budget. Le run Mac est une référence float32. Le run Windows RTX 5090 est préparé pour CUDA float16 et reste à exécuter.'},
    {title:'La provenance et la reproduction',type:'provenance',note:'Sources : provenance_report.json, TRANSFER_MANIFEST.json de l’archive Windows et docs/remise/PASSAGE_PC_RTX5090.md. Chaque ligne du texte est reliée par SHA-256 à son document d’origine. Les URLs et licences sont des déclarations et non une validation juridique de chaque œuvre. Le réentraînement T4 par l’organisateur exige le même hash et un BPB à ±0,02.'},
    {title:'La recette retenue et les travaux restants',type:'conclusion',note:'Sources : docs/remise/EXIGENCES.md et AGENTS.md. Cette version du support reflète l’état de travail du 10/11. Il reste le run CUDA, les autres expériences du plan, le masquage des données personnelles, la publication, les autorisations de commit/push et la répétition orale. Ne pas présenter ces éléments comme déjà terminés.'},
  ];
  for(let i=input.start;i<input.end;i++){
    if(presentation.slides.items.length>i && presentation.slides.getItem(i).toSnapshot().shapes.length)continue;
    const d=descriptors[i];const s=presentation.slides.items.length>i?presentation.slides.getItem(i):presentation.slides.add({width:1280,height:720});
    s.background.fill='#FFFFFF';text(s,d.title,72,62,1136,95,44,true);text(s,String(i+1).padStart(2,'0'),1160,660,52,30,20,false,C.muted);
    s.speakerNotes.text=d.note;
    if(d.type==='constraints'){
      table(s,[['Élément','Valeur officielle'],['Modèle','10 166 400 paramètres'],['Budget','100 007 936 tokens vus'],['Vocabulaire','8 192 tokens, sans perte'],['Mesure','BPB (bits par octet), global et par genre']],190,[400,736]);
      text(s,'Un modèle commun permet de comparer le choix des données.',76,540,1120,72,28,false,C.teal);
    } else if(d.type==='sources'){
      chart(s,['Wikipédia','Wikisource','Gutenberg','FineWeb','Assemblée','Wikinews','Crawl'],[46.24,32.04,10.41,9.62,1.44,.14,.11],'Part du texte (%)',C.teal,'0.00" %"');
      text(s,'72 612 documents sélectionnés, 419,3 Mo de texte',76,610,1120,50,27);
    } else if(d.type==='wiki'){
      table(s,[['Exemple de test','Rendu attendu'],['{{nombre|12345}}','12 345'],['{{date|14|mars|1885}}','14 mars 1885'],["L'''algorithmique'''","L'algorithmique"]],205,[600,536]);
      text(s,'2 000 articles : 15 trous repérés (0,75 %)\nAucun premier mot collé',76,495,1120,90,29,false,C.teal);
    } else if(d.type==='pipeline'){
      table(s,[['Étape','But'],['1. Exclusion','Retirer les IDs tenus à l’écart'],['2. Encodage','Réparer le texte et les espaces'],['3. Extraction','Garder le contenu éditorial'],['4–5. Langue','Noter puis filtrer le français'],['6. Paragraphes','Reconstituer les livres'],['7. Contamination','Retirer les recouvrements problématiques'],['8. Doublons exacts','Conserver une copie de chaque texte']],170,[350,786]);
    } else if(d.type==='language'){
      table(s,[['Contrôle','Portée'],['Classement automatique','300 puis 80 extraits'],['Analyse automatique intégrale','888 documents, 9 016 passages'],['Relecture automatisée','64 intégrales et 11 par passages']],200,[510,626]);
      text(s,'Les détecteurs peuvent s’accorder sur un livre bilingue.',76,465,1120,70,30);
      text(s,'Décision : conserver le français seul au seuil de 0,65.',76,555,1120,70,30,true,C.teal);
    } else if(d.type==='protocol'){
      table(s,[['Paramètre','Identique dans les deux recettes'],['Budget local','25 001 984 tokens vus'],['Modèle et tokenizer','Même configuration et même vocabulaire'],['Graine du modèle','1337'],['Tirages d’entrée','1337 et 2027, appariés par recette']],190,[460,676]);
      text(s,'L’effet mesuré porte sur la recette de données.',76,555,1120,65,30,false,C.teal);
    } else if(d.type==='results'){
      table(s,[['Tirage','Référence','Variante','Différence'],['1337','1,615490','1,630004','+0,014514'],['2027','1,629042','1,649257','+0,020216'],['Moyenne','1,622266','1,639631','+0,017365']],190,[180,310,310,336]);
      text(s,'Un BPB plus élevé indique une moins bonne prédiction.',76,475,1120,65,29);
      text(s,'Décision : conserver la référence.',76,555,1120,65,32,true,C.orange);
    } else if(d.type==='genres'){
      chart(s,['Dialogue','Encyclopédique','Littéraire','Presse'],[.0129730644,.0167619749,.0105071679,.0294693360],'Hausse du BPB',C.orange,'0.0000');
      text(s,'Différence moyenne : variante moins référence. Plus haut est moins bon.',76,610,1120,50,25);
    } else if(d.type==='full'){
      table(s,[['Préparation','Mesure'],['Texte disponible','419,3 Mo, 72 612 documents'],['Flux préparé','114 705 877 tokens'],['Budget réellement demandé','100 007 936 tokens'],['Score Mac (float32)','1,287391 BPB']],190,[475,661]);
      text(s,'Calcul Mac terminé. Le run CUDA RTX 5090 reste à vérifier.',76,555,1120,75,28,false,C.teal);
    } else if(d.type==='provenance'){
      text(s,'72 612 lignes reliées à leur document d’origine',76,190,1120,80,36,true,C.teal);
      text(s,'Chaque ligne possède une URL et une licence déclarée.',76,300,1120,80,29);
      text(s,'Flux préparé : e117b978a787…\nTokenizer : 35c0f184dade…',76,400,1120,115,29);
      text(s,'Versions figées, kit intact, contrôle de contamination à 0 signalé',76,555,1120,75,28);
    } else if(d.type==='conclusion'){
      text(s,'Référence conservée',76,185,530,70,36,true,C.teal);
      text(s,'Dump Wikipédia 20261001\nFrançais seul au seuil de 0,65\nTokenisation sans perte\nComparaisons appariées',76,290,530,245,29);
      text(s,'À vérifier avant remise',680,185,520,70,36,true,C.orange);
      text(s,'Run CUDA et résultat final\nExpériences restantes du plan\nDiffusion documentée des données\nPrésentation orale avec Ben',680,290,520,245,29);
      text(s,'État du dossier au 10 novembre 2026',76,610,1100,45,23,false,C.muted);
    }
  }
  presentation.slides.getItem(0).speakerNotes.text='Source : docs/KIT_README.md. Présentation de travail au 10/11/2026. Expliquer que le modèle est fixé et que les choix portent sur la donnée. Les résultats présentés sont ceux des essais locaux réellement exécutés.';
  return {slides:presentation.slides.items.length};
}
