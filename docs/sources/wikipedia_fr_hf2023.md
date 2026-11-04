Prochaine action : lire les limites de cette archive avant une ablation — 2 min.

# Wikipédia HF 2023 — `wikipedia_fr_hf2023`

1. **Rôle.** Conserver l’ancienne version pour une ablation, c’est-à-dire une comparaison contrôlée entre les données HF 2023 et le dump officiel 2026. Elle est désactivée dans `configs/collect.yaml` et exclue des configurations courantes `brut` et `nettoye`. Elle ne rejoint pas automatiquement le corpus d’entraînement.

2. **Origine inchangée.** Dataset [Hugging Face `wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), configuration `20231101.fr`, révision figée `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Le collecteur historique est conservé dans `pipeline/collect/wikipedia_hf2023.py` ; seul son nom de source change. Le champ `date` reste **2023-11-01**, date du dump, pas une date de révision propre à chaque article.

3. **Migration vérifiée.** Les données sont dans `data/raw.nosync/wikipedia_fr_hf2023/`, accessibles par le lien `data/raw`. **2 564 645 documents**, **7 814 155 529 octets de texte**, **87 fichiers**, environ **2,75 Go compressés**. Chaque document a été comparé avant/après : seuls `source` et le préfixe de `id` changent. Les textes et les autres métadonnées sont identiques. Preuve par fichier et empreinte des textes : `_migration_hf2023.json`. Le script rejouable est `tools/archive_wikipedia_hf2023.py`.

4. **Limites conservées volontairement.** Le convertisseur du dataset supprimait de nombreux modèles : dates, nombres, unités, parfois apostrophe initiale ou sujet. Ces défauts ne sont pas corrigés dans l’archive ; les corriger empêcherait de comparer les deux versions. La différence avec 2026 inclut aussi les évolutions éditoriales et les nouveaux articles. Aucun résultat d’entraînement de cette ablation n’est encore mesuré.

5. **Licence et attribution.** Textes sous **CC BY-SA 4.0** ; URLs d’origine et métadonnées HF conservées. Voir les [conditions Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content).

6. **Commandes dédiées.** À lancer depuis `Inteligence-lab-piscine/`. La première commande reconnaît l’archive complète déjà présente. Le nettoyage dédié utilise les mêmes paramètres que `nettoye`, avec le traitement Wikipédia appliqué à ce nom de source, et écrit dans un dossier distinct. Ce nettoyage d’ablation n’a pas été lancé dans le chantier du dump 2026.

```bash
uv run python -m pipeline.collect wikipedia_fr_hf2023
uv run python -m pipeline.run --config configs/pipeline/nettoye_hf2023.yaml
```

Le champ `enabled: false` protège le lancement `all`. Les volumes affichent cette archive séparément du total actif. Ne pas mélanger les deux versions dans une même recette de comparaison.
