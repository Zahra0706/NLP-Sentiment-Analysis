# FRAME NLP — IMDb Sentiment Intelligence

Interface Streamlit pour DistilBERT fine-tuné sur IMDb et explications XAI avec Captum.

## Installation

1. Extraire les quatre fichiers de `best_distilbert` dans `model/` : `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`.
2. Créer un environnement virtuel : `python -m venv .venv`.
3. Activer l'environnement (Windows) : `.venv\Scripts\activate`.
4. Installer les dépendances : `python -m pip install -r requirements.txt`.
5. Démarrer : `python -m streamlit run app.py`.

## Fonctionnement

- Prédiction DistilBERT (tokens au plus 256, sans réentraînement).
- Probabilités softmax (non calibrées).
- Explications locales optionnelles via `LayerIntegratedGradients`, attribuées à la classe prédite.
- Les couleurs montrent le soutien/opposition à la décision par rapport à une baseline `[PAD]` (tokens spéciaux conservés).
- XAI peut être plus lent sur CPU.

## Remarques

Les résultats du test affichés dans « À propos » proviennent du projet d'entraînement décrit et ne sont pas recalculés par l'application. Les métriques peuvent être modifiées si les résultats finaux changent.

N'ajoutez pas le modèle lourd dans un dépôt public sans examiner les limites de stockage et de déploiement.
