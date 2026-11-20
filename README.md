# FRAME NLP — IMDb Sentiment Intelligence & XAI Studio

Plateforme d'analyse de sentiments et d'explicabilité locale (XAI) pour modèles de langage Transformers (DistilBERT fine-tuné sur IMDb) avec **Layer Integrated Gradients** (Captum).

---

## 🚀 Nouveautés & Améliorations de l'Interface

- **Pipeline Unifié en 1 Clic** : Dès que l'utilisateur saisit ou charge une critique et clique sur *"Analyser le sentiment"*, la classification DistilBERT et le calcul complet de l'explicabilité XAI s'exécutent automatiquement en une seule étape (aucun bouton séparé requis).
- **Indicateurs Utiles dans la Barre Latérale (Sidebar)** :
  - **Diagnostic du texte en direct** : Décompte des mots, caractères, diversité lexicale (% de mots uniques), et estimation du temps de lecture.
  - **Jauge de contexte de tokens** : Visualisation en temps réel de l'utilisation de la fenêtre d'attention (`X / 256 tokens`) avec indicateur de troncature.
  - **Exemples prêts à tester** : Presets intégrés (éloge positif fort, échec cuisant négatif, avis nuancé/contrasté, ironie/satire subtile).
  - **Benchmarks officiels** : Rappel des métriques validées sur le jeu test IMDb (Accuracy 92.0%, F1 92.1%, ROC-AUC 97.5%).
  - **Contrôles XAI interactifs** : Ajustement du nombre d'étapes d'intégration Riemann et du Top-K des mots influents.
- **Interface Moderne & Impactante** :
  - Thème sombre haute fidélité avec cartes dynamiques en verre dépoli (*glassmorphism*).
  - Carte de verdict héroïque bicolore (Émeraude pour positif, Corail pour négatif) avec niveau de certitude et temps d'exécution en ms.
  - Double jauge de probabilité comparative interactive.
  - Synthèse en langage naturel des termes ayant le plus fait basculer la décision.
  - Visualisation du texte surligné avec infobulles détaillées (score d'attribution exact et pourcentage d'impact).
  - Graphique à barres horizontales Altair bicolore des mots déterminants.
  - Onglets avancés : *Explication visuelle*, *Table complète des mots*, *Diagnostic de convergence Captum*, et *Export du rapport JSON*.

---

## 📦 Installation & Lancement

1. **Activer l'environnement virtuel** :
   ```powershell
   .venv\Scripts\activate
   ```
2. **Vérifier les dépendances** :
   ```powershell
   pip install -r requirements.txt
   ```
3. **Lancer l'application** :
   ```powershell
   streamlit run app.py
   ```

---

## 🧠 Architecture & XAI

- **Modèle** : `DistilBertForSequenceClassification` fine-tuné pour la classification binaire (0 = Négatif, 1 = Positif).
- **Explicabilité** : `LayerIntegratedGradients` appliquée aux couches d'embeddings avec référence `[PAD]`.
- **Intégrité du Modèle** : Les poids et la configuration du modèle (`model/`) restent strictement intacts et ne sont jamais modifiés.
