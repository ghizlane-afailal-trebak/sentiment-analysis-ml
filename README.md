# Analyse de sentiments des tweets en arabe — Machine Learning

Projet de classification de sentiments de tweets en arabe, avec une interface web Flask et un notebook Jupyter consacré à l'analyse exploratoire, au prétraitement linguistique et à la comparaison de modèles.

## Fonctionnalités

- Prétraitement du texte arabe : normalisation, suppression des diacritiques, URLs, mentions et mots vides.
- Vectorisation TF-IDF.
- Prédiction du sentiment positif ou négatif via Naive Bayes ou AdaBoost.
- Interface web en arabe avec affichage du résultat et d'un score de confiance.
- Notebook d'analyse exploratoire et d'évaluation : métriques de classification, matrices de confusion et courbes ROC/PR.

## Technologies

Python, Flask, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, WordCloud, HTML et CSS.

## Structure

```text
sentiment-analysis-ml/
├── app.py
├── preprocessing.py
├── train_models.py
├── test_preprocessing.py
├── requirements.txt
├── models/
│   ├── adaboost_model.pkl
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── notebooks/
    └── NOTEBOOK_G10.ipynb
```

## Installation

Python 3.10 ou une version compatible avec les versions des bibliothèques listées est recommandé.

```bash
git clone https://github.com/USERNAME/sentiment-analysis-ml.git
cd sentiment-analysis-ml
python -m venv .venv
```

Activation de l'environnement :

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Installation des dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer l'interface web

Les fichiers de modèles pré-entraînés sont inclus dans `models/`. À partir de la racine du projet :

```bash
python app.py
```

Puis ouvrir http://127.0.0.1:5000 dans le navigateur.

Si les modèles sont absents ou si vous souhaitez les réentraîner, consultez `train_models.py` et fournissez le jeu de données attendu par ce script.

## Exécuter le notebook

```bash
jupyter notebook notebooks/NOTEBOOK_G10.ipynb
```

Le notebook charge le dataset Twitter arabe depuis `./archive/Twitter`, avec des dossiers `Positive/` et `Negative/`. **Ce dataset n'est pas inclus dans ce dépôt** : placez-le localement à cet emplacement avant d'exécuter les cellules qui le chargent. Vérifiez également les chemins de fichiers et les dépendances propres à votre environnement.

## API

- `GET /` : interface web.
- `POST /predict` : reçoit un JSON contenant `tweet` et, facultativement, `model` (`naive_bayes` ou `adaboost`).
- `GET /health` : état du chargement des modèles.

Exemple de corps JSON pour `/predict` :

```json
{
  "tweet": "هذا منتج رائع جدا",
  "model": "naive_bayes"
}
```

## Remarques

- Le score affiché est la probabilité estimée par le modèle pour la classe prédite ; il ne garantit pas que la prédiction soit correcte.
- Les fichiers `.pkl` sont conservés car l'application en a besoin. Ne chargez jamais des fichiers pickle provenant d'une source non fiable.
- Les données du dataset ne sont pas versionnées dans Git.

## Licence

Aucune licence open source n'est spécifiée actuellement. Ajoutez une licence si vous souhaitez définir explicitement les conditions de réutilisation.
