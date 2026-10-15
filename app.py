"""
Application Flask pour l'analyse de sentiment des tweets en arabe
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify
from preprocessing import preprocess_arabic_text
import numpy as np

app = Flask(__name__)

# Variables globales pour les modèles
vectorizer = None
nb_model = None
ada_model = None


def load_models():
    """Charge les modèles et le vectorizer depuis les fichiers pickle"""
    global vectorizer, nb_model, ada_model
    
    models_dir = 'models'
    
    try:
        # Charger le vectorizer
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer chargé")
        
        # Charger Naive Bayes
        with open(os.path.join(models_dir, 'naive_bayes_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        print("✓ Modèle Naive Bayes chargé")
        
        # Charger AdaBoost
        with open(os.path.join(models_dir, 'adaboost_model.pkl'), 'rb') as f:
            ada_model = pickle.load(f)
        print("✓ Modèle AdaBoost chargé")
        
        return True
    except FileNotFoundError as e:
        print(f"Erreur: Modèles non trouvés. {e}")
        print("Veuillez exécuter train_models.py d'abord.")
        return False


def predict_sentiment(text, model_name='naive_bayes'):
    """
    Prédit le sentiment d'un texte arabe
    
    Args:
        text: Texte à analyser
        model_name: 'naive_bayes' ou 'adaboost'
    
    Returns:
        dict: {'sentiment': 'positive' ou 'negative', 'probability': float}
    """
    if not vectorizer:
        return {'error': 'Modèles non chargés'}
    
    # Prétraitement
    processed_text = preprocess_arabic_text(text)
    
    if not processed_text or len(processed_text.strip()) == 0:
        return {'error': 'Le texte est vide après prétraitement'}
    
    # Vectorisation
    text_vectorized = vectorizer.transform([processed_text])
    
    # Prédiction selon le modèle choisi
    if model_name == 'naive_bayes' and nb_model:
        prediction = nb_model.predict(text_vectorized)[0]
        probabilities = nb_model.predict_proba(text_vectorized)[0]
    elif model_name == 'adaboost' and ada_model:
        # AdaBoost nécessite des données denses
        text_vectorized_dense = text_vectorized.toarray()
        prediction = ada_model.predict(text_vectorized_dense)[0]
        probabilities = ada_model.predict_proba(text_vectorized_dense)[0]
    else:
        return {'error': 'Modèle non disponible'}
    
    # Obtenir la probabilité de la classe prédite
    classes = nb_model.classes_ if model_name == 'naive_bayes' else ada_model.classes_
    prob_index = list(classes).index(prediction)
    probability = probabilities[prob_index]
    
    return {
        'sentiment': prediction,
        'probability': float(probability) * 100,  # Convertir en pourcentage
        'processed_text': processed_text
    }


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Route pour la prédiction de sentiment"""
    try:
        data = request.get_json()
        tweet = data.get('tweet', '')
        model_name = data.get('model', 'naive_bayes')
        
        if not tweet:
            return jsonify({'error': 'Veuillez entrer un tweet'}), 400
        
        # Prédiction
        result = predict_sentiment(tweet, model_name)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Route de vérification de santé"""
    models_loaded = vectorizer is not None and nb_model is not None and ada_model is not None
    return jsonify({
        'status': 'ok' if models_loaded else 'models_not_loaded',
        'models_loaded': models_loaded
    })


if __name__ == '__main__':
    print("=== Démarrage de l'application Flask ===")
    
    # Charger les modèles au démarrage
    if load_models():
        print("\n=== Application prête ===\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n=== Erreur: Veuillez d'abord entraîner les modèles ===")
        print("Exécutez: python train_models.py")

