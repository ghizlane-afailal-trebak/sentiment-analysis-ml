"""
Module d'entraînement des modèles ML pour l'analyse de sentiment
Entraîne Naive Bayes et AdaBoost sur un dataset de tweets arabes
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_arabic_text
import os


def load_or_create_dataset(csv_path='arabic_tweets_ml_ready.csv'):
    """
    Charge un dataset existant ou crée un dataset exemple si aucun n'existe
    """
    if os.path.exists(csv_path):
        print(f"Chargement du dataset depuis {csv_path}...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        return df
    else:
        print(f"Dataset non trouvé. Création d'un dataset exemple...")
        
        example_data = {
            'tweet': [
                'أحب هذا المنتج كثيراً، رائع جداً',
                'سيء جداً، لا أنصح به',
                'ممتاز وسريع في التوصيل',
                'خدمة عملاء سيئة، غير راضٍ',
                'تجربة رائعة وممتازة',
                'منتج رديء الجودة',
                'أفضل منتج استخدمته',
                'لا أنصح بالشراء، تجربة سيئة',
                'رائع جداً ومفيد',
                'سيء وغير عملي',
                'ممتاز وشكراً لكم',
                'خدمة رديئة ومخيبة للآمال',
                'أجمل شيء رأيته',
                'مخيب للآمال',
                'مذهل ورائع',
                'لا يعمل بشكل صحيح',
                'رائع وسريع',
                'جودة سيئة',
                'ممتاز ومفيد جداً',
                'غير جيد ولا أنصح'
            ],
            'sentiment': [
                'positive', 'negative', 'positive', 'negative', 'positive',
                'negative', 'positive', 'negative', 'positive', 'negative',
                'positive', 'negative', 'positive', 'negative', 'positive',
                'negative', 'positive', 'negative', 'positive', 'negative'
            ]
        }
        df = pd.DataFrame(example_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Dataset exemple créé dans {csv_path}")
        return df


def preprocess_dataset(df):
    """Prétraite tous les tweets du dataset"""
    print("Prétraitement des tweets...")
    df['processed_tweet'] = df['tweet'].apply(preprocess_arabic_text)
    # Supprimer les tweets vides après prétraitement
    df = df[df['processed_tweet'].str.len() > 0]
    return df


def train_models(df, test_size=0.2, random_state=42):
    """
    Entraîne les modèles Naive Bayes et AdaBoost
    """
    # Prétraitement
    df = preprocess_dataset(df)
    
    # Séparation des données
    X = df['processed_tweet']
    y = df['sentiment']
    
    # Vectorisation TF-IDF
    print("Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_vectorized = vectorizer.fit_transform(X)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} échantillons")
    print(f"Test set: {X_test.shape[0]} échantillons")
    
    # Entraînement Naive Bayes
    print("\n=== Entraînement Naive Bayes ===")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Accuracy Naive Bayes: {nb_accuracy:.4f}")
    print(classification_report(y_test, nb_pred))
    
    # Entraînement AdaBoost
    print("\n=== Entraînement AdaBoost ===")
    # Utiliser un classifieur de base
    # DecisionTreeClassifier nécessite des données denses
    from sklearn.tree import DecisionTreeClassifier
    # Convertir les matrices éparses en denses pour AdaBoost
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    ada_model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=1.0,
        random_state=random_state
    )
    ada_model.fit(X_train_dense, y_train)
    ada_pred = ada_model.predict(X_test_dense)
    ada_accuracy = accuracy_score(y_test, ada_pred)
    print(f"Accuracy AdaBoost: {ada_accuracy:.4f}")
    print(classification_report(y_test, ada_pred))
    
    # Sauvegarder les modèles et le vectorizer
    print("\n=== Sauvegarde des modèles ===")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarder le vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Vectorizer sauvegardé")
    
    # Sauvegarder Naive Bayes
    with open('models/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print("✓ Modèle Naive Bayes sauvegardé")
    
    # Sauvegarder AdaBoost
    with open('models/adaboost_model.pkl', 'wb') as f:
        pickle.dump(ada_model, f)
    print("✓ Modèle AdaBoost sauvegardé")
    
    return {
        'vectorizer': vectorizer,
        'naive_bayes': nb_model,
        'adaboost': ada_model,
        'nb_accuracy': nb_accuracy,
        'ada_accuracy': ada_accuracy
    }


if __name__ == '__main__':
    print("=== Entraînement des modèles ===")
    
    # Charger le dataset
    df = load_or_create_dataset()
    print(f"Dataset chargé: {len(df)} tweets")
    print(f"Distribution des sentiments:\n{df['sentiment'].value_counts()}\n")
    
    # Entraîner les modèles
    models = train_models(df)
    
    print("\n=== Entraînement terminé ===")
    print(f"Naive Bayes Accuracy: {models['nb_accuracy']:.4f}")
    print(f"AdaBoost Accuracy: {models['ada_accuracy']:.4f}")

