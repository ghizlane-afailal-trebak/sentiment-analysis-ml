"""
Module de prétraitement du texte arabe pour l'analyse de sentiment
Gère la normalisation, la suppression des diacritiques et des stopwords
"""

import re
import string
import unicodedata

# Liste des stopwords arabes courants
ARABIC_STOPWORDS = [
    'في', 'من', 'إلى', 'على', 'أن', 'أن', 'هذا', 'هذه', 'ذلك', 'تلك',
    'الذي', 'التي', 'الذين', 'اللاتي', 'كان', 'كانت', 'كانوا', 'يكون',
    'يكون', 'كان', 'كان', 'و', 'أو', 'لكن', 'إذا', 'إن', 'ما', 'لا',
    'لم', 'لن', 'هل', 'كيف', 'متى', 'أين', 'لماذا', 'ما', 'هو', 'هي',
    'هم', 'هن', 'نحن', 'أنت', 'أنتم', 'أنتن', 'أنا', 'كان', 'ليس',
    'لن', 'لم', 'لما', 'أما', 'أيضا', 'إذن', 'حين', 'حيث', 'حتى',
    'لأن', 'لذلك', 'لذا', 'إذ', 'إذا', 'إلا', 'إلا', 'لكن', 'بل',
    'بين', 'بينما', 'خلال', 'دون', 'عند', 'عندما', 'عن', 'مع', 'منذ',
    'نحو', 'هنا', 'هناك', 'هناك', 'هكذا', 'ذلك', 'هذا', 'هذه', 'تلك'
]


def remove_diacritics(text):
    """
    Supprime les diacritiques (harakat) du texte arabe
    """
    arabic_diacritics = re.compile("""
                                ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
    return arabic_diacritics.sub('', text)


def normalize_arabic(text):
    """
    Normalise le texte arabe :
    - Normalise les caractères similaires (أ إ آ, ي ى, etc.)
    - Supprime les espaces multiples
    - Supprime les caractères non-arabes sauf les espaces
    """
    # Normalisation Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Remplacer les variantes de Alef
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Remplacer les variantes de Yeh et Alef Maqsura
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    
    # Normaliser les Tatweel (kashida)
    text = re.sub(r'ـ+', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text


def remove_urls(text):
    """Supprime les URLs du texte"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)


def remove_mentions_and_hashtags(text):
    """Supprime les mentions @ et hashtags #"""
    text = re.sub(r'@[\w_]+', '', text)
    text = re.sub(r'#[\w_]+', '', text)
    return text


def remove_english_chars(text):
    """Supprime les caractères anglais et chiffres"""
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    return text


def remove_punctuation(text):
    """Supprime la ponctuation"""
    # Garder seulement les caractères arabes et espaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text


def remove_arabic_stopwords(text):
    """Supprime les stopwords arabes"""
    words = text.split()
    filtered_words = [word for word in words if word not in ARABIC_STOPWORDS]
    return ' '.join(filtered_words)


def remove_extra_whitespace(text):
    """Supprime les espaces multiples"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_arabic_text(text):
    """
    Pipeline complet de prétraitement du texte arabe
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convertir en string et normaliser
    text = str(text)
    
    # Étape 1: Supprimer les URLs
    text = remove_urls(text)
    
    # Étape 2: Supprimer les mentions et hashtags
    text = remove_mentions_and_hashtags(text)
    
    # Étape 3: Supprimer les diacritiques
    text = remove_diacritics(text)
    
    # Étape 4: Normaliser le texte arabe
    text = normalize_arabic(text)
    
    # Étape 5: Supprimer les caractères non-arabes
    text = remove_english_chars(text)
    text = remove_punctuation(text)
    
    # Étape 6: Supprimer les stopwords
    text = remove_arabic_stopwords(text)
    
    # Étape 7: Nettoyer les espaces
    text = remove_extra_whitespace(text)
    
    return text


