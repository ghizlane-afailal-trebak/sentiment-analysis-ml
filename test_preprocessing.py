"""
Script de test rapide pour le prétraitement du texte arabe
"""

from preprocessing import preprocess_arabic_text

# Exemples de tweets à tester
test_tweets = [
    "أحب هذا المنتج كثيراً، رائع جداً! 😊",
    "سيء جداً، لا أنصح به #badproduct",
    "ممتاز وسريع في التوصيل @shop",
    "خدمة عملاء سيئة، غير راضٍ https://example.com",
    "أحبُ هذا المُنتَج كَثِيراً ورائِعٌ جِدّاً"
]

print("=== Test du prétraitement ===\n")

for i, tweet in enumerate(test_tweets, 1):
    print(f"Tweet {i}:")
    print(f"  Original: {tweet}")
    processed = preprocess_arabic_text(tweet)
    print(f"  Prétraité: {processed}")
    print()


