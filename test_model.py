import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load assets
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
clf = joblib.load('models/sentiment_rf_model.pkl')
analyzer = joblib.load('models/vader_analyzer.pkl')

standard_stops = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'but', 'however', 'neither', 'nor', 'none', 'cannot', 'isnt', 'wasnt', 'werent', 'dont', 'didnt', 'havent', 'hasnt', 'hadnt', 'wont', 'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
sentimental_stops = standard_stops - negation_words
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace("n't", "nt").replace("'re", " are").replace("'s", " is")
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in sentimental_stops]
    return " ".join(tokens)

def get_vader(text):
    scores = analyzer.polarity_scores(text)
    return np.array([[scores['compound'], scores['pos'], scores['neu'], scores['neg']]])

test_cases = [
    "This product is amazing and I love it!",
    "It is okay, nothing special but works fine.",
    "Worst purchase ever. Broke in two days. Do not buy!",
    "The item is not good at all, very disappointing.",
    "I expected better quality but it is just average.",
    "Total waste of money, never buying again."
]

print("--- Phase 3 Prediction Test ---")
for text in test_cases:
    cleaned = clean_text(text)
    vec_t = tfidf.transform([cleaned])
    vec_v = get_vader(text)
    combined = sp.hstack([vec_t, sp.csr_matrix(vec_v)])
    
    pred = clf.predict(combined)[0]
    probs = clf.predict_proba(combined)[0]
    
    print(f"Text: {text}")
    print(f"Prediction: {pred} (Confidence: {max(probs):.2f})\n")
