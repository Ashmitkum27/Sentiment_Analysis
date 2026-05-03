import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Custom Stopword List: REMOVE negation words from the list so we KEEP them in the text
standard_stops = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'but', 'however', 'neither', 'nor', 'none', 'cannot', 'isnt', 'wasnt', 'werent', 'dont', 'didnt', 'havent', 'hasnt', 'hadnt', 'wont', 'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
sentimental_stops = standard_stops - negation_words

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def clean_text_with_negations(text):
    if not isinstance(text, str):
        return ""
    # Lowercase and handle some basic punctuations like "don't" -> "dont"
    text = text.lower()
    text = text.replace("n't", "nt").replace("'re", " are").replace("'s", " is")
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = text.split()
    # Keep only sentimental_stops filtered tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in sentimental_stops]
    return " ".join(tokens)

class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        for text in X:
            scores = analyzer.polarity_scores(text)
            features.append([scores['compound'], scores['pos'], scores['neu'], scores['neg']])
        return np.array(features)

def map_sentiment(rating):
    if rating <= 2: return 'Negative'
    elif rating == 3: return 'Neutral'
    else: return 'Positive'

def main():
    print("Loading data...")
    dataset_path = 'Musical_instruments_reviews_updated.csv'
    if not os.path.exists(dataset_path): return

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['reviewText', 'overall'])
    df['Sentiment'] = df['overall'].apply(map_sentiment)
    
    print("Pre-processing with Negation-Awareness...")
    # We clean the text for TF-IDF but keep it slightly more raw for VADER
    df['clean_review'] = df['reviewText'].apply(clean_text_with_negations)

    X = df['reviewText'] # Pass raw text to the pipeline
    y = df['Sentiment']

    # Train Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Building Feature Pipeline...")
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    
    # Process Train Features
    print("Extracting TF-IDF and VADER features...")
    X_train_clean = [clean_text_with_negations(t) for t in X_train_raw]
    X_test_clean = [clean_text_with_negations(t) for t in X_test_raw]
    
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)
    
    vader_extractor = SentimentFeatureExtractor()
    X_train_vader = vader_extractor.transform(X_train_raw)
    X_test_vader = vader_extractor.transform(X_test_raw)
    
    # Combine TF-IDF and VADER features
    import scipy.sparse as sp
    X_train_combined = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_vader)])
    X_test_combined = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_vader)])

    # Balancing Strategy: Undersample Positive, Oversample Negative/Neutral
    print("Strategically Balancing: 1:1:1 Ratio...")
    # First, bring Positive down to a more manageable number (e.g. 1500)
    # Then upsample others to 1500
    sampling_strategy_under = {'Positive': 2000} 
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
    X_under, y_under = under.fit_resample(X_train_combined, y_train)
    
    # Now oversample others to match the 2000
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_under, y_under)

    print(f"Final Training Distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

    # Ensemble
    print("Training Enhanced Bias-Fix Ensemble...")
    base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000, C=0.2)
    svc = CalibratedClassifierCV(base_svc, method='sigmoid', cv=5)
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    nb = MultinomialNB() # NB doesn't handle negative features well, but VADER pos/neg are 0-1

    ensemble = VotingClassifier(
        estimators=[('svc', svc), ('lr', lr)], # removed NB to focus on those with class weights
        voting='soft',
        weights=[3, 1]
    )

    ensemble.fit(X_resampled, y_resampled)

    # Evaluation
    print("Evaluating Optimized Model...")
    y_pred = ensemble.predict(X_test_combined)
    print("Full Classification Report:\n", classification_report(y_test, y_pred))

    # Saving
    print("Saving Models to 'models/'...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    joblib.dump(ensemble, 'models/sentiment_rf_model.pkl') # Ensemble
    
    # Save the fitted sub-models from the ensemble
    joblib.dump(ensemble.named_estimators_['svc'], 'models/svc_model.pkl')
    joblib.dump(ensemble.named_estimators_['lr'], 'models/lr_model.pkl')
    
    joblib.dump(analyzer, 'models/vader_analyzer.pkl')
    print("Training complete!")

if __name__ == "__main__":
    main()
