import io
import pandas as pd
import joblib
import re
import os
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import easyocr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Stability for language detection
DetectorFactory.seed = 42

# Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Negation-Aware Stopwords
standard_stops = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'but', 'however', 'neither', 'nor', 'none', 'cannot', 'isnt', 'wasnt', 'werent', 'dont', 'didnt', 'havent', 'hasnt', 'hadnt', 'wont', 'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
sentimental_stops = standard_stops - negation_words

lemmatizer = WordNetLemmatizer()

# Load models
print("Loading Showdown Models...")
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
ensemble_model = joblib.load('models/sentiment_rf_model.pkl')
svc_model = joblib.load('models/svc_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
analyzer = joblib.load('models/vader_analyzer.pkl')

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def clean_text_with_negations(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace("n't", "nt").replace("'re", " are").replace("'s", " is")
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in sentimental_stops]
    return " ".join(tokens)

def get_vader_features(text):
    scores = analyzer.polarity_scores(text)
    return np.array([[scores['compound'], scores['pos'], scores['neu'], scores['neg']]])

def translate_if_needed(text):
    try:
        if len(text.strip()) < 3: return text, "unknown"
        lang = detect(text)
        if lang != 'en':
            # Translate to English for more robust processing by our specialized model
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated, lang
        return text, 'en'
    except:
        return text, 'en'

app = FastAPI(title="Multilingual Model Showdown API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPredictionRequest(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(request: TextPredictionRequest):
    original_text = request.text
    translated_text, source_lang = translate_if_needed(original_text)
    
    # Preprocessing on English (or translated) text
    cleaned = clean_text_with_negations(translated_text)
    vec_tfidf = tfidf.transform([cleaned])
    vec_vader = get_vader_features(translated_text)
    combined = sp.hstack([vec_tfidf, sp.csr_matrix(vec_vader)])
    
    # Run predictions on all components
    showdown = {}
    models = {
        "Final Ensemble": ensemble_model,
        "Linear SVC": svc_model,
        "Logistic Reg": lr_model
    }
    
    classes = ensemble_model.classes_
    for name, model in models.items():
        raw_pred = model.predict(combined)[0]
        
        # Map index to label if model returns numeric class
        if str(raw_pred).isdigit() or isinstance(raw_pred, (int, np.integer)):
            pred_label = classes[int(raw_pred)]
        else:
            pred_label = raw_pred
            
        probs = model.predict_proba(combined)[0]
        showdown[name] = {
            "sentiment": str(pred_label),
            "confidence": float(max(probs))
        }
        
    return {
        "source_lang": source_lang,
        "translated_text": translated_text if source_lang != 'en' else None,
        "showdown": showdown,
        # Backward compatibility for the main badge
        "sentiment": showdown["Final Ensemble"]["sentiment"],
        "confidence": showdown["Final Ensemble"]["confidence"],
        "model": "Ensemble (with Translation)"
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    results = reader.readtext(contents)
    extracted_text = " ".join([text for (_, text, _) in results])
    
    if not extracted_text.strip():
        return {"error": "No text could be extracted.", "extracted_text": ""}

    # Reuse predict_text logic implicitly
    res = await predict_text(TextPredictionRequest(text=extracted_text))
    res["extracted_text"] = extracted_text
    return res

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Try to read CSV with different encodings
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        try:
            import chardet
            detected = chardet.detect(contents)
            df = pd.read_csv(io.BytesIO(contents), encoding=detected['encoding'])
        except Exception as e:
            return {"error": f"Could not read CSV file: {str(e)}"}
    
    # Find the review text column (case-insensitive)
    text_col = None
    for col in df.columns:
        if col.strip().lower() in ['reviewtext', 'review_text', 'review', 'text', 'comment', 'feedback']:
            text_col = col
            break
    
    if text_col is None:
        return {"error": f"No review text column found. Available columns: {list(df.columns)}. Expected one of: reviewText, review_text, review, text, comment, feedback"}
    
    results = []
    for idx, row in df.iterrows():
        text = str(row[text_col])
        if not text.strip() or text.strip().lower() == 'nan':
            results.append({
                "row": idx,
                "text": text,
                "Predicted_Sentiment": "Unknown",
                "confidence": 0.0
            })
            continue
        
        translated_text, source_lang = translate_if_needed(text)
        cleaned = clean_text_with_negations(translated_text)
        vec_tfidf = tfidf.transform([cleaned])
        vec_vader = get_vader_features(translated_text)
        combined = sp.hstack([vec_tfidf, sp.csr_matrix(vec_vader)])
        
        pred = ensemble_model.predict(combined)[0]
        probs = ensemble_model.predict_proba(combined)[0]
        
        # Map index to label if needed
        if str(pred).isdigit() or isinstance(pred, (int, np.integer)):
            pred_label = ensemble_model.classes_[int(pred)]
        else:
            pred_label = pred
        
        results.append({
            "row": idx,
            "text": text[:100],
            "Predicted_Sentiment": str(pred_label),
            "confidence": float(max(probs)),
            "source_lang": source_lang
        })
    
    # Summary
    sentiment_counts = {}
    for r in results:
        s = r["Predicted_Sentiment"]
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
    
    return {
        "total_processed": len(results),
        "summary": sentiment_counts,
        "sample_results": results
    }

@app.get("/health")
def health():
    return {"status": "Healthy", "version": "Showdown v4.0"}
