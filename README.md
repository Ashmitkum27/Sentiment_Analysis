# AI-Powered Multilingual Sentiment Analysis System

## Project Overview
This project is an end-to-end sentiment analysis system designed for e-commerce reviews. It features a robust Machine Learning backend that can classify reviews into **Positive, Neutral, or Negative** sentiments. The system is specifically engineered to handle real-world challenges like class imbalance, multilingual text, and image-based reviews (OCR).

## Key Features
- **Multilingual Support**: Automatically detects and translates 100+ languages (Hindi, Spanish, French, etc.) into English before analysis.
- **Model Showdown**: Compares three different model architectures (Ensemble, Linear SVC, Logistic Regression) in real-time to show which one is more confident.
- **Negation Awareness**: Advanced text preprocessing that preserves critical words like "not," "no," and "never."
- **Optical Character Recognition (OCR)**: Ability to upload images of reviews and analyze the text within them.
- **Interactive Dashboard**: Real-time visualization using Chart.js, including a history of analyzed reviews.
- **Bias Correction**: Balanced training using combined undersampling and oversampling (SMOTE-style logic).

## Technology Stack
- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning**: Scikit-Learn, NLTK (VADER, Lemmatization), Imbalanced-Learn
- **OCR Engine**: EasyOCR
- **Translation**: Deep-Translator, Langdetect
- **Frontend**: HTML5, CSS3 (Glassmorphism design), JavaScript, Chart.js

## Project Workflow
1. **Data Preprocessing**: Raw text is cleaned, negated words are preserved, and VADER sentiment scores are extracted.
2. **Feature Extraction**: Text is converted to numerical data using TF-IDF bigrams, combined with VADER feature vectors.
3. **Classification**: The input is fed into an **Ensemble Voting Classifier** (LinearSVC + Logistic Regression).
4. **Translation Layer**: If the input is non-English, it is translated via Google Translate API before hitting the ML models.
5. **UI Update**: The frontend displays the prediction result, confidence levels, and a side-by-side comparison of all models.

## How to Run the Project

### 1. Prerequisite
Ensure you have Python 3.9+ installed and a virtual environment set up.

### 2. Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Setup NLTK Data
The project requires specific NLTK resources. Run this command once:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### 4. Train the Model (Optional)
If you wish to re-train the models from the dataset:
```bash
python train_model.py
```

### 5. Start the Backend API
Run the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

### 6. Launch the Frontend
Simply open the `index.html` file in any modern web browser (Chrome, Firefox, Edge).

---
**Developed as a Final Year Engineering Project.**
