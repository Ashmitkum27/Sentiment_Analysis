# Technical Implementation Report: AI Sentiment Analysis System

## 1. Introduction
The objective of this project was to develop a robust, high-performance sentiment analysis system for e-commerce. Unlike standard classifiers, this system was designed to handle specific real-world challenges: extreme class imbalance, linguistic nuances (negations), and multilingual inputs.

## 2. Dataset Analysis
The project utilizes the **Amazon Musical Instruments Reviews** dataset.
- **Initial Observation**: The dataset exhibited a significant "Positive Bias" (approx. 90% Positive, 5% Neutral, 5% Negative).
- **Challenge**: Standard Machine Learning models trained on this data would default to "Positive" for almost every input, failing to protect the business from negative feedback.

## 3. Experimental Methodology

### 3.1 Advanced Preprocessing
Standard preprocessing often removes "stopwords." However, for sentiment, words like *not, no,* and *never* are critical.
- **Negation Preservation**: We customized the NLTK stopword list to exclude 20+ negation words.
- **Lemmatization**: Used NLTK’s WordNetLemmatizer to reduce words to their root forms (e.g., "better" and "good" are linked), increasing model generalization.

### 3.2 Feature Engineering (Hybrid Approach)
We implemented a dual-feature extraction strategy:
1.  **TF-IDF (Term Frequency-Inverse Document Frequency)**: Captured the importance of specific words and word pairs (Bigrams).
2.  **VADER Sentiment Intensity**: A rule-based engine that extracts "Polarity Scores." This provides the model with a baseline "opinion" before it even reads the specific vocabulary.

### 3.3 Sampling Strategy
To combat the 90% positive bias, we implemented a **Hybrid Sampling** technique:
- **Major Undersampling**: Reduced the number of Positive training samples to prevent them from "drowning out" other signals.
- **Over-sampling**: Artificially boosted the Negative and Neutral samples to create a 1:1:1 balanced training ratio.

## 4. System Architecture

### 4.1 Ensemble Learning
The core engine is a **Voting Classifier** (Ensemble) consisting of:
- **Linear SVC (Support Vector Classifier)**: High precision in high-dimensional text space.
- **Logistic Regression**: Reliable probabilistic classification.
By "voting" between these two calibrated models, the system reduces individual model errors.

### 4.2 Multilingual Integration
Most ML models fail on non-English text. We implemented a **Translation Layer** using a detection engine. If non-English text (e.g., Hindi: *"उत्पाद अच्छा नहीं है"*) is detected, it is translated to English (*"Product is not good"*) before being fed into the optimized English model.

## 5. Results & Evaluation
- **Bias Reduction**: Success. The system now correctly identifies negative phrases like "not good" which were previously classified as Positive.
- **Recall Improvement**: The "Recall" for Negative and Neutral classes improved significantly (from ~0% to over 50% in testing).
- **Model Showdown**: The frontend allows users to compare the Ensemble, SVC, and Logistic Regression models side-by-side, providing transparency on the system's decision-making process.

## 6. Conclusion
The implementation successfully transitioned from a biased baseline to a robust, multilingual ensemble system. The combination of hybrid features (TF-IDF + VADER) and strategic sampling ensures that the system is accurate and commercially viable for monitoring customer satisfaction across diverse languages.
