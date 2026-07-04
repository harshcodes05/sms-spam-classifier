# SMS Spam Classifier

A machine learning web app that classifies SMS/text messages as **Spam** or **Not Spam**, built using NLP preprocessing and classical ML models.

## Overview

This project implements an end-to-end spam detection pipeline:
- Text preprocessing (lowercasing, tokenization, stopword removal, stemming)
- Feature extraction using TF-IDF vectorization
- Benchmarking of 8 machine learning classifiers
- Deployment as an interactive web app using Streamlit

## Tech Stack

- **Python**
- **scikit-learn** — model training and evaluation
- **NLTK** — text preprocessing
- **XGBoost** — gradient boosting classifier
- **Pandas / NumPy** — data handling
- **Streamlit** — web app interface

## Model Comparison

| Model | Accuracy | Precision |
|---|---|---|
| **Naive Bayes** | 97.1% | **100%** |
| KNN | 90.5% | 100% |
| Extra Trees | 97.5% | 97.5% |
| SVM (SVC) | 97.6% | 97.5% |
| Random Forest | 97.4% | 98.3% |
| XGBoost | 97.0% | 95.7% |
| Logistic Regression | 95.6% | 96.0% |
| AdaBoost | 92.2% | 82.0% |

**Final model chosen: Multinomial Naive Bayes** — selected for its perfect precision (critical for spam detection, where false positives are especially undesirable) combined with strong overall accuracy.

## How It Works

1. User enters a message in the web app
2. Text is cleaned and transformed (tokenized, stopwords removed, stemmed)
3. Transformed text is vectorized using a pre-fitted TF-IDF vectorizer
4. The trained Naive Bayes model predicts whether the message is spam or not

## Running Locally

```bash
git clone https://github.com/harshcodes05/sms-spam-classifier.git
cd sms-spam-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit web app
├── model.pkl           # Trained Naive Bayes classifier
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Dependencies
└── notebook/            # EDA and model comparison notebook
```

## Author

**Harsh Sharma** — [GitHub](https://github.com/harshcodes05)