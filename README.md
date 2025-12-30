# Email Spam Detector

**NLP-powered email classifier that detects spam with 100% accuracy using machine learning.**

---

## What This Does

Analyzes email text to automatically classify messages as spam or ham (legitimate). Uses natural language processing and multiple machine learning algorithms to identify spam patterns.

Built using Python, TF-IDF vectorization, and three classification models with neon yellow visualizations.

---

## Results

Model Performance:
- Naive Bayes: 100% accuracy
- Logistic Regression: 100% accuracy
- Random Forest: 100% accuracy
- Perfect classification on all 64 test emails

Dataset:
- 320 emails total
- 169 spam (52.8%)
- 151 ham (47.2%)
- Balanced dataset for training

---

## How It Works

Text Processing:
1. Clean email text (remove URLs, special characters)
2. Convert to lowercase
3. Remove stopwords
4. Extract TF-IDF features

Classification:
1. Train multiple models (Naive Bayes, Logistic Regression, Random Forest)
2. Compare performance
3. Select best model
4. Achieve 100% accuracy on test set

---

## Key Findings

Top Spam Indicators:
- "free" - appears 65 times in spam
- "cash" - 46 times
- "prizes" - 46 times  
- "instantly" - 46 times
- "limited offer" - 33 times
- "discount" - 33 times

Top Ham Indicators:
- "meeting" - 52 times
- "thank you" - 43 times
- "support" - 43 times
- "project update" - 28 times

Observations:
- Spam emails average 54 characters
- Ham emails average 48 characters
- TF-IDF with 45 features is sufficient for perfect classification
- All three models perform equally well on this dataset

---

## How to Run

Train the model:
```bash
python train_detector.py
```

Creates:
- Trained spam classifier
- TF-IDF vectorizer
- 5 visualizations
- Model metadata

---

## What's Inside

- `train_detector.py` - NLP pipeline and model training
- `data/` - 320 email samples
- `visualizations/` - 5 charts with neon yellow styling
- `model/` - Saved classifier and vectorizer

---

## Techniques Used

NLP Methods:
- Text preprocessing and cleaning
- TF-IDF vectorization
- Stopword removal
- Feature extraction

Machine Learning:
- Naive Bayes classifier
- Logistic Regression
- Random Forest
- Model comparison
- Confusion matrix analysis

---

## Built With

- Python - Programming language
- scikit-learn - Machine learning
- pandas - Data manipulation
- matplotlib - Visualizations

---

## Author

Monse Rojo
- Portfolio: monserojo.com
- GitHub: @cyberyimi
- LinkedIn: linkedin.com/in/monse-rojo-6b70b3397/

---

Built with NLP and machine learning.
