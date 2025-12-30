"""
Email Spam Detector - NLP Classification
Uses NLP techniques to classify emails as spam or ham (legitimate)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import re
from collections import Counter
import pickle
import os

# Simple stopwords list
STOPWORDS = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
                 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
                 'their', 'from', 'as', 'by'])

# Neon Yellow color scheme for Project 5
NEON_YELLOW = '#ffff00'

print("=" * 70)
print("EMAIL SPAM DETECTOR - NLP PROJECT")
print("=" * 70)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n📊 Loading data...")
df = pd.read_csv('data/email_spam_dataset.csv')

print(f"✅ Loaded {len(df)} emails")
print(f"   Spam: {len(df[df['label']=='spam'])} ({len(df[df['label']=='spam'])/len(df)*100:.1f}%)")
print(f"   Ham: {len(df[df['label']=='ham'])} ({len(df[df['label']=='ham'])/len(df)*100:.1f}%)")

# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing text...")

def clean_text(text):
    """Clean and preprocess email text"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['cleaned_text'] = df['email_text'].apply(clean_text)

print("✅ Text cleaned")

# Extract features
df['char_count'] = df['email_text'].apply(len)
df['word_count'] = df['email_text'].apply(lambda x: len(x.split()))
df['exclamation_count'] = df['email_text'].apply(lambda x: x.count('!'))
df['question_count'] = df['email_text'].apply(lambda x: x.count('?'))
df['uppercase_count'] = df['email_text'].apply(lambda x: sum(1 for c in x if c.isupper()))

print("✅ Features extracted")

# ============================================================================
# 3. EXPLORATORY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("TEXT ANALYSIS")
print("=" * 70)

# Average lengths
spam_avg_len = df[df['label']=='spam']['char_count'].mean()
ham_avg_len = df[df['label']=='ham']['char_count'].mean()

print(f"\nAverage Email Length:")
print(f"   Spam: {spam_avg_len:.0f} characters")
print(f"   Ham: {ham_avg_len:.0f} characters")

# Common words in spam
spam_text = ' '.join(df[df['label']=='spam']['cleaned_text'])
spam_words = [word for word in spam_text.split() if word not in STOPWORDS and len(word) > 3]
spam_common = Counter(spam_words).most_common(15)

print(f"\nTop 15 Words in Spam Emails:")
for word, count in spam_common:
    print(f"   {word}: {count}")

# Common words in ham
ham_text = ' '.join(df[df['label']=='ham']['cleaned_text'])
ham_words = [word for word in ham_text.split() if word not in STOPWORDS and len(word) > 3]
ham_common = Counter(ham_words).most_common(15)

print(f"\nTop 15 Words in Ham Emails:")
for word, count in ham_common:
    print(f"   {word}: {count}")

# ============================================================================
# 4. PREPARE DATA FOR MODELING
# ============================================================================
print("\n🔧 Preparing data for modeling...")

# Convert labels to binary
df['is_spam'] = (df['label'] == 'spam').astype(int)

# Split data
X = df['cleaned_text']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Split data:")
print(f"   Training: {len(X_train)} emails")
print(f"   Testing: {len(X_test)} emails")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Vectorized text (TF-IDF)")
print(f"   Features: {X_train_vec.shape[1]}")

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================
print("\n🤖 Training models...")

# Model 1: Naive Bayes
print("\n   Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_prob = nb_model.predict_proba(X_test_vec)[:, 1]
nb_acc = accuracy_score(y_test, nb_pred)
nb_auc = roc_auc_score(y_test, nb_prob)
print(f"   ✅ Naive Bayes Accuracy: {nb_acc:.4f} | AUC: {nb_auc:.4f}")

# Model 2: Logistic Regression
print("\n   Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_prob = lr_model.predict_proba(X_test_vec)[:, 1]
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)
print(f"   ✅ Logistic Regression Accuracy: {lr_acc:.4f} | AUC: {lr_auc:.4f}")

# Model 3: Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_vec, y_train)
rf_pred = rf_model.predict(X_test_vec)
rf_prob = rf_model.predict_proba(X_test_vec)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
print(f"   ✅ Random Forest Accuracy: {rf_acc:.4f} | AUC: {rf_auc:.4f}")

# Select best model
models = {'Naive Bayes': (nb_model, nb_pred, nb_prob, nb_acc), 
          'Logistic Regression': (lr_model, lr_pred, lr_prob, lr_acc),
          'Random Forest': (rf_model, rf_pred, rf_prob, rf_acc)}

best_model_name = max(models, key=lambda k: models[k][3])
best_model, best_pred, best_prob, best_acc = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_acc:.4f})")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, best_pred)
print(f"\nConfusion Matrix:")
print(cm)

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================
print("\n📊 Creating visualizations...")

os.makedirs('visualizations', exist_ok=True)

plt.style.use('dark_background')

# Visualization 1: Class Distribution
print("   Creating class distribution chart...")
fig, ax = plt.subplots(figsize=(10, 6))
class_counts = df['label'].value_counts()
bars = ax.bar(range(len(class_counts)), class_counts.values, 
              color=NEON_YELLOW, edgecolor='white', linewidth=2)
ax.set_xticks(range(len(class_counts)))
ax.set_xticklabels(['Spam', 'Ham'])
ax.set_ylabel('Number of Emails', fontsize=14, fontweight='bold')
ax.set_title('Email Distribution (Spam vs Ham)', 
             fontsize=16, fontweight='bold', color=NEON_YELLOW, pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(class_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: class_distribution.png")

# Visualization 2: Word Count Comparison
print("   Creating word count comparison...")
fig, ax = plt.subplots(figsize=(12, 6))
spam_wc = df[df['label']=='spam']['word_count']
ham_wc = df[df['label']=='ham']['word_count']
ax.hist(spam_wc, bins=20, alpha=0.7, color='#ff6600', label='Spam', edgecolor='white')
ax.hist(ham_wc, bins=20, alpha=0.7, color=NEON_YELLOW, label='Ham', edgecolor='white')
ax.set_xlabel('Word Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Word Count Distribution: Spam vs Ham', 
             fontsize=16, fontweight='bold', color=NEON_YELLOW, pad=20)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/word_count_distribution.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: word_count_distribution.png")

# Visualization 3: Top Spam Words
print("   Creating top spam words chart...")
fig, ax = plt.subplots(figsize=(12, 8))
words, counts = zip(*spam_common)
bars = ax.barh(range(len(words)), counts, color=NEON_YELLOW, edgecolor='white', linewidth=2)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xlabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Top 15 Words in Spam Emails', 
             fontsize=16, fontweight='bold', color=NEON_YELLOW, pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/spam_words.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: spam_words.png")

# Visualization 4: Confusion Matrix
print("   Creating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=[[0, 0, 0], [1, 1, 0]], 
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actually Ham', 'Actually Spam'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', 
             color=NEON_YELLOW, pad=20)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: confusion_matrix.png")

# Visualization 5: Model Comparison
print("   Creating model comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(models.keys())
accuracies = [models[name][3] for name in model_names]
bars = ax.bar(range(len(model_names)), accuracies, 
              color=NEON_YELLOW, edgecolor='white', linewidth=2)
# Highlight best model
best_idx = model_names.index(best_model_name)
bars[best_idx].set_color('#00ff00')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', 
             fontsize=16, fontweight='bold', color=NEON_YELLOW, pad=20)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("✅ Saved: model_comparison.png")

# ============================================================================
# 8. SAVE MODEL
# ============================================================================
print("\n💾 Saving model...")

os.makedirs('model', exist_ok=True)

with open('model/spam_detector.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

metadata = {
    'model_name': best_model_name,
    'accuracy': best_acc,
    'total_emails': len(df),
    'features': X_train_vec.shape[1]
}

with open('model/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Saved model and vectorizer")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)

print(f"\n✅ Analyzed {len(df)} emails")
print(f"✅ Created 5 visualizations")
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Accuracy: {best_acc:.2%}")
print(f"✅ Model saved and ready for deployment")

print("\n🚀 Spam detector ready to use!")
