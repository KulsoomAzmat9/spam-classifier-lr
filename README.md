# Email Spam Classifier using Logistic Regression

This is a Machine Learning project that classifies emails as Spam or Not Spam using Logistic Regression and TF-IDF Vectorization.

## 📌 Project Overview
Spam detection is a binary text classification problem. This project trains a Logistic Regression model on an email dataset and provides a web interface to test new email messages.

## ⚙️ Features
- Text preprocessing: lowercase, punctuation removal, stopword removal
- TF-IDF feature extraction
- Logistic Regression classifier with regularization
- Model and vectorizer saved using joblib

## 🛠️ Tech Stack
- Language: Python 
- Libraries: scikit-learn, pandas, numpy, joblib
- Vectorizer: TfidfVectorizer
- Model: LogisticRegression

## 📊 Algorithm Comparison

We compared 3 algorithms for spam detection on the same dataset:

### 1. Logistic Regression - Final Model Used
  Type: Linear model for classification
  Pros: Simple, interpretable, gives probability scores, works well with regularization
  Accuracy on my dataset: 77.78%
  Reason for selection: Good baseline model and easy to interpret

### 2. Naive Bayes
- *Type*: Probabilistic model based on Bayes' Theorem
- *Pros*: Very fast, works great with text data
- *Accuracy on my dataset: **100%*
- *Observation*: Performed better than LR on this dataset

### 3. Support Vector Machine (SVM)
- *Type*: Finds optimal hyperplane to separate classes
- *Pros*: High accuracy on text data
- *Note*: Not tested due to time constraints

### Comparison Bar Graph
The bar graph shows Accuracy comparison of all 3 algorithms.

### Conclusion
Logistic Regression achieved 77.78% accuracy on the dataset. While Naive Bayes performed better with 100%, Logistic Regression is still a strong baseline model. With hyperparameter tuning and more data, LR performance can be improved. For this project LR was implemented to compare with NB.
