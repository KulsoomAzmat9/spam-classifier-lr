import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Spam Classifier - Logistic Regression")

# 1. Load data
df = pd.read_csv("email_data.csv")

# 2. Split text and label
X = df['text']
y = df['label']

# 3. Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)
# 4. Split train/test
# 5. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)
y_pred = model.predict(X_vec)
accuracy = accuracy_score(y, y_pred)
# 6. Show accuracy
st.write(f"*Accuracy: {acc*100:.2f}%*")

# 7. Test input
user_input = st.text_input("Enter an email to test:")
if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    st.write("Prediction:", prediction)
