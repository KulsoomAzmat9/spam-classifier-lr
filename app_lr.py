import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("📧 Spam Classifier - Logistic Regression")
st.write("This model uses Logistic Regression to detect SPAM vs NOT SPAM emails")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("email_data.csv")
    return df

df = load_data()

# 2. Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) # <-- This is the only change from NB
model.fit(X_train, y_train)

# 3. Show Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {acc:.2f}")

# 4. Prediction Box
st.subheader("Test Your Own Email")
user_input = st.text_area("Enter email text here:")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]

    if prediction == "SPAM":
        st.error(f"🚨 This is: {prediction}")
    else:
        st.success(f"✅ This is: {prediction}")