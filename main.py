import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_FILE = 'trained_model.joblib'
VECTORIZER_FILE = 'vectorizer.joblib'

def train_and_save_model():
    print("Training new model...")

    # Step 1: Load general dataset
    general_df = pd.read_csv('./generalizedDatasetEmails.csv')
    X_general = general_df['Text']
    y_general = general_df['Label']

    # Step 2: Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X_general_vectors = vectorizer.fit_transform(X_general)

    # Step 3: Train on general data
    model = MultinomialNB()
    model.fit(X_general_vectors, y_general)

    # Step 4: Fine-tune on college-specific dataset
    college_df = pd.read_csv('./TestDataSet.csv')
    X_college = college_df['Text']
    y_college = college_df['Label']
    X_college_vectors = vectorizer.transform(X_college)
    model.fit(X_college_vectors, y_college)


    # Step 5: Save model and vectorizer
    # joblib.dump(model, MODEL_FILE)
    # joblib.dump(vectorizer, VECTORIZER_FILE)

    print("Model trained and saved.")
    return model, vectorizer

def load_model_and_vectorizer():
    print("Loading existing model...")
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

def classify_email(model, vectorizer, text, threshold=0.4):
    vect = vectorizer.transform([text])
    probs = model.predict_proba(vect)[0]
    max_prob = probs.max()
    predicted_class = model.classes_[probs.argmax()]

    if max_prob < threshold:
        return f"defaulted to 'other' : {predicted_class} (failed threshold check) ({max_prob:.2f})"
    return f"{predicted_class} ({max_prob:.2f})"

def main():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model, vectorizer = load_model_and_vectorizer()
    else:
        model, vectorizer = train_and_save_model()

    while True:
        user_input = input("\nEnter an email to classify (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        label = classify_email(model, vectorizer, user_input)
        print(f"Predicted Category: {label}")

if __name__ == "__main__":
    main()
