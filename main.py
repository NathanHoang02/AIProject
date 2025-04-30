import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


MODEL_FILE = 'trained_model.joblib'
VECTORIZER_FILE = 'vectorizer.joblib'

def train_and_save_model():
    print("Training new model...")

    # Step 1: Load general dataset
    general_df = pd.read_csv('./generalizedDatasetFinal.csv')
    general_df['Label'] = general_df['Label'].str.strip()
    X_general = general_df['Text']
    y_general = general_df['Label']

    # Step 2: Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X_general_vectors = vectorizer.fit_transform(X_general)

    # Step 3: Train on general data
    model = MultinomialNB()
    model.fit(X_general_vectors, y_general)

    # Step 4: Fine-tune on college-specific dataset
    college_df = pd.read_csv('./fineTuningDatasetFinalized.csv')
    college_df['Label'] = college_df['Label'].str.strip()
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

def evaluate_model(model, vectorizer, test_file):
    # Load the test dataset
    test_df = pd.read_csv(test_file)
    test_df['Label'] = test_df['Label'].str.strip()
    X_test = test_df['Text']
    y_test = test_df['Label']

    # Transform the test data using the vectorizer
    X_test_vectors = vectorizer.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_vectors)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # print("Model Classes:", model.classes_)
    # print("Unique Labels in Test Data:", y_test.unique())

    # # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model, vectorizer = load_model_and_vectorizer()
    else:
        model, vectorizer = train_and_save_model()

    while True:
        print("\nChoose an option:")
        print("1. Evaluate the model using the test dataset")
        print("2. Manually test the model with custom input")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            # Evaluate the model using the specific dataset
            evaluate_model(model, vectorizer, './testDataset2.csv')
        elif choice == '2':
            # Manually test the model
            while True:
                user_input = input("\nEnter an email to classify (or type 'exit' to go back): ")
                if user_input.lower() == 'exit':
                    break
                label = classify_email(model, vectorizer, user_input)
                print(f"Predicted Category: {label}")
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
