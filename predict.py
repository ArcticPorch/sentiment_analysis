import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess_data

def predict():
    df_test = pd.read_excel("data/test_ps3.xlsx")
    df_test = preprocess_data(df_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(df_test['clean_text']).toarray()

    roles = pd.get_dummies(df_test.iloc[:, 1])
    X_combined = np.hstack([X_text, roles.values])

    model = load_model("sentiment_model.h5")
    predictions = model.predict(X_combined)
    predicted_labels = (predictions > 0.5).astype(int)

    print(predicted_labels)
