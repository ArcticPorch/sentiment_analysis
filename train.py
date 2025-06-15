import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.preprocess import preprocess_data

def train_model():
    df = pd.read_excel("data/train_ps3.xlsx")
    df = preprocess_data(df)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df.iloc[:, 3].values

    roles = pd.get_dummies(df.iloc[:, 1])
    X_combined = np.hstack([X_text, roles.values])

    X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_combined.shape[1]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    model.save("sentiment_model.h5")
    return vectorizer, roles.columns.tolist()
