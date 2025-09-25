# scripts/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -----------------------------
# Preprocess Numeric Features
# -----------------------------
def preprocess_numeric(df, numeric_features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[numeric_features])
    else:
        X_num = scaler.transform(df[numeric_features])
    return X_num, scaler


# -----------------------------
# Preprocess Categorical Features
# -----------------------------
def preprocess_categorical(df, categorical_features, encoder=None):
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore" , sparse_output=False)
        X_cat = encoder.fit_transform(df[categorical_features])
    else:
        X_cat = encoder.transform(df[categorical_features])
    return X_cat, encoder


# -----------------------------
# Preprocess Text Features
# -----------------------------
def preprocess_text(df, text_feature, tokenizer=None, max_len=30, max_words=5000):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(df[text_feature])
    sequences = tokenizer.texts_to_sequences(df[text_feature])
    # Make sure max_len is applied here
    X_text = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return X_text, tokenizer


