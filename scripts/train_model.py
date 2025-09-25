# scripts/train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scripts.preprocess import preprocess_numeric, preprocess_categorical, preprocess_text
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Dropout

# --------------------------
# 1. Load Dataset
# --------------------------
df = pd.read_csv('data/strong_marketing_campaign.csv')

# --------------------------
# 2. Define Features
# --------------------------
numeric_features = ["PastClicks", "PastPurchases", "PreviousResponse", "CustomerLifetimeValue"]
categorical_features = ["Channel", "CampaignType"]
text_feature = "CampaignText"
target = "Response"

X = df[numeric_features + categorical_features + [text_feature]]
y = df[target]

# --------------------------
# 3. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# 4. Preprocessing
# --------------------------
X_train_num, scaler = preprocess_numeric(X_train, numeric_features)
X_test_num, _ = preprocess_numeric(X_test, numeric_features, scaler)

X_train_cat, ohe = preprocess_categorical(X_train, categorical_features)
X_test_cat, _ = preprocess_categorical(X_test, categorical_features, ohe)

X_train_text, tokenizer = preprocess_text(X_train, text_feature)
X_test_text, _ = preprocess_text(X_test, text_feature, tokenizer)

# --------------------------
# 5. Build DL Fusion Model
# --------------------------
max_words = 5000
max_len = 30

# Numeric Input
num_input = Input(shape=(X_train_num.shape[1],), name='numeric_input')
num_dense = Dense(64, activation='relu')(num_input)

# Categorical Input
cat_input = Input(shape=(X_train_cat.shape[1],), name='categorical_input')
cat_dense = Dense(32, activation='relu')(cat_input)

# Text Input
text_input = Input(shape=(max_len,), name='text_input')
embedding = Embedding(input_dim=max_words, output_dim=64, input_length=max_len)(text_input)
lstm = LSTM(64)(embedding)

# Fusion Layer
fusion = Concatenate()([num_dense, cat_dense, lstm])
fusion = Dense(64, activation='relu')(fusion)
fusion = Dropout(0.3)(fusion)
output = Dense(1, activation='sigmoid')(fusion)

model = Model(inputs=[num_input, cat_input, text_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# 6. Train Model
# --------------------------
history = model.fit(
    [X_train_num, X_train_cat, X_train_text],
    y_train,
    validation_data=([X_test_num, X_test_cat, X_test_text], y_test),
    epochs=10,
    batch_size=32
)

# --------------------------
# 7. Save Model and Preprocessing Objects
# --------------------------
model.save('models/marketing_fusion_model.h5')

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/ohe_encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)

with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and preprocessing objects saved successfully.")
