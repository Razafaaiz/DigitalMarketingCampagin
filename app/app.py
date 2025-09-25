# app/app.py

# --------------------------
# 1️⃣ Set page config first
# --------------------------
import streamlit as st
st.set_page_config(
    page_title="Marketing Campaign Predictor",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# 2️⃣ Import required libraries
# --------------------------
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# --------------------------
# 3️⃣ Load model and preprocessing objects
# --------------------------
@st.cache_resource
def load_model_objects():
    model = tf.keras.models.load_model('models/marketing_fusion_model.h5')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/ohe_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, scaler, ohe, tokenizer

model, scaler, ohe, tokenizer = load_model_objects()
max_len = 30

# --------------------------
# 4️⃣ App Header & Instructions
# --------------------------
st.title("📊 Digital Marketing Campaign Predictor")
st.markdown("""
Predict if a customer is likely to respond to a marketing campaign using **ML + Deep Learning**.  
Fill in the customer details and campaign text to get **real-time predictions**!
""")

# --------------------------
# 5️⃣ Sidebar Inputs
# --------------------------
st.sidebar.header("Customer & Campaign Inputs")

# Numeric Inputs
st.sidebar.subheader("Customer Data")
PastClicks = st.sidebar.number_input("Past Clicks", 0, 50, 2)
PastPurchases = st.sidebar.number_input("Past Purchases", 0, 20, 1)
PreviousResponse = st.sidebar.number_input("Previous Responses", 0, 10, 1)
CustomerLifetimeValue = st.sidebar.number_input("Customer Lifetime Value", 0.0, 10000.0, 500.0)

# Categorical Inputs
st.sidebar.subheader("Campaign Info")
Channel = st.sidebar.selectbox("Channel", ["Email", "Social", "Push", "SMS", "Web Ads"])
CampaignType = st.sidebar.selectbox("Campaign Type", ["Discount", "Loyalty Reward", "New Product Launch", "Flash Sale"])

# Text Input
CampaignText = st.sidebar.text_area("Campaign Text", "Limited time offer: 20% off today!")

# --------------------------
# 6️⃣ Prediction
# --------------------------
if st.button("Predict Response ✅"):
    with st.spinner("Analyzing campaign response..."):
        time.sleep(1)  # small delay for UX
        
        # Preprocess inputs
        X_num = scaler.transform(np.array([[PastClicks, PastPurchases, PreviousResponse, CustomerLifetimeValue]]))
        X_cat = ohe.transform([[Channel, CampaignType]])
        seq = tokenizer.texts_to_sequences([CampaignText])
        X_text = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        
        # Predict probability
        prob = float(model.predict([X_num, X_cat, X_text])[0][0])  # cast to Python float
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results 📊")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Probability", f"{prob:.2f}")
        if prob >= 0.5:
            col2.success("✅ Customer likely to respond!")
        else:
            col2.warning("❌ Customer unlikely to respond!")
        
        # Progress bar (scaled 0-100)
        st.progress(int(prob * 100))
        
        st.markdown("**💡 Tip:** Use this prediction to optimize your marketing campaigns and target customers effectively.")

# --------------------------
# 7️⃣ Footer
# --------------------------
st.markdown("""
---
Made by FAIZ RAZA.  
Powered by **Python, TensorFlow & Streamlit**.
""")


