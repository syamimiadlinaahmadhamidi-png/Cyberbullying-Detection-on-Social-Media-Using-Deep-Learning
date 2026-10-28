from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
import pickle
import re
import nltk
import nltk



from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Download NLTK resources
# ==========================

nltk.download("punkt")

# ==========================
# Load Model & Tokenizer
# ==========================
model = tf.keras.models.load_model("best_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Same max length used during training
MAX_LEN = 100



# ==========================
# Text Preprocessing
# ==========================
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    # KEEP stopwords
    return " ".join(tokens)

# ==========================
# FastAPI
# ==========================
app = FastAPI(title="CyberShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Request Model
# ==========================
class PredictionRequest(BaseModel):
    text: str

# ==========================
# Home Endpoint
# ==========================
@app.get("/")
def home():
    return {
        "message": "CyberShield API is running."
    }

# ==========================
# Prediction Endpoint
# ==========================
@app.post("/predict")
def predict(data: PredictionRequest):

    # Original input
    original_text = data.text

    # Clean text
    cleaned_text = clean_text(original_text)

    print("=" * 50)
    print("Original Text :", original_text)
    print("Cleaned Text  :", cleaned_text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict
    probability = float(model.predict(padded, verbose=0)[0][0])

    print("Raw Probability:", probability)

    # Binary classification
    if probability >= 0.30:
        label = "Cyberbullying"
        confidence = probability
    else:
        label = "Not Cyberbullying"
        confidence = 1 - probability

    print("Prediction:", label)
    print("Confidence:", confidence)
    print("=" * 50)

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }