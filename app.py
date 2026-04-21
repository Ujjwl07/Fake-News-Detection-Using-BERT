import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= CONFIG =================
MODEL_DIR = "model"  # your saved model folder
MAX_LEN = 64

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# ================= PREDICTION =================
def predict_news(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence

# ================= UI =================
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or article to check if it is **Fake or Real**.")

# Input
user_input = st.text_area("Enter News Text:", height=150)

# Button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred, confidence = predict_news(user_input)

        if pred == 0:
            st.error(f"❌ FAKE NEWS (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ REAL NEWS (Confidence: {confidence:.2f})")

# Footer
st.markdown("---")
st.caption("Built using BERT (Transformers) + Streamlit")