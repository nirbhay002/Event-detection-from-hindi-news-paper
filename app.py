import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import random
import xml.etree.ElementTree as ET
from utils import get_entity_info  # make sure this exists and works

# Identity functions needed for loading model if they were used in pickling

def identity_preprocessor(x):
    return x

def identity_tokenizer(x):
    return x

# Load models
crf_model = joblib.load("models/task1_sequence_labeler.crf")
clf_model = joblib.load("models/task2_event_classifier.pkl")
vectorizer = clf_model.named_steps['tfidf']

# Load example directory
EXAMPLE_DIR = "data/raw/Hindi_Test"

def extract_text_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        paragraphs = [elem.text.strip() for elem in root.findall(".//P") if elem.text]
        return "\n\n".join(paragraphs)
    except Exception as e:
        return f"⚠️ Error reading {file_path}: {str(e)}"

def get_all_xml_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".xml")]

# --- Session State Initialization ---
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if "example_filename" not in st.session_state:
    st.session_state.example_filename = ""

# --- Title and Input ---
st.title("📰 Hindi News Event Detection")
st.markdown("Upload or paste a Hindi news article below to extract events (Task 1) and classify the event type (Task 2).")

# --- Generate Example Text Button ---
# col_gen, _ = st.columns([1, 3])
# with col_gen:
if st.button("✨ Generate Example Text"):
    all_files = get_all_xml_files(EXAMPLE_DIR)
    if all_files:
        random_file = random.choice(all_files)
        st.session_state.generated_text = extract_text_from_xml(random_file)
        st.session_state.example_filename = os.path.basename(random_file)
        st.rerun()

# --- Input Text Area ---
st.text_area("✍️ Paste Hindi News Text Here:", value=st.session_state.generated_text, height=200, key="input_text")

if st.session_state.example_filename:
    st.markdown(f"**📂 Example File Used:** `{st.session_state.example_filename}`")
    #print(f"📂 Example File Used: {st.session_state.example_filename}")

# --- Tokenizer ---
def simple_tokenize(text):
    return text.strip().split()

# --- CRF Entity Tagging ---
def predict_bio_crf(text):
    tokens = simple_tokenize(text)
    features = [{
        'bias': 1.0,
        'word.lower()': token.lower(),
        'word[-3:]': token[-3:],
        'word[-2:]': token[-2:],
        'word.isupper()': token.isupper(),
        'word.istitle()': token.istitle(),
        'word.isdigit()': token.isdigit(),
        '-1:word.lower()': tokens[i-1].lower() if i > 0 else '',
        '+1:word.lower()': tokens[i+1].lower() if i < len(tokens)-1 else '',
        'BOS': i == 0,
        'EOS': i == len(tokens)-1
    } for i, token in enumerate(tokens)]
    predicted_tags = crf_model.predict([features])[0]
    return list(zip(tokens, predicted_tags))

# --- Classifier ---
def predict_event_class(text):
    tokens = simple_tokenize(text)
    transformed_text = clf_model.named_steps["tfidf"].transform([tokens])
    prediction = clf_model.named_steps["model"].predict(transformed_text)[0]
    proba = clf_model.named_steps["model"].predict_proba(transformed_text)[0]
    class_labels = clf_model.named_steps["model"].classes_
    return prediction, dict(zip(class_labels, proba))

# --- Extract Entity Spans ---
def extract_all_entities(tagged_tokens):
    entities = {}
    current_tokens = []
    current_label = None
    for token, label in tagged_tokens:
        if label == "O":
            if current_tokens and current_label:
                entities.setdefault(current_label, []).append(" ".join(current_tokens))
            current_tokens = []
            current_label = None
        elif label.startswith("B-"):
            if current_tokens and current_label:
                entities.setdefault(current_label, []).append(" ".join(current_tokens))
            current_label = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_label:
            current_tokens.append(token)
        else:
            current_tokens = []
            current_label = None
    if current_tokens and current_label:
        entities.setdefault(current_label, []).append(" ".join(current_tokens))
    return entities

# --- Predict Button ---
if st.button("🔍 Predict"):
    if st.session_state.input_text.strip():
        st.subheader("📌 Task 1: Entity Extraction")
        bio_results = predict_bio_crf(st.session_state.input_text)
        bio_df = pd.DataFrame(bio_results, columns=["Token", "Label"])
        st.dataframe(bio_df)

        st.markdown("### 🧾 Entity Summary for Easy Understanding")
        all_entities = extract_all_entities(bio_results)
        if all_entities:
            for label_type, spans in all_entities.items():
                st.markdown(f"**{label_type}**: {', '.join(spans)}")
        else:
            st.info("No entities detected in the text.")

        st.subheader("🧠 Task 2: Event Classification")
        pred_label, pred_probs = predict_event_class(st.session_state.input_text)
        st.success(f"Predicted Event Type: **{pred_label}**")

        prob_df = pd.DataFrame({
            "Class": list(pred_probs.keys()),
            "Probability": list(pred_probs.values())
        }).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))
    else:
        st.warning("⚠️ Please enter some text to analyze.")