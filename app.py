# Copyright (c) 2025 Ujjwal Kumar
# Licensed under the MIT License. See LICENSE file in the project root for details.
import re
import random
import pandas as pd
import numpy as np
import csv
import streamlit as st
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="AI Health Chatbot", layout="centered")

# ------------------ Utility: cached loaders & model training ------------------
@st.cache_data(ttl=3600)
def load_data():
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')

    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
    training = training.loc[:, ~training.columns.duplicated()]
    testing = testing.loc[:, ~testing.columns.duplicated()]

    return training, testing

@st.cache_data(ttl=3600)
def load_master_files():
    description_list = {}
    severityDictionary = {}
    precautionDictionary = {}

    try:
        with open('MasterData/symptom_Description.csv', encoding='utf-8') as csv_file:
            for row in csv.reader(csv_file):
                if len(row) >= 2:
                    description_list[row[0]] = row[1]
    except FileNotFoundError:
        pass

    try:
        with open('MasterData/symptom_severity.csv', encoding='utf-8') as csv_file:
            for row in csv.reader(csv_file):
                try:
                    severityDictionary[row[0]] = int(row[1])
                except:
                    pass
    except FileNotFoundError:
        pass

    try:
        with open('MasterData/symptom_precaution.csv', encoding='utf-8') as csv_file:
            for row in csv.reader(csv_file):
                if len(row) >= 5:
                    precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    except FileNotFoundError:
        pass

    return description_list, severityDictionary, precautionDictionary

@st.cache_resource
def train_model(training):
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    return model, le, cols

# ------------------ Symptom Synonyms (auto-expanded) ------------------
# ------------------ Symptom Synonyms (auto-expanded) ------------------
symptom_synonyms = {}

def add_synonyms(base_text, original_symptom, variants):
    """Helper to map multiple human-readable variants to the same symptom key."""
    for v in variants:
        symptom_synonyms[v.lower()] = original_symptom

for s in cols:  # cols is now available ✅
    human_text = s.replace("_", " ").lower()
    symptom_synonyms[human_text] = s

    # Common synonym patterns
    if "pain" in human_text:
        add_synonyms(human_text, s, [
            human_text.replace("pain", "ache"),
            human_text.replace("pain", "discomfort"),
            human_text.replace("pain", "soreness"),
            "ache", "hurting", "painful"
        ])
    if "fever" in human_text:
        add_synonyms(human_text, s, [
            "temperature", "high temperature", "pyrexia", "feverish"
        ])
    if "cough" in human_text:
        add_synonyms(human_text, s, [
            "coughing", "dry cough", "wet cough", "continuous cough"
        ])
    if "diarrhea" in human_text:
        add_synonyms(human_text, s, [
            "loose motion", "motions", "loose stools", "runny tummy"
        ])
    if "chills" in human_text:
        add_synonyms(human_text, s, ["cold", "shivering"])
    if "breathlessness" in human_text or "shortness of breath" in human_text:
        add_synonyms(human_text, s, [
            "breathing issue", "difficulty breathing", "breath short", "breath problem"
        ])
    if "vomit" in human_text:
        add_synonyms(human_text, s, ["throwing up", "puking", "nausea"])
    if "headache" in human_text or "head ache" in human_text:
        add_synonyms(human_text, s, ["head pain", "migraine"])
    if "fatigue" in human_text:
        add_synonyms(human_text, s, ["tiredness", "exhaustion", "weakness", "low energy"])
    if "sore throat" in human_text or "throat" in human_text:
        add_synonyms(human_text, s, ["throat pain", "throat irritation", "pharyngitis"])
    if "runny nose" in human_text or "nasal" in human_text:
        add_synonyms(human_text, s, ["blocked nose", "stuffy nose", "congestion"])
    if "rash" in human_text:
        add_synonyms(human_text, s, ["skin eruption", "spots", "skin patches"])
    if "anxiety" in human_text:
        add_synonyms(human_text, s, ["nervousness", "worry", "restlessness"])
    if "depression" in human_text:
        add_synonyms(human_text, s, ["sadness", "low mood", "feeling down"])
    if "itch" in human_text:
        add_synonyms(human_text, s, ["pruritus", "skin irritation", "itching"])


def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = str(user_input).lower().replace("-", " ")

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    all_symptom_words = [s.replace("_", " ") for s in all_symptoms]
    for word in words:
        close = get_close_matches(word, all_symptom_words, n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

def predict_disease(symptoms_list, symptoms_dict, model, le, feature_columns):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    input_df = pd.DataFrame([input_vector], columns=feature_columns)
    pred_proba = model.predict_proba(input_df)[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence, pred_proba

# ------------------ App UI ------------------
st.title("🤖 AI Health Chatbot (Web)")
st.write("Enter your basic info and symptoms. The model will suggest a likely condition and precautions. This is for informational purposes only — not a substitute for professional medical advice.")

with st.spinner("Loading data and preparing model..."):
    try:
        training, testing = load_data()
    except Exception as e:
        st.error(f"Error loading Data CSVs: {e}")
        st.stop()

    description_list, severityDictionary, precautionDictionary = load_master_files()
    model, le, cols = train_model(training)

st.info(f"✅ Training data loaded: {training.shape[0]} rows × {training.shape[1]} columns. Features: {len(cols)}")

symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

# ------------------ User Input ------------------
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("👉 What is your name?")
with col2:
    age = st.number_input("👉 Please enter your age", min_value=0, max_value=120, value=25)

gender = st.selectbox("👉 What is your gender?", options=["Prefer not to say", "Male", "Female", "Other"])

symptoms_input = st.text_area("👉 Describe your symptoms in a sentence (e.g., 'I have fever and stomach pain')", height=100)

detect_btn = st.button("🔎 Detect Symptoms")

if "detected" not in st.session_state:
    st.session_state.detected = []
if "final_prediction" not in st.session_state:
    st.session_state.final_prediction = None
if "extra_info" not in st.session_state:
    st.session_state.extra_info = {}

# ------------------ Detect & Predict ------------------
if detect_btn:
    detected = extract_symptoms(symptoms_input, cols)
    st.session_state.detected = detected

    if not detected:
        st.error("❌ Sorry, I could not detect valid symptoms. Please try again with more specific symptoms (e.g., 'fever and cough').")
    else:
        st.success(f"✅ Detected symptoms: {', '.join(detected)}")

        # Immediate prediction
        disease2, confidence2, proba2 = predict_disease(detected, symptoms_dict, model, le, cols)
        st.session_state.final_prediction = (disease2, confidence2, proba2)

        # Collect extra info
        with st.form(key="follow_up_form"):
            num_days = st.number_input(
                "👉 For how many days have you had these symptoms?", 
                min_value=0, max_value=365, value=1
            )
            severity_scale = st.slider(
                "👉 On a scale of 1–10, how severe do you feel your condition is?", 
                1, 10, 5
            )
            pre_exist = st.text_input(
                "👉 Do you have any pre-existing conditions (e.g., diabetes, hypertension)?",
                placeholder="Type 'None' if no condition"
            )
            lifestyle = st.text_input(
                "👉 Do you smoke, drink alcohol, or have irregular sleep?",
                placeholder="Type 'None' if no lifestyle issues"
            )
            family = st.text_input(
                "👉 Any family history of similar illness?",
                placeholder="Type 'None' if no family history"
            )
            submit_followup = st.form_submit_button("✅ Update info (optional)")

        if submit_followup:
            st.session_state.extra_info = {
                "num_days": num_days,
                "severity_scale": severity_scale,
                "pre_exist": pre_exist,
                "lifestyle": lifestyle,
                "family": family
            }

# ------------------ Show Results ------------------
if st.session_state.final_prediction:
    disease2, confidence2, proba2 = st.session_state.final_prediction
    st.markdown("## 🏁 Final Result")
    st.write(f"🩺 **Based on your answers, you may have:** **{disease2}**")
    #st.write(f"🔎 **Confidence:** {confidence2}%")
    st.write(f"📋 **Detected symptoms used:** {', '.join(st.session_state.detected)}")
    st.write(f"📖 **About:** {description_list.get(disease2, 'No description available.')}")

    if disease2 in precautionDictionary:
        st.write("🛡️ **Suggested precautions:**")
        for i, prec in enumerate(precautionDictionary[disease2], 1):
            st.write(f"{i}. {prec}")

    # Show extra info if provided
    if st.session_state.extra_info:
        st.write("📝 **Your extra details:**")
        for k, v in st.session_state.extra_info.items():
            st.write(f"- {k.replace('_',' ').title()}: {v}")

    st.success(random.choice([
        "🌸 Health is wealth, take care of yourself.",
        "💪 A healthy outside starts from the inside.",
        "☀️ Every day is a chance to get stronger and healthier.",
        "🌿 Take a deep breath, your health matters the most.",
        "🌺 Remember, self-care is not selfish."
    ]))
    st.balloons()

st.write("---")
st.write("⚠️ *This tool is for educational and informational purposes only and is not a substitute for professional medical advice.*")
