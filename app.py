import streamlit as st
import pandas as pd
import time
import re
import docx
from PyPDF2 import PdfReader
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sentiment_rules import apply_sentiment_rules, rule_based_training_examples, clean_for_rules

# Run: streamlit run app.py

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)


# SPLASH SCREEN
def show_splash_screen():
    st.markdown(
        """
        <style>
        .splash-box {
            padding: 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e5e7eb;
            text-align: center;
            margin-top: 3rem;
        }
        .splash-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .splash-subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="splash-box">
            <div class="splash-title">Sentiment Analysis System</div>
            <div class="splash-subtitle">
                A tool that classifies sentences into positive, neutral, or negative categories using a LinearSVC algorithm trained on labeled datasets, enhanced with a rule-based correction system.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("Start Analysis", type="primary", use_container_width=True):
        st.session_state["show_app"] = True
        st.rerun()

if "show_app" not in st.session_state:
    st.session_state["show_app"] = False

if not st.session_state["show_app"]:
    show_splash_screen()
    st.stop()


# TEXT PREPROCESSING
def clean_text(text: str) -> str:
    return clean_for_rules(text)


# MODEL TRAINING
@st.cache_resource
def train_model():
    with st.spinner('Training model.'):
        try:
            try:
                df_old = pd.read_csv('dataset/sentiment_analysis.csv')[['text', 'sentiment']]
            except:
                df_old = pd.DataFrame(columns=['text', 'sentiment'])

            df_new = pd.read_csv('dataset/sentiment_data.csv')

            label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df_new['Sentiment'] = df_new['Sentiment'].map(label_mapping)
            df_new = df_new.rename(columns={'Comment': 'text', 'Sentiment': 'sentiment'})[['text', 'sentiment']].dropna()

            counts = df_new['sentiment'].value_counts()
            min_count = counts.min()

            df_new_sampled = df_new.groupby('sentiment').sample(n=min_count, random_state=42).reset_index(drop=True)

            df_rules = rule_based_training_examples()

            df_final = pd.concat([df_old, df_new_sampled, df_rules], ignore_index=True)
            df_final = df_final.dropna(subset=["text", "sentiment"])

            df_final["sentiment"] = df_final["sentiment"].str.lower().str.strip()
            df_final = df_final[df_final["sentiment"].isin(["negative", "neutral", "positive"])]

            df_final['text_clean'] = df_final['text'].apply(clean_text)
            df_final = df_final[df_final['text_clean'].str.len() > 2]

            vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=120000, sublinear_tf=True, min_df=1)
            X = vectorizer.fit_transform(df_final['text_clean'])
            y = df_final['sentiment']

            model = LinearSVC(class_weight='balanced', random_state=42, max_iter=3000)
            model.fit(X, y)

            return model, vectorizer, len(df_final)
        
        except Exception as e:
            st.error(f"Training error: {e}")
            return None, None, 0
        
model, vectorizer, data_count = train_model()

st.title("Sentiment Analysis System")
st.caption("Using LinearSVC and rule-based correction.")

if model:
    st.success(f"Training model succeeded with {data_count:,} rows.")
else:
    st.warning("Model failed.")
    st.stop()

st.divider()


# FILE EXTRACTION
def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")

def extract_text_from_pdf(uploaded_file) -> str:
    if PdfReader is None:
        st.error("PyPDF2 is not installed.")
        return ""

    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
        
    raw_text = "\n".join(pages)
    cleaned_text = re.sub(r"\n+", " ", raw_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def extract_text_from_docx(uploaded_file) -> str:
    if docx is None:
        st.error("python-docx is not installed.")
        return ""

    document = docx.Document(uploaded_file)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    st.error("Unsupported file type. Please upload TXT, DOCX, or PDF.")
    return ""


# PREDICTION HELPERS
def predict_sentiment(text: str) -> Tuple[str, str, List[str]]:
    cleaned_text = clean_text(text)

    if cleaned_text.strip() == "":
        return "neutral", cleaned_text, ["Input text is empty after cleaning."]

    vec_input = vectorizer.transform([cleaned_text])
    raw_prediction = model.predict(vec_input)[0]
    final_prediction, rule_reasons = apply_sentiment_rules(text, raw_prediction)

    return final_prediction, cleaned_text, rule_reasons

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

def analyze_long_text(text: str) -> pd.DataFrame:
    sentences = split_into_sentences(text)

    if not sentences:
        return pd.DataFrame(columns=["sentence", "sentiment", "rule_reason"])

    rows = []
    for sentence in sentences:
        sentiment, _, reasons = predict_sentiment(sentence)
        rows.append({
            "sentence": sentence,
            "sentiment": sentiment,
            "rule_reason": "; ".join(reasons) if reasons else ""
        })

    return pd.DataFrame(rows)


# MAIN APP
input_method = st.radio(
    "Choose text input method:",
    ["Paste Text", "Upload TXT / DOCX / PDF"],
    horizontal=True
)

full_text = ""

if input_method == "Paste Text":
    full_text = st.text_area(
        "Paste your full text here:",
        height = 220,
        placeholder="Paste a paragraph, article, or long text.",
        help="Hint: Separate multiple sentences using standard punctuation (. ! ?)."
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a TXT, DOCX, or PDF file.",
        type=["txt", "docx", "pdf"]
    )

    if uploaded_file is not None:
        full_text = extract_text_from_file(uploaded_file)

        with st.expander("Preview extracted text"):
            preview = full_text[:3000]
            st.text(preview if preview else "No text extracted.")
    
if st.button("Analyze Full Text", type="primary"):
    if full_text.strip() == "":
        st.warning("Please input or upload text first.")
    else:
        result_df = analyze_long_text(full_text)

        if result_df.empty:
            st.warning("No valid sentences found.")
        else:
            st.subheader("Summary")

            sentiment_counts = result_df["sentiment"].value_counts()
            total = len(result_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", int(sentiment_counts.get("positive", 0)))
            col2.metric("Neutral", int(sentiment_counts.get("neutral", 0)))
            col3.metric("Negative", int(sentiment_counts.get("negative", 0)))

            st.subheader("Sentence-level Result")

            display_df = result_df.drop(columns=["rule_reason"], errors="ignore")
            display_df.index = display_df.index + 1
            st.dataframe(display_df, use_container_width=True)

            csv = display_df.to_csv(index=True, index_label="no").encode("utf-8")

st.divider()

if st.button("Clear Cache"):
    st.cache_resource.clear()
    st.rerun()