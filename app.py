import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# streamlit run app.py

st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="centered"
)

st.title("Sentiment Analysis System")
st.markdown("Using LinearSVC and labeled datasets")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def train_model():
    with st.spinner('Training model.'):
        try:
            try:
                df_old = pd.read_csv('sentiment_analysis.csv')[['text', 'sentiment']]
            except:
                df_old = pd.DataFrame(columns=['text', 'sentiment'])

            df_new = pd.read_csv('sentiment_data.csv')

            label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df_new['Sentiment'] = df_new['Sentiment'].map(label_mapping)
            df_new = df_new.rename(columns={'Comment': 'text', 'Sentiment': 'sentiment'})[['text', 'sentiment']].dropna()

            counts = df_new['sentiment'].value_counts()
            min_count = counts.min()

            df_new_sampled = df_new.groupby('sentiment').sample(n=min_count, random_state=42).reset_index(drop=True)

            df_final = pd.concat([df_old, df_new_sampled], ignore_index=True)

            df_final['text_clean'] = df_final['text'].apply(clean_text)
            df_final = df_final[df_final['text_clean'].str.len() > 3]

            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100000)
            X = vectorizer.fit_transform(df_final['text_clean'])
            y = df_final['sentiment']

            model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
            model.fit(X, y)

            return model, vectorizer, len(df_final)
        
        except Exception as e:
            st.error(f"Training error: {e}")
            return None, None, 0
        
model, vectorizer, data_count = train_model()

if model:
    st.success(f"Training model successed with {data_count:,} rows.")
else:
    st.warning("Model failed.")
    st.stop()

st.divider()


user_input = st.text_area("Input a sentence in English:", height=100, placeholder="Example: The service was terrible and slow.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please input text.")
    else:
        cleaned_input = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vec_input)[0]
        st.subheader("Result:")

        if prediction == 'positive':
            st.success(f"POSITIVE")
        elif prediction == 'negative':
            st.error(f"NEGATIVE")
        else:
            st.info("NEUTRAL")