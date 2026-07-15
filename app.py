"""
Customer Complaint Analyzer — Streamlit deployment
====================================================
Reproduces the pipeline from the original NLP notebook:
  1. Text preprocessing (clean, tokenize, lemmatize)
  2. Intent classification (TF-IDF + Logistic Regression)
  3. Sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)
  4. Complaint clustering (SentenceTransformer embeddings + KMeans)
  5. Automated reply generation (a small instruction-tuned LLM)

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to GitHub and deploy on Streamlit Community Cloud,
    or containerize it (see README.md / Dockerfile).
"""

import os
import re
import string
import json

import joblib
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Complaint Analyzer",
    page_icon="💬",
    layout="centered",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Text-generation model used for the auto-reply. The original notebook used
# Qwen/Qwen2.5-1.5B-Instruct. That works great locally / on a GPU box, but is
# heavy for a small free-tier web host, so it's configurable via env var.
GEN_MODEL_NAME = os.environ.get("GEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CLUSTER_NAMES_PATH = os.path.join(MODELS_DIR, "cluster_names.json")
DEFAULT_CLUSTER_NAMES = {
    "0": "Payment & Refund Issues",
    "1": "Account Management",
    "2": "Customer Service & Subscription",
    "3": "Order & Product Issues",
    "4": "Shipping & Address Issues",
}


# ----------------------------------------------------------------------
# Text preprocessing (matches the notebook's cleaning steps)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nltk():
    import nltk

    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text: str, stop_words, lemmatizer) -> str:
    import contractions
    from nltk.tokenize import word_tokenize

    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2]

    return " ".join(tokens)


# ----------------------------------------------------------------------
# Cached model loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading intent classifier...")
def load_intent_model():
    clf_path = os.path.join(MODELS_DIR, "complaint_classifier.pkl")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not (os.path.exists(clf_path) and os.path.exists(vec_path)):
        return None, None
    classifier = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)
    return classifier, vectorizer


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_pipeline():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)


@st.cache_resource(show_spinner="Loading clustering model...")
def load_cluster_model():
    from sentence_transformers import SentenceTransformer

    kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    kmeans = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedder, kmeans


@st.cache_resource(show_spinner="Loading reply-generation model (first run can take a while)...")
def load_generator():
    import torch
    from transformers import pipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        "text-generation",
        model=GEN_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )


def load_cluster_names():
    if os.path.exists(CLUSTER_NAMES_PATH):
        with open(CLUSTER_NAMES_PATH) as f:
            return json.load(f)
    return DEFAULT_CLUSTER_NAMES


def generate_reply(generator, complaint, intent, sentiment):
    prompt = f"""
You are a professional customer support agent.

Customer Complaint:
{complaint}

Category:
{intent}

Sentiment:
{sentiment}

Response:
"""
    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("💬 Customer Complaint Analyzer")
st.caption(
    "Intent classification · sentiment analysis · complaint clustering · "
    "automated reply generation"
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app runs the full complaint-analysis pipeline from the "
        "companion notebook: TF-IDF + Logistic Regression for intent, "
        "a RoBERTa sentiment model, sentence-embedding + KMeans clustering, "
        "and a small LLM for a draft reply."
    )
    st.divider()
    generate_llm_reply = st.checkbox("Generate an automated reply (LLM)", value=True)
    st.caption(
        f"Reply model: `{GEN_MODEL_NAME}`. This is the heaviest step — "
        "uncheck it for a faster, lighter-weight run."
    )
    st.divider()
    st.caption(
        "Missing `models/*.pkl`? Run `train_artifacts.py` first, or drop your "
        "own `tfidf_vectorizer.pkl`, `complaint_classifier.pkl`, and "
        "`kmeans_model.pkl` into the `models/` folder."
    )

classifier, vectorizer = load_intent_model()
cluster_names = load_cluster_names()

complaint = st.text_area(
    "Customer complaint",
    placeholder="e.g. I paid for my order but I still didn't receive it.",
    height=120,
)

analyze = st.button("Analyze complaint", type="primary", use_container_width=True)

if analyze:
    if not complaint.strip():
        st.warning("Please enter a complaint to analyze.")
        st.stop()

    stop_words, lemmatizer = load_nltk()
    cleaned = clean_text(complaint, stop_words, lemmatizer)

    col1, col2 = st.columns(2)

    # --- Intent ---
    with col1:
        st.subheader("Predicted intent")
        if classifier is not None and vectorizer is not None:
            vec = vectorizer.transform([cleaned])
            intent = classifier.predict(vec)[0]
            st.success(intent)
        else:
            intent = "unknown"
            st.error(
                "No intent classifier found in `models/`. "
                "Run `train_artifacts.py` to generate one."
            )

    # --- Sentiment ---
    with col2:
        st.subheader("Sentiment")
        sentiment_pipe = load_sentiment_pipeline()
        sent_result = sentiment_pipe(complaint)[0]
        sentiment_label = sent_result["label"]
        st.info(f"{sentiment_label}  ({sent_result['score']:.2f} confidence)")

    # --- Cluster ---
    st.subheader("Topic cluster")
    embedder, kmeans = load_cluster_model()
    if kmeans is not None:
        emb = embedder.encode([cleaned])
        cluster_id = int(kmeans.predict(emb)[0])
        cluster_label = cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")
        st.write(f"**{cluster_label}** (cluster {cluster_id})")
    else:
        st.warning(
            "No clustering model found in `models/`. "
            "Run `train_artifacts.py` to generate one."
        )

    # --- Reply generation ---
    if generate_llm_reply:
        st.subheader("Suggested reply")
        with st.spinner("Generating reply..."):
            generator = load_generator()
            reply = generate_reply(generator, complaint, intent, sentiment_label)
        st.write(reply)

st.divider()
st.caption(
    "Built from the original NLP customer-support notebook. "
    "See README.md for training / deployment instructions."
)
