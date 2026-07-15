"""
train_artifacts.py
===================
Regenerates the artifacts the Streamlit app needs:
    models/tfidf_vectorizer.pkl
    models/complaint_classifier.pkl
    models/kmeans_model.pkl
    models/cluster_names.json

This mirrors the preprocessing / modeling steps from the original notebook
(Parts 1, 2, and 4). Run it once locally before deploying, or whenever you
want to retrain on fresh data:

    python train_artifacts.py

Requires internet access (downloads the dataset + sentence-transformer
model from Hugging Face) and the packages in requirements.txt.
"""

import json
import os
import re
import string

import joblib
import nltk
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CLUSTERS = 5
RANDOM_STATE = 42


def download_nltk_data():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)


def build_cleaner():
    import contractions
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean(text: str) -> str:
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

    return clean


def main():
    print("Downloading NLTK data...")
    download_nltk_data()

    print("Loading dataset (bitext/Bitext-customer-support-llm-chatbot-training-dataset)...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = dataset["train"].to_pandas()
    df = df[["instruction", "category", "intent", "response"]]
    df.rename(columns={"instruction": "text", "response": "answer"}, inplace=True)

    print("Cleaning text...")
    clean = build_cleaner()
    df["clean_text"] = df["text"].apply(clean)

    # ---------------- Intent classifier ----------------
    print("Training TF-IDF + Logistic Regression intent classifier...")
    X = df["clean_text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    classifier.fit(X_train_tfidf, y_train)

    accuracy = classifier.score(X_test_tfidf, y_test)
    print(f"Intent classifier test accuracy: {accuracy:.3f}")

    joblib.dump(classifier, os.path.join(MODELS_DIR, "complaint_classifier.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    # ---------------- Clustering ----------------
    print("Encoding embeddings for clustering (this can take a few minutes)...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["clean_text"].tolist(), show_progress_bar=True)

    print(f"Running KMeans with k={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))

    # Derive human-readable cluster names from top keywords per cluster.
    print("Naming clusters from top keywords...")
    cluster_names = {}
    for cluster_id in sorted(df["cluster"].unique()):
        texts = df.loc[df["cluster"] == cluster_id, "clean_text"]
        vectorizer = CountVectorizer(stop_words="english", max_features=5)
        vectorizer.fit(texts)
        keywords = ", ".join(vectorizer.get_feature_names_out())
        cluster_names[str(int(cluster_id))] = keywords

    with open(os.path.join(MODELS_DIR, "cluster_names.json"), "w") as f:
        json.dump(cluster_names, f, indent=2)

    print("\nDone. Artifacts written to:", MODELS_DIR)
    print(
        "\nNote: cluster_names.json currently holds each cluster's top "
        "keywords. Feel free to hand-edit it with friendlier labels, e.g.:\n"
        '  {"0": "Payment & Refund Issues", "1": "Account Management", ...}'
    )


if __name__ == "__main__":
    main()
