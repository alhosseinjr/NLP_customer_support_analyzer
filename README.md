# Customer Complaint Analyzer

A Streamlit deployment of the NLP customer-support notebook. Given a
customer complaint, the app predicts:

- **Intent** (TF-IDF + Logistic Regression)
- **Sentiment** (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Topic cluster** (sentence embeddings + KMeans)
- **A suggested reply** (small instruction-tuned LLM, optional toggle)

## Project structure

```
.
├── app.py                 # Streamlit app
├── train_artifacts.py     # Rebuilds the sklearn artifacts from the dataset
├── requirements.txt
├── models/                # tfidf_vectorizer.pkl, complaint_classifier.pkl,
│                           # kmeans_model.pkl, cluster_names.json go here
└── .streamlit/config.toml # Theming
```

## 1. Get the model artifacts

The app needs three small files in `models/`:

- `tfidf_vectorizer.pkl`
- `complaint_classifier.pkl`
- `kmeans_model.pkl`
- (optional) `cluster_names.json` — friendly names for each cluster id

You have two options:

**Option A — reuse what you already trained in the notebook.**
Your notebook already saves these with `joblib.dump(...)`. Just download
`complaint_classifier.pkl`, `tfidf_vectorizer.pkl`, and `kmeans_model.pkl`
from your Kaggle/Colab session and drop them into `models/`. Add a
`models/cluster_names.json` like:

```json
{
  "0": "Payment & Refund Issues",
  "1": "Account Management",
  "2": "Customer Service & Subscription",
  "3": "Order & Product Issues",
  "4": "Shipping & Address Issues"
}
```

**Option B — regenerate them from scratch.**

```bash
pip install -r requirements.txt
python train_artifacts.py
```

This re-downloads the dataset from Hugging Face, re-runs the cleaning /
TF-IDF / Logistic Regression / KMeans steps from the notebook, and writes
the artifacts into `models/`.

The sentiment model and the reply-generation LLM are downloaded
automatically by `transformers` the first time the app runs — no manual
step needed for those.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Deploy

### Streamlit Community Cloud (easiest)
1. Push this folder to a GitHub repo (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file to `app.py`.
3. **Heads up:** the default reply model (`Qwen/Qwen2.5-1.5B-Instruct`) is
   a few GB and may be too heavy for the free tier. You can:
   - uncheck "Generate an automated reply" in the app sidebar, or
   - set the `GEN_MODEL_NAME` environment variable (in the app's
     "Advanced settings") to a smaller model, e.g.
     `Qwen/Qwen2.5-0.5B-Instruct` or `distilgpt2`.

### Any other host (Docker, Render, Fly.io, a VM, etc.)
A GPU-backed host is recommended if you want to keep the 1.5B reply model.
Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t complaint-analyzer .
docker run -p 8501:8501 complaint-analyzer
```

## 4. Push to GitHub

```bash
cd nlp-customer-support-analyzer
git init
git add .
git commit -m "Initial commit: complaint analyzer Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you're committing the `.pkl` files (Option A above), remove or edit the
`models/*.pkl` line in `.gitignore` first — and consider Git LFS if they're
large.

## Notes

- `GEN_MODEL_NAME` env var lets you swap the reply-generation model without
  touching code.
- All heavy models are loaded with `st.cache_resource`, so they're only
  loaded once per app session, not on every click.
- If `models/*.pkl` are missing, the app still runs — it just skips intent
  classification / clustering and tells you to run `train_artifacts.py`.
