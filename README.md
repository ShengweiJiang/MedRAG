# MedRAG — Medical Q&A Assistant with RAG

A Retrieval-Augmented Generation (RAG) system for medical question answering, built end-to-end as a portfolio project to demonstrate RAG architecture and cloud deployment. Uses LanceDB for vector search, sentence-transformers for embeddings, and the Anthropic Claude API for generation. Containerized with Docker and deployed on AWS EC2 behind a custom HTTPS domain.

> ⚠️ **Disclaimer:** MedRAG is a portfolio project intended to demonstrate RAG architecture and cloud deployment. It is **not** a medical device and must not be used for actual diagnosis or treatment decisions.

---

## Highlights

- **End-to-end RAG pipeline** — ingestion, embedding, vector retrieval, and grounded generation in a single deployable application.
- **Grounded answers with source attribution** — every response is anchored to retrieved documents, which are surfaced in the UI for verification.
- **Reproducible deployment** — single `docker run` command works identically on a laptop and on AWS EC2.
- **Production hosting** — Dockerized, hosted on AWS EC2 with a stable Elastic IP and a custom HTTPS domain.

A running instance is available at [medrag.galaxydarklight.com](https://medrag.galaxydarklight.com).

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────┐
│  Streamlit UI       │  app.py
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌──────────────────────────────┐
│  Retriever          │ ───► │  LanceDB (vector store)      │
│  (sentence-         │      │  embeddings: all-MiniLM-L6-v2│
│   transformers)     │      │  dim: 384                    │
└──────────┬──────────┘      └──────────────────────────────┘
           │ top-k retrieved docs
           ▼
┌─────────────────────┐      ┌──────────────────────────────┐
│  Generator          │ ───► │  Anthropic Claude API        │
│  (system prompt +   │      │  (claude-haiku-4-5)          │
│   retrieved context)│      └──────────────────────────────┘
└──────────┬──────────┘
           │
           ▼
   Grounded answer + cited source documents
```

### Design decisions

| Component | Choice | Rationale |
|---|---|---|
| Vector store | **LanceDB** | Embedded, file-based, zero-ops. Lives in the same container as the app, so there is no external database to manage on EC2 and no extra network hop at query time. |
| Embedding model | **all-MiniLM-L6-v2** (384-dim, ~80 MB) | Fast on CPU, good enough baseline for short Q&A-style text. Pre-downloaded into the Docker image layer so containers cold-start in seconds instead of waiting on Hugging Face. |
| LLM | **Claude via Anthropic API** | Earlier iterations used a local Ollama model. Moving to the hosted API meant the system could run on a small CPU-only EC2 instance — no GPU, lower cost, simpler ops. |
| UI | **Streamlit** | Minimal code surface for a chat interface, trivial to containerize, no separate frontend build step. |
| Containerization | **Single-stage Docker image** | Embedding model baked into the image at build time so runtime has no external model-download dependency. Health check included for orchestrators. |

### What's *not* in scope (deliberately)

These are reasonable next additions, but were left out to keep the v1 surface small and the deployment story clean:

- A re-ranking stage (cross-encoder over the top-k).
- Streaming responses from Claude.
- A separate API service. The whole thing is one Streamlit container; splitting it into a FastAPI backend + frontend is straightforward but unnecessary at this scale.
- A managed vector DB (Pinecone, Weaviate). LanceDB's embedded file-based store is the right level of complexity for a single-node deployment.

---

## Tech Stack

- **Language:** Python 3.11
- **RAG / ML:** LanceDB, sentence-transformers, Anthropic SDK
- **App:** Streamlit
- **Infra:** Docker, AWS EC2, Elastic IP, custom HTTPS domain

---

## Project Structure

```
MedRAG/
├── app.py              # Streamlit UI and chat loop
├── ingestion.py        # Loads sample data, embeds, writes to LanceDB
├── retriever.py        # Vector search wrapper over LanceDB
├── generator.py        # Prompt assembly + Claude API call
├── run.py              # Convenience entrypoint
├── Dockerfile          # Slim Python 3.11 image, embedding model pre-cached
├── requirements.txt
├── data/
│   └── sample_medical_data.json   # Small sample corpus (see "Data" below)
└── lancedb/                       # Generated at runtime (vector store)
```

---

## Data

`data/sample_medical_data.json` is a small set of generic medical Q&A pairs included solely to demonstrate the pipeline end-to-end. It is **not** sourced from a curated medical corpus, and is not intended to provide medically authoritative answers. See the Roadmap below for the plan to replace it.

---

## Running Locally

### Prerequisites

- Python 3.11+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))

### Option 1: Docker (recommended)

```bash
# Build
docker build -t medrag .

# Run
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  medrag
```

Open <http://localhost:8501> and click **"Initialize knowledge base"** in the sidebar on the first run.

### Option 2: Python directly

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."   # or put it in a .env file

streamlit run app.py
```

---

## Deployment (AWS EC2)

The live instance runs on AWS with the following setup:

1. **EC2 instance** — small CPU-only instance. No GPU needed because LLM inference is delegated to the Anthropic API and embeddings (MiniLM, 384-dim) run cheaply on CPU.
2. **Docker** — image built locally and deployed to the instance, run as a long-lived container exposing port 8501. The embedding model is included in the image, so the container starts without any external model download.
3. **Elastic IP** — attached to the instance so the public IP survives stop/start cycles and reboots.
4. **Domain + HTTPS (Cloudflare Full strict)** — DNS for `medrag.galaxydarklight.com` is managed by Cloudflare, which terminates TLS at the edge with its own certificate. Traffic from Cloudflare to the EC2 instance is re-encrypted over a second TLS connection terminated by Nginx on the EC2 host, using a Let's Encrypt certificate provisioned and auto-renewed via Certbot. Nginx then reverse-proxies plaintext to the Streamlit container on `127.0.0.1:8501`. This is the Cloudflare "Full (strict)" topology — both legs of the connection are encrypted and certificates are validated end to end.
5. **Secrets** — `ANTHROPIC_API_KEY` is passed as an environment variable at container runtime and never baked into the image or committed to the repo.
6. **Health checks** — the Dockerfile defines a healthcheck against Streamlit's `/_stcore/health` endpoint, suitable for use with container orchestrators.

---

## How It Works

**1. Ingestion** (`ingestion.py`)
Each Q&A pair in `sample_medical_data.json` is concatenated into a single text block (`"Question: ...\n\nAnswer: ..."`), embedded with `all-MiniLM-L6-v2`, and written to a LanceDB table called `medical_docs`. Re-running ingestion drops and recreates the table, so it's safe to re-run on a corpus change.

**2. Retrieval** (`retriever.py`)
At query time, the user's question is embedded with the same model. LanceDB returns the top-k nearest neighbors by vector distance. `top_k` is exposed as a slider in the UI (default 3).

**3. Generation** (`generator.py`)
The retrieved documents are formatted into a context block and passed to Claude with a system prompt that instructs the model to:
- Answer only from the provided context
- Say "I don't know" when the context is insufficient
- Cite which document(s) supported the answer
- Include a medical disclaimer

**4. UI** (`app.py`)
A Streamlit chat interface shows the answer alongside an expandable panel with the cited source documents, plus a sidebar with API/knowledge-base status and example questions.

---

## Roadmap

- **Swap in a trusted external dataset.** Replace the placeholder sample data with a curated corpus (e.g., MedQuAD or PubMedQA). Loader changes are isolated to `ingestion.py`.
- **Add a re-ranker.** A cross-encoder pass over the top-k retrieval candidates would meaningfully improve precision before the LLM call.
- **Streaming responses.** Switch the Anthropic call to streaming and render tokens incrementally in Streamlit.
- **Evaluation harness.** Add retrieval-quality metrics (recall@k against a labeled set) and a generation-quality eval (LLM-as-judge or human spot-check) so future changes are measurable.
- **Split UI from API.** Extract a FastAPI service exposing a `/query` endpoint, with the Streamlit app (or a static frontend) as one client among others.

---

## License

MIT

---

## Author

**Shengwei Jiang** — [GitHub](https://github.com/ShengweiJiang)
