# Trademarkia AI/ML Engineer Task

## Lightweight Semantic Search with Fuzzy Clustering and Semantic Cache

## Project Overview

This project implements a **lightweight semantic search system** over the **20 Newsgroups dataset (~20,000 documents)**.

The system allows users to submit **natural language queries**, and it retrieves semantically related documents using **vector embeddings and a vector database**.

To improve efficiency and avoid recomputation, a **semantic cache layer** is implemented that detects **similar queries even when phrased differently**.

The system exposes this functionality through a **FastAPI service**.

The project satisfies all four components required in the assignment:

```
1. Embedding & Vector Database Setup
2. Fuzzy Clustering
3. Semantic Cache
4. FastAPI API Service
```

The system architecture is designed so that:

```
User Query
    ↓
Embedding Model
    ↓
Cluster Prediction
    ↓
Semantic Cache Lookup
    ↓
Cache Hit → Return Cached Result
    ↓
Cache Miss → Vector Search (FAISS)
    ↓
Store Result in Cache
    ↓
Return Response
```

---

# Dataset

The dataset used is:

```
20 Newsgroups Dataset
~20,000 documents
20 topic categories
```

Each document is a **Usenet post containing headers and message body**. These documents contain **noise and metadata**, so preprocessing is required before semantic analysis.

---

# Data Preprocessing

Notebook used: `01_data_preparation.ipynb`

Steps performed:
1. Load dataset (19997 documents)
2. Removed components (Email headers, Quoted replies, Metadata lines, Extra whitespace)
3. Lowercase normalization and basic punctuation cleanup.

Cleaned dataset saved as: `data/processed/newsgroups_cleaned.csv`

---

# Embedding Generation

Notebook used: `02_embeddings_vector_db.ipynb`

Goal: Convert text documents into **dense vector representations**.

Model used: `sentence-transformers/all-MiniLM-L6-v2`
*   Lightweight and fast inference
*   384-dimensional embeddings
*   Good semantic performance without requiring heavy GPU resources.

Embeddings generated for all documents and saved as: `data/embeddings/document_embeddings.npy`

---

# Vector Database

To enable efficient similarity search, embeddings are stored in a **FAISS vector index**.
FAISS configuration is set to `IndexFlatIP` combined with `faiss.normalize_L2(embeddings)` because calculating the inner product of L2-normalized vectors calculates mathematically exact cosine similarity. This setup scales extremely effectively without needing external bulky infrastructures.

Vector index saved as: `data/embeddings/faiss_index.bin`

---

# Fuzzy Clustering

Notebook used: `03_fuzzy_clustering.ipynb`

Purpose: Discover the **latent semantic structure of the corpus**.

Important requirement from assignment:
*   Hard clusters are NOT acceptable.
*   Documents must belong to multiple clusters.

Therefore we used: **Gaussian Mixture Model (GMM)** Instead of KMeans.
GMM handles boundary cases and evaluates soft probability groupings out of overlapping semantic themes spanning multiple labeled categories. 

### Cluster Selection
The number of clusters is justified with evidence using the **BIC (Bayesian Information Criterion)**. By exploring ranges (5 to 30), a steep initial drop normalized closely forming an elbow near 20 clusters—striking a balance between model complexity and semantic clarity over 20 Newsgroups documents.

---

# Semantic Cache

Part 3 of the assignment requires building a **semantic cache without Redis**.
The goal is to avoid recomputing results for **similar queries phrased differently**.

Example:
*   Query A: "How does the space shuttle launch?"
*   Query B: "Explain shuttle liftoff mechanism"

A traditional string-matching cache naturally misses this. The semantic cache resolves this utilizing embeddings, vector cosine-distance matching, and topic-aware segmentations. 

### Cluster-Aware Caching
The problem with a continuously expanding flat cache is lookup latency. Scanning through 1,000,000 cached arrays takes heavy computation. Therefore, the cache structure is nested using **predicted cluster IDs**. When users ask a space question, the system solely computes distance vectors across existing cache questions strictly registered inside the predicted "space" cluster limit, skipping entirely the thousands of irrelevant cached inquiries inside domains like "politics" or "religion."

---

### Tunable Parameter Analysis: The Similarity Threshold

The beating heart of our semantic caching accuracy stems from a singular highly tunable decision threshold. Here is what adjusting this parameter determines about the system's behavior: 

*   `similarity_threshold = 0.70` -> **High hit rate, High False-Positive Risk:** Because variations tolerate broad topics, unrelated queries sharing similar vocabulary overlap easily. Two separate queries ("is windows 95 good" and "windows crashing issue") might mistakenly share identical cache responses.
*   `similarity_threshold = 0.85` -> **Balanced Precision and Recall (Our Choice):** An 0.85 threshold yields aggressive tolerance for identical semantic queries ("How does the space shuttle launch?" VS "Explain shuttle liftoff mechanism") while firmly blocking differently intended queries. The recall isn't hyper-punitive but doesn't hallucinate results.
*   `similarity_threshold = 0.95` -> **Low hit rate, High Precision:** Acts almost like an exact string matching cache. Semantic value degrades immensely as normal synonymous queries start counting as continuous cache misses. Perfect if hallucination carries immense penalty.

---

# FastAPI Service

The system is exposed via a **FastAPI API** handling proper state management across endpoints.
Server is started cleanly with a single command mapping on port 8000: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

Swagger Interface available at: `http://127.0.0.1:8000/docs`

# API Endpoints

1.  **POST /query**
    Accepts JSON body: `{ "query": "<natural language query>" }`
    Embeds the user query, bounds it by a generated topic cluster, assesses if a corresponding historical query overlaps structurally past the similarity curve—and if so, bypasses the massive datastore completely saving extreme computation latency.

2.  **GET /cache/stats**
    Returns internal running stats over total tracked variables detailing exactly how the semantic caching mechanism dynamically alleviates server pressure over time.

3.  **DELETE /cache**
    Nullifies memory dict payload safely and resets tracker counts.

---

# Deployment (Docker)

To prove production readiness, this application is optionally fully Dockerized containing clean images using the lightweight python-3.12-slim base image. Data pipelines, FAISS indices, datasets are properly formatted leveraging volume mapping without recalculation costs over server restart states.
