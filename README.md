# Production-Grade Evaluation-Driven RAG Application

> **Evaluation-driven RAG architecture with two-stage cross-encoder reranking, hallucination mitigation, and automated CI/CD metric gating.**

---

## 🌟 Overview

This project implements a high-precision, production-grade Retrieval-Augmented Generation (RAG) system built to solve hallucination and relevance degradation common in naive RAG pipelines. 

Designed around **lecture transcripts and course materials**, the system features:
- **Two-Stage Retrieval**: Vector similarity search (bi-encoder) combined with cross-encoder re-ranking.
- **Strict Faithfulness Grounding**: Grounded generation via Gemini 2.0 Flash with automatic abstention when context is insufficient.
- **Interactive UI**: Real-time Streamlit chat interface with tunable retrieval parameters and full context inspection.
- **Eval-Driven Architecture**: Exposes the full RAG triad (`query`, `context`, `answer`) for continuous evaluation of Faithfulness, Answer Relevance, and Context Precision.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Over-Retrieval (Bi-Encoder)        │
│ ChromaDB + all-MiniLM-L6-v2 (fetch_k = 10)   │
└──────────────────────┬───────────────────────┘
                       │ Top 10 Chunks
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Reranking (Cross-Encoder)          │
│ ms-marco-MiniLM-L-6-v2 (top_k = 5)           │
└──────────────────────┬───────────────────────┘
                       │ Top 5 Reranked Chunks
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Grounded Generation (Gemini Flash)  │
│ Strict Faithfulness Prompt & Abstention      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              Grounded Response / Abstention
```

---

## 📁 Project Structure

```
.
├── data/
│   └── Session_1.vtt           # Course transcript source data
├── src/
│   ├── __init__.py
│   ├── retriever.py            # Transcript parser, chunker & ChromaDB vector store
│   ├── reranker.py             # Cross-encoder reranking retriever
│   ├── generator.py            # Gemini 2.0 Flash generation & streaming
│   ├── rag_pipeline.py         # End-to-end RAG orchestrator exposing RAG triad
│   └── app.py                  # Streamlit chat interface
├── export_chroma_chunks.py     # Utility to dump stored chunks for evaluation
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Free Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sisodevaibhav11/Production-grade-Eval-Driven-RAG-application.git
cd Production-grade-Eval-Driven-RAG-application

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

Create your `.env` file from the template:

```bash
cp .env.example .env
```

Add your Gemini API key to `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GENERATOR_MODEL=gemini-2.0-flash
JUDGE_MODEL=gemini-2.0-flash
```

### 4. Run the Application

Launch the Streamlit web interface:

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚙️ Key Technical Features

### Two-Stage Retrieval
1. **Bi-Encoder (`all-MiniLM-L6-v2`)**: Fast, scalable cosine similarity search fetches candidate passages from ChromaDB.
2. **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**: Jointly scores `(query, document)` pairs, capturing full cross-attention semantics to select the top-k highest quality chunks.

### Hallucination Mitigation & Abstention
The prompt enforces strict context boundaries. If the retrieved context does not contain sufficient facts to answer the question, the model responds with:
> *"I don't have enough information in the course material to answer that."*

---

## 📄 License
MIT License.
