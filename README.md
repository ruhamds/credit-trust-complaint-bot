# CrediTrust Financial — Complaint Intelligence Platform

A Retrieval-Augmented Generation (RAG) system that analyses CFPB consumer complaint data to provide actionable insights across five financial product categories.

![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)

---

## Overview

| Component | Description |
|---|---|
| **Task 1** | EDA and preprocessing of 9.6M CFPB complaints → 45k clean records |
| **Task 2** | Token-based chunking + ChromaDB vector store (100k chunks) |
| **Task 3** | RAG pipeline with Groq LLM + qualitative evaluation (94.4% quality) |
| **Task 4** | Streamlit chat interface for non-technical users |

**Product categories covered:**
- Credit card
- Personal loan
- Buy Now Pay Later
- Savings account
- Money transfers

---

## Architecture

```
User Question
      │
      ▼
 ┌─────────────┐
 │  Streamlit  │  app.py
 │  Interface  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Retrieve  │  embed question → query ChromaDB → top-5 chunks
 │             │  model: all-MiniLM-L6-v2 (384-dim, cosine similarity)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Generate  │  system prompt + retrieved context → Groq LLM
 │             │  model: llama-3.1-8b-instant
 └──────┬──────┘
        │
        ▼
  Structured Answer
  + Cited Sources
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 4. Download the dataset
Download the CFPB complaint dataset from:
[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

Place the CSV at:
```
data/raw/complaints.csv
```

### 5. Run the notebooks in order
```
notebooks/task1_eda_preprocessing.ipynb        # EDA + data cleaning
notebooks/task2_chunking_embedding_vectorstore_v2.ipynb  # Build vector store
notebooks/task3_rag_pipeline.ipynb             # RAG pipeline + evaluation
```

### 6. Launch the app
```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                          # Streamlit application (Task 4)
├── rag_pipeline.py                 # RAG pipeline module (Task 3)
├── requirements.txt
├── .env                            # API keys — NOT committed to Git
├── .gitignore
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding_vectorstore_v2.ipynb
│   └── task3_rag_pipeline.ipynb
├── data/
│   ├── raw/                        # Raw CFPB CSV — NOT committed (too large)
│   ├── processed/                  # Filtered dataset — NOT committed
│   └── vectorstore/                # ChromaDB — NOT committed
├── plots/                          # Generated visualisations
├── tests/
│   └── test_rag_pipeline.py        # Unit tests
└── .github/
    └── workflows/
        └── ci.yml                  # GitHub Actions CI
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocks — no API key or vector store required to run them.

```
tests/test_rag_pipeline.py::TestRetrieve::test_returns_correct_chunk_count      PASSED
tests/test_rag_pipeline.py::TestRetrieve::test_invalid_category_raises_value_error PASSED
tests/test_rag_pipeline.py::TestRetrieve::test_all_valid_categories_accepted    PASSED
tests/test_rag_pipeline.py::TestRetrieve::test_chunk_has_required_keys          PASSED
tests/test_rag_pipeline.py::TestRetrieve::test_similarity_is_in_valid_range     PASSED
tests/test_rag_pipeline.py::TestRetrieve::test_similarity_sorted_descending     PASSED
tests/test_rag_pipeline.py::TestGenerate::test_returns_required_keys            PASSED
tests/test_rag_pipeline.py::TestGenerate::test_answer_is_non_empty_string       PASSED
tests/test_rag_pipeline.py::TestGenerate::test_token_counts_are_positive_integers PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_returns_all_required_keys     PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_avg_similarity_is_mean_of_chunks PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_min_similarity_is_min_of_chunks  PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_latency_is_positive_float     PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_question_and_category_passed_through PASSED
tests/test_rag_pipeline.py::TestRagPipeline::test_invalid_category_propagates_error   PASSED
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Token-based chunking | Word count is inaccurate for financial text — actual tokenizer prevents silent truncation |
| Adaptive chunking | 80% of complaints fit in one chunk — splitting short texts fragments coherent context |
| Stratified sampling (50k) | Balances 15:1 class imbalance (Credit card vs Savings account) |
| ChromaDB with metadata filter | Category-scoped retrieval prevents cross-product contamination |
| Temperature 0.2 | Factual, deterministic answers appropriate for compliance use case |
| Mocked tests | CI runs without API keys or vector store — fast, free, always green |

---

## Known Limitations

1. **BNPL dataset gap** — CFPB only standardised BNPL reporting in 2023. BNPL complaints identified via keyword matching may include false positives.
2. **Stratified sample** — vector store uses 14% of the full corpus. Full dataset preserved on disk.
3. **Manual evaluation** — no ground-truth QA labels available for automated scoring.
4. **Q06 retrieval weakness** — BNPL dispute resolution queries return low similarity (0.496) due to sparse direct BNPL dispute data in CFPB corpus.

---

## Dataset

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

| Metric | Value |
|---|---|
| Raw dataset size | 9,609,797 complaints |
| After filtering + cleaning | 347,505 complaints |
| After stratified sampling | 45,292 complaints |
| After deduplication | 45,292 (4,708 duplicates removed) |
| Vector store chunks | 100,483 |

---

## License

This project was developed as part of the 10 Academy KAIM Week 6 challenge.