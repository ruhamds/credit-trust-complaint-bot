````md
# 🛡️ CreditTrust Financial — Complaint Intelligence Platform

**AI-Powered Complaint Intelligence System**

An end-to-end **Retrieval-Augmented Generation (RAG)** platform that transforms **9.6 million real-world consumer complaints** into clear, actionable insights for Product, Support, and Compliance teams.

Built with **ChromaDB • Groq (Llama 3.1) • Streamlit**

The system processes over 9.6 million real-world consumer complaints and enables grounded question-answering across multiple financial product categories using ChromaDB, Sentence Transformers, and Llama 3.1.

---

## Key Features

- Semantic retrieval over 100k+ complaint chunks
- Tokenizer-aware adaptive chunking pipeline
- Metadata-filtered vector search for category-specific retrieval
- ChromaDB persistent vector store
- Groq-powered low-latency RAG responses
- Streamlit conversational interface
- Unit-tested retrieval and generation pipeline
- Evaluation framework for retrieval similarity and grounding quality

---

## Financial Products Covered

- Credit card
- Personal loan
- Buy Now Pay Later (BNPL)
- Savings account
- Money transfers

---

## System Architecture

```text
User Question
      │
      ▼
 ┌─────────────┐
 │  Streamlit  │
 │  Interface  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Retriever  │
 │             │
 │ Embed query │
 │ Query Chroma│
 │ Top-K chunks│
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Generator  │
 │             │
 │ Prompt +    │
 │ Retrieved   │
 │ Context     │
 └──────┬──────┘
        │
        ▼
 Grounded Financial Answer
 + Supporting Complaint Evidence
````

---

## System Components

| Layer            | Description                                         |
| ---------------- | --------------------------------------------------- |
| Data Processing  | Cleaning and preprocessing 9.6M CFPB complaints     |
| Vector Pipeline  | Adaptive token-based chunking and embeddings        |
| Retrieval Engine | ChromaDB semantic retrieval with metadata filtering |
| Generation Layer | Groq-hosted Llama 3.1 grounded response generation  |
| Frontend         | Streamlit conversational interface                  |
| Testing          | Unit-tested retrieval and RAG evaluation pipeline   |

---

## Engineering Challenges & Solutions

| Challenge                                       | Solution                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| 9.6M-row dataset exceeded memory limits         | Chunked streaming pipeline using `pandas.read_csv(chunksize=...)` |
| Word-count chunking caused embedding truncation | Switched to tokenizer-aware adaptive chunking                     |
| Heavy class imbalance across product categories | Stratified sampling by financial product                          |
| Semantic overlap between fraud complaints       | Metadata-aware retrieval filtering                                |
| CPU-only embedding pipeline too slow            | Deduplication + adaptive chunking reduced total embeddings        |

---

## Dataset Statistics

| Metric                     | Value                |
| -------------------------- | -------------------- |
| Raw dataset size           | 9,609,797 complaints |
| After filtering & cleaning | 347,505 complaints   |
| Stratified sample          | 50,000 complaints    |
| After deduplication        | 45,292 complaints    |
| Vector store chunks        | 100,483              |

### Data Source

CFPB Consumer Complaint Database
[https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)

---

## Vector Pipeline Design

### Embedding Model

* `all-MiniLM-L6-v2`
* 384-dimensional embeddings
* Cosine similarity retrieval

### Chunking Strategy

The pipeline uses tokenizer-aware adaptive chunking instead of word-count splitting.

```python
if token_count <= 200:
    store_as_single_chunk()
else:
    split_with_overlap()
```

### Why Adaptive Chunking?

* 80% of complaints fit within a single chunk
* Splitting short narratives fragments coherent context
* Actual tokenizer limits prevent silent embedding truncation

---

## Retrieval & Generation Pipeline

### Retrieval

* ChromaDB persistent vector store
* Top-K semantic similarity search (`K=5`)
* Metadata filtering by financial product category

### Generation

* Groq API
* `llama-3.1-8b-instant`
* Temperature = `0.2` for deterministic, grounded responses

### Why Groq Instead of OpenAI?

Groq-hosted Llama 3.1 was selected because of:

* significantly lower latency
* free experimentation during development
* sufficient reasoning quality for grounded RAG workflows
* lower operational cost for retrieval-heavy pipelines

---

## Evaluation Results

| Metric                   | Result |
| ------------------------ | ------ |
| Avg retrieval similarity | 0.6211 |
| Avg answer quality       | 94.5%  |
| Avg query latency        | 7.62s  |
| Total tokens consumed    | 12,800 |
| Retrieval top-K          | 5      |

### Evaluation Methodology

The system was evaluated using 10 representative financial queries across all product categories.

Answer quality was manually evaluated using:

* Grounding quality
* Response relevance
* Response structure

Each metric used a 1–3 scoring scale.

---

## Example Questions

* “What are the most common credit card fraud complaints?”
* “What problems do consumers face with international wire transfers?”
* “How are BNPL disputes typically resolved?”
* “What complaints are common with high-interest personal loans?”

---

## Project Structure

```text
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example        
├── .gitignore
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_vector_pipeline.ipynb
│   └── task3_rag_pipeline.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
├── plots/
├── tests/
│   └── test_rag_pipeline.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ruhamds/credit-trust-complaint-bot.git
cd ruhamds/credit-trust-complaint-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Get a free API key from:
[https://console.groq.com](https://console.groq.com)

---

## Running the Pipeline

### Step 1 — Data Processing

```bash
jupyter notebook notebooks/task1_eda_preprocessing.ipynb
```

### Step 2 — Build Vector Store

```bash
jupyter notebook notebooks/task2_vector_pipeline.ipynb
```

### Step 3 — Run RAG Evaluation

```bash
jupyter notebook notebooks/task3_rag_pipeline.ipynb
```

### Step 4 — Launch Streamlit App

```bash
streamlit run app.py
```

---

## Testing

The project includes unit tests covering:

* Retrieval correctness
* Similarity score validation
* Metadata filtering
* Response schema validation
* Error propagation

Run tests with:

```bash
pytest tests/ -v
```

Example result:

```text
15 passed in 2.31s
```

---

## Known Limitations

1. BNPL complaint coverage is limited because CFPB only standardised BNPL reporting recently.
2. Vector store uses a stratified sample instead of the full corpus for practical CPU-based embedding.
3. Manual evaluation was used because no public financial RAG benchmark exists for this dataset.
4. Fraud-related complaints across products still exhibit semantic overlap despite metadata filtering.

---

## Future Improvements

* Hybrid retrieval (BM25 + dense embeddings)
* Cross-encoder reranking
* Near-duplicate detection using MinHash LSH
* Automated RAG evaluation framework
* Multi-query retrieval augmentation
* GPU-based embedding acceleration

---

## License

Developed as part of the 10 Academy KAIM challenge and extended into a full end-to-end financial RAG engineering project.

```
```
