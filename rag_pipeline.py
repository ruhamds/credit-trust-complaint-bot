"""
rag_pipeline.py
---------------
CrediTrust Financial — Intelligent Complaint Analysis
RAG (Retrieval-Augmented Generation) Pipeline

Exports:
    retrieve()      — embed question, query ChromaDB, return top-k chunks
    generate()      — build prompt, call Groq LLM, return answer
    rag_pipeline()  — end-to-end: question → retrieve → generate → answer

Usage:
    from rag_pipeline import rag_pipeline

    result = rag_pipeline(
        question         = "What are common credit card fraud complaints?",
        product_category = "Credit card"
    )
    print(result["answer"])

Environment:
    Requires GROQ_API_KEY in a .env file at the project root.
    pip install groq chromadb sentence-transformers python-dotenv
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import time
import logging

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Valid product categories ──────────────────────────────────────────────────
VALID_CATEGORIES = {
    'Credit card',
    'Personal loan',
    'Buy Now Pay Later',
    'Savings account',
    'Money transfers',
}

# ── Default config (overridable at call time) ─────────────────────────────────
DEFAULT_TOP_K           = 5
DEFAULT_TEMPERATURE     = 0.2
DEFAULT_MAX_TOKENS      = 1024
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_GROQ_MODEL      = 'llama-3.1-8b-instant'
DEFAULT_COLLECTION_NAME = 'cfpb_complaints'

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst at CrediTrust Financial.
Your role is to analyse consumer complaint data and provide clear, actionable insights
to internal teams.

You will be given:
1. A question about consumer complaints in a specific product category
2. A set of relevant complaint excerpts retrieved from the CFPB complaint database

Your response MUST:
- Be grounded ONLY in the provided complaint excerpts — do not use outside knowledge
- Follow the exact structure below
- Be concise but specific — cite complaint details where relevant
- If the provided context is insufficient to answer, explicitly state that

Response structure:
**Summary:** 2-3 sentence overview of the main complaint pattern
**Key Issues:** Bullet list of the specific problems consumers report
**Companies Mentioned:** Which companies appear most in these complaints
**Consumer Impact:** How consumers describe being affected
**Recommendation:** One actionable suggestion for CrediTrust's compliance team
"""


# ── Module-level singletons (lazy-loaded on first use) ────────────────────────
# Avoids reloading the model/client on every function call.
_embedder   = None
_collection = None
_groq_client = None


def _get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Return the embedding model, loading it once and caching."""
    global _embedder
    if _embedder is None:
        logger.info(f'Loading embedding model: {model_name}')
        _embedder = SentenceTransformer(model_name)
        logger.info(f'Embedding model ready. Dimensions: {_embedder.get_sentence_embedding_dimension()}')
    return _embedder


def _get_collection(
    vectorstore_dir: str = None,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> chromadb.Collection:
    """Return the ChromaDB collection with flexible path detection."""
    global _collection
    if _collection is not None:
        return _collection

    # Try multiple possible paths (works both locally and on Streamlit Cloud)
    possible_paths = []

    if vectorstore_dir:
        possible_paths.append(vectorstore_dir)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths.extend([
        os.path.join(base_dir, 'data', 'vectorstore'),                    # Standard
        os.path.join(base_dir, '..', 'data', 'vectorstore'),             # One level up
        os.path.join(base_dir, 'Notebooks', 'data', 'vectorstore'),      # Your current structure
        os.path.join(base_dir, '..', 'Notebooks', 'data', 'vectorstore'), # Another possibility
    ])

    for path in possible_paths:
        if os.path.exists(path) and os.listdir(path):
            try:
                logger.info(f"Found vector store at: {path}")
                client = chromadb.PersistentClient(
                    path=path,
                    settings=Settings(anonymized_telemetry=False)
                )
                _collection = client.get_collection(collection_name)
                logger.info(f"Collection loaded successfully: {_collection.count():,} chunks")
                return _collection
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")

    # If no path worked
    raise FileNotFoundError(
        f"Vector store not found.\n"
        f"Searched in:\n" + "\n".join(possible_paths) + 
        f"\n\nPlease make sure the vectorstore folder is uploaded to the repository."
    )
def _get_groq_client() -> Groq:
    """Return the Groq client, initialising once and caching."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise EnvironmentError(
                'GROQ_API_KEY not found. '
                'Add it to your .env file: GROQ_API_KEY=your_key_here'
            )
        _groq_client = Groq(api_key=api_key)
        logger.info('Groq client initialised.')
    return _groq_client


# ── Core functions ────────────────────────────────────────────────────────────

def retrieve(
    question         : str,
    product_category : str,
    top_k            : int  = DEFAULT_TOP_K,
    vectorstore_dir  : str  = None,
    collection_name  : str  = DEFAULT_COLLECTION_NAME,
    embedding_model  : str  = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant complaint chunks.

    Args:
        question         : Natural language question from the user.
        product_category : One of the 5 target categories. Used as a metadata
                           filter to scope retrieval — prevents cross-category
                           contamination (e.g. fraud queries returning savings
                           account results).
        top_k            : Number of chunks to return. Default 5.
        vectorstore_dir  : Path to ChromaDB directory. Defaults to
                           data/vectorstore/ relative to this file.
        collection_name  : ChromaDB collection name. Default 'cfpb_complaints'.
        embedding_model  : HuggingFace model name. Must match Task 2.

    Returns:
        List of dicts, each containing:
            text         : complaint chunk text
            similarity   : cosine similarity score (0–1, higher = more relevant)
            company      : company named in the complaint
            state        : US state of the complainant
            issue        : CFPB issue category
            complaint_id : original CFPB complaint ID

    Raises:
        ValueError : if product_category is not one of the 5 valid categories.
        FileNotFoundError : if the vector store does not exist.
    """
    if product_category not in VALID_CATEGORIES:
        raise ValueError(
            f'Invalid product_category: "{product_category}". '
            f'Must be one of: {sorted(VALID_CATEGORIES)}'
        )

    embedder   = _get_embedder(embedding_model)
    collection = _get_collection(vectorstore_dir, collection_name)

    # Embed the question using the same model as the corpus
    q_embedding = embedder.encode(
        [question], normalize_embeddings=True
    ).tolist()

    # Query with metadata filter — always scope to the requested category
    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = top_k,
        where            = {'product_category': product_category},
        include          = ['documents', 'metadatas', 'distances']
    )

    chunks = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        chunks.append({
            'text'        : doc,
            'similarity'  : round(1 - dist, 4),   # convert distance → similarity
            'company'     : meta.get('company', 'Unknown'),
            'state'       : meta.get('state', ''),
            'issue'       : meta.get('issue', ''),
            'complaint_id': meta.get('complaint_id', ''),
        })

    logger.debug(
        f'Retrieved {len(chunks)} chunks for "{question[:50]}..." '
        f'(avg sim: {np.mean([c["similarity"] for c in chunks]):.4f})'
    )
    return chunks


def _build_user_prompt(
    question         : str,
    product_category : str,
    chunks           : list[dict]
) -> str:
    """
    Build the user message by injecting retrieved chunks as numbered evidence.

    Args:
        question         : original question
        product_category : product category label
        chunks           : retrieved complaint chunks from retrieve()

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_block = ''
    for i, chunk in enumerate(chunks, 1):
        context_block += (
            f'--- Complaint {i} ---\n'
            f'Company : {chunk["company"]}\n'
            f'State   : {chunk["state"]}\n'
            f'Issue   : {chunk["issue"]}\n'
            f'Excerpt : {chunk["text"]}\n\n'
        )

    return (
        f'Product Category: {product_category}\n\n'
        f'Question: {question}\n\n'
        f'Retrieved Complaint Excerpts:\n'
        f'{context_block}'
        f'Based solely on the complaint excerpts above, answer the question.'
    )


def generate(
    question         : str,
    product_category : str,
    chunks           : list[dict],
    groq_model       : str   = DEFAULT_GROQ_MODEL,
    temperature      : float = DEFAULT_TEMPERATURE,
    max_tokens       : int   = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Generate an answer using the Groq LLM given retrieved complaint context.

    Args:
        question         : original user question
        product_category : product category (included in prompt for context)
        chunks           : retrieved chunks from retrieve()
        groq_model       : Groq model name. Default 'llama-3.1-8b-instant'.
        temperature      : LLM temperature. 0.2 = factual and deterministic,
                           appropriate for analyst-style answers.
        max_tokens       : maximum tokens in the LLM response.

    Returns:
        Dict containing:
            answer        : generated answer text
            input_tokens  : number of prompt tokens consumed
            output_tokens : number of completion tokens generated
            model         : model name confirmed by the API response
    """
    client      = _get_groq_client()
    user_prompt = _build_user_prompt(question, product_category, chunks)

    response = client.chat.completions.create(
        model       = groq_model,
        temperature = temperature,
        max_tokens  = max_tokens,
        messages    = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_prompt},
        ]
    )

    return {
        'answer'       : response.choices[0].message.content,
        'input_tokens' : response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'model'        : response.model,
    }


def rag_pipeline(
    question         : str,
    product_category : str,
    top_k            : int   = DEFAULT_TOP_K,
    groq_model       : str   = DEFAULT_GROQ_MODEL,
    temperature      : float = DEFAULT_TEMPERATURE,
    vectorstore_dir  : str   = None,
    collection_name  : str   = DEFAULT_COLLECTION_NAME,
    embedding_model  : str   = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    """
    End-to-end RAG pipeline: question → retrieve → generate → answer.

    This is the main entry point. Combines retrieve() and generate() into a
    single call and returns a structured result with all metadata needed for
    evaluation and logging.

    Args:
        question         : Natural language question about consumer complaints.
        product_category : One of: 'Credit card', 'Personal loan',
                           'Buy Now Pay Later', 'Savings account',
                           'Money transfers'.
        top_k            : Number of complaint chunks to retrieve. Default 5.
        groq_model       : Groq model ID. Default 'llama-3.1-8b-instant'.
        temperature      : LLM temperature (0.0–1.0). Default 0.2.
        vectorstore_dir  : Path to ChromaDB. Defaults to data/vectorstore/.
        collection_name  : ChromaDB collection name.
        embedding_model  : Sentence transformer model name.

    Returns:
        Dict containing:
            question         : original question
            product_category : product category used
            chunks           : list of retrieved chunk dicts (from retrieve())
            answer           : LLM-generated answer string
            avg_similarity   : mean cosine similarity of retrieved chunks
            min_similarity   : minimum cosine similarity (weakest chunk)
            input_tokens     : LLM prompt tokens consumed
            output_tokens    : LLM completion tokens generated
            latency_sec      : total pipeline wall-clock time in seconds

    Example:
        >>> from rag_pipeline import rag_pipeline
        >>> result = rag_pipeline(
        ...     question         = "What are common unauthorized charge complaints?",
        ...     product_category = "Credit card"
        ... )
        >>> print(result["answer"])
        >>> print(f"Avg similarity: {result['avg_similarity']}")
    """
    start = time.time()

    # Step 1 — Retrieve relevant complaint chunks
    chunks = retrieve(
        question         = question,
        product_category = product_category,
        top_k            = top_k,
        vectorstore_dir  = vectorstore_dir,
        collection_name  = collection_name,
        embedding_model  = embedding_model,
    )

    # Step 2 — Generate answer from retrieved context
    generation = generate(
        question         = question,
        product_category = product_category,
        chunks           = chunks,
        groq_model       = groq_model,
        temperature      = temperature,
    )

    elapsed = time.time() - start
    similarities = [c['similarity'] for c in chunks]

    result = {
        'question'        : question,
        'product_category': product_category,
        'chunks'          : chunks,
        'answer'          : generation['answer'],
        'avg_similarity'  : round(float(np.mean(similarities)), 4),
        'min_similarity'  : round(float(min(similarities)), 4),
        'input_tokens'    : generation['input_tokens'],
        'output_tokens'   : generation['output_tokens'],
        'latency_sec'     : round(elapsed, 2),
    }

    logger.info(
        f'RAG complete | category={product_category} | '
        f'avg_sim={result["avg_similarity"]} | '
        f'latency={result["latency_sec"]}s'
    )
    return result
