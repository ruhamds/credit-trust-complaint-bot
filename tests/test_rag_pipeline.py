"""
tests/test_rag_pipeline.py
--------------------------
Unit tests for rag_pipeline.py

Tests are isolated using mocks — no real API calls, no real vector store.
This means tests run in CI without GROQ_API_KEY or ChromaDB on disk.

Run:
    pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import time

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_chunks():
    """Five realistic fake complaint chunks — one per category."""
    return [
        {
            'text'        : 'unauthorized charge appeared on my credit card statement.',
            'similarity'  : 0.74,
            'company'     : 'BANK OF AMERICA, NATIONAL ASSOCIATION',
            'state'       : 'CA',
            'issue'       : 'Problem with a purchase shown on your statement',
            'complaint_id': '12345',
        },
        {
            'text'        : 'the interest rate on my loan was not disclosed upfront.',
            'similarity'  : 0.68,
            'company'     : 'SYNCHRONY FINANCIAL',
            'state'       : 'TX',
            'issue'       : 'Charged fees or interest you didn\'t expect',
            'complaint_id': '12346',
        },
        {
            'text'        : 'klarna reported a late payment that i had already paid.',
            'similarity'  : 0.63,
            'company'     : 'Klarna AB',
            'state'       : 'NY',
            'issue'       : 'Incorrect information on your report',
            'complaint_id': '12347',
        },
        {
            'text'        : 'my savings account was closed without any prior notice.',
            'similarity'  : 0.61,
            'company'     : 'CITIBANK, N.A.',
            'state'       : 'FL',
            'issue'       : 'Closing an account',
            'complaint_id': '12348',
        },
        {
            'text'        : 'wire transfer never arrived after two weeks of waiting.',
            'similarity'  : 0.59,
            'company'     : 'WESTERN UNION COMPANY, THE',
            'state'       : 'IL',
            'issue'       : 'Money was not available when promised',
            'complaint_id': '12349',
        },
    ]


@pytest.fixture
def mock_generation():
    """Fake LLM generation result."""
    return {
        'answer'       : '**Summary:** Consumers report unauthorized charges.\n**Key Issues:**\n- Unauthorized transactions\n**Companies Mentioned:** Bank of America\n**Consumer Impact:** Financial loss\n**Recommendation:** Review fraud detection procedures.',
        'input_tokens' : 512,
        'output_tokens': 256,
        'model'        : 'llama-3.1-8b-instant',
    }


# ─────────────────────────────────────────────────────────────────────────────
# retrieve() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieve:

    def test_returns_correct_chunk_count(self, mock_chunks):
        """retrieve() returns exactly top_k chunks."""
        with patch('rag_pipeline._get_embedder') as mock_emb, \
             patch('rag_pipeline._get_collection') as mock_col:

            # Mock embedder
            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            # Mock ChromaDB collection query result
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': [[c['text']       for c in mock_chunks]],
                'metadatas': [[{k: v for k, v in c.items() if k != 'text' and k != 'similarity'}
                               for c in mock_chunks]],
                'distances': [[1 - c['similarity'] for c in mock_chunks]],
            }
            mock_col.return_value = mock_collection

            from rag_pipeline import retrieve
            result = retrieve('unauthorized charge', 'Credit card', top_k=5)

            assert len(result) == 5

    def test_invalid_category_raises_value_error(self):
        """retrieve() raises ValueError for unknown product category."""
        from rag_pipeline import retrieve
        with pytest.raises(ValueError, match='Invalid product_category'):
            retrieve('test question', 'Invalid Category')

    def test_all_valid_categories_accepted(self):
        """retrieve() accepts all 5 valid categories without raising."""
        from rag_pipeline import VALID_CATEGORIES, retrieve

        with patch('rag_pipeline._get_embedder') as mock_emb, \
             patch('rag_pipeline._get_collection') as mock_col:

            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': [['sample text']],
                'metadatas': [[{'company': 'Test', 'state': 'CA',
                                'issue': 'test', 'complaint_id': '1',
                                'product_category': 'Credit card'}]],
                'distances': [[0.3]],
            }
            mock_col.return_value = mock_collection

            for category in VALID_CATEGORIES:
                result = retrieve('test question', category, top_k=1)
                assert isinstance(result, list)

    def test_chunk_has_required_keys(self, mock_chunks):
        """Each returned chunk contains all required keys."""
        required_keys = {'text', 'similarity', 'company', 'state', 'issue', 'complaint_id'}

        with patch('rag_pipeline._get_embedder') as mock_emb, \
             patch('rag_pipeline._get_collection') as mock_col:

            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': [[c['text'] for c in mock_chunks[:3]]],
                'metadatas': [[{k: v for k, v in c.items() if k not in ('text', 'similarity')}
                               for c in mock_chunks[:3]]],
                'distances': [[1 - c['similarity'] for c in mock_chunks[:3]]],
            }
            mock_col.return_value = mock_collection

            from rag_pipeline import retrieve
            results = retrieve('test', 'Credit card', top_k=3)

            for chunk in results:
                assert required_keys.issubset(chunk.keys()), \
                    f'Missing keys: {required_keys - chunk.keys()}'

    def test_similarity_is_in_valid_range(self, mock_chunks):
        """All similarity scores must be between 0 and 1."""
        with patch('rag_pipeline._get_embedder') as mock_emb, \
             patch('rag_pipeline._get_collection') as mock_col:

            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': [[c['text'] for c in mock_chunks]],
                'metadatas': [[{k: v for k, v in c.items() if k not in ('text', 'similarity')}
                               for c in mock_chunks]],
                'distances': [[1 - c['similarity'] for c in mock_chunks]],
            }
            mock_col.return_value = mock_collection

            from rag_pipeline import retrieve
            results = retrieve('test', 'Credit card', top_k=5)

            for chunk in results:
                assert 0.0 <= chunk['similarity'] <= 1.0, \
                    f'Similarity {chunk["similarity"]} out of range [0, 1]'

    def test_similarity_sorted_descending(self, mock_chunks):
        """Returned chunks should be ordered by similarity, highest first."""
        with patch('rag_pipeline._get_embedder') as mock_emb, \
             patch('rag_pipeline._get_collection') as mock_col:

            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            # Return chunks sorted by ChromaDB (ascending distance = descending similarity)
            sorted_chunks = sorted(mock_chunks, key=lambda x: x['similarity'], reverse=True)
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'documents': [[c['text'] for c in sorted_chunks]],
                'metadatas': [[{k: v for k, v in c.items() if k not in ('text', 'similarity')}
                               for c in sorted_chunks]],
                'distances': [[1 - c['similarity'] for c in sorted_chunks]],
            }
            mock_col.return_value = mock_collection

            from rag_pipeline import retrieve
            results = retrieve('test', 'Credit card', top_k=5)

            similarities = [r['similarity'] for r in results]
            assert similarities == sorted(similarities, reverse=True), \
                'Chunks are not ordered by descending similarity'


# ─────────────────────────────────────────────────────────────────────────────
# generate() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:

    def test_returns_required_keys(self, mock_chunks, mock_generation):
        """generate() returns dict with answer, input_tokens, output_tokens, model."""
        required_keys = {'answer', 'input_tokens', 'output_tokens', 'model'}

        with patch('rag_pipeline._get_groq_client') as mock_groq:
            mock_client   = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content  = mock_generation['answer']
            mock_response.usage.prompt_tokens          = mock_generation['input_tokens']
            mock_response.usage.completion_tokens      = mock_generation['output_tokens']
            mock_response.model                        = mock_generation['model']
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            from rag_pipeline import generate
            result = generate('test question', 'Credit card', mock_chunks)

            assert required_keys.issubset(result.keys())

    def test_answer_is_non_empty_string(self, mock_chunks, mock_generation):
        """generate() returns a non-empty answer string."""
        with patch('rag_pipeline._get_groq_client') as mock_groq:
            mock_client   = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content  = mock_generation['answer']
            mock_response.usage.prompt_tokens          = 512
            mock_response.usage.completion_tokens      = 256
            mock_response.model                        = 'llama-3.1-8b-instant'
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            from rag_pipeline import generate
            result = generate('test question', 'Credit card', mock_chunks)

            assert isinstance(result['answer'], str)
            assert len(result['answer'].strip()) > 0

    def test_token_counts_are_positive_integers(self, mock_chunks, mock_generation):
        """Token counts must be positive integers."""
        with patch('rag_pipeline._get_groq_client') as mock_groq:
            mock_client   = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content  = mock_generation['answer']
            mock_response.usage.prompt_tokens          = 512
            mock_response.usage.completion_tokens      = 256
            mock_response.model                        = 'llama-3.1-8b-instant'
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            from rag_pipeline import generate
            result = generate('test question', 'Credit card', mock_chunks)

            assert isinstance(result['input_tokens'],  int) and result['input_tokens']  > 0
            assert isinstance(result['output_tokens'], int) and result['output_tokens'] > 0


# ─────────────────────────────────────────────────────────────────────────────
# rag_pipeline() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRagPipeline:

    def test_returns_all_required_keys(self, mock_chunks, mock_generation):
        """rag_pipeline() result contains all required output keys."""
        required_keys = {
            'question', 'product_category', 'chunks',
            'answer', 'avg_similarity', 'min_similarity',
            'input_tokens', 'output_tokens', 'latency_sec'
        }

        with patch('rag_pipeline.retrieve', return_value=mock_chunks), \
             patch('rag_pipeline.generate', return_value=mock_generation):

            from rag_pipeline import rag_pipeline
            result = rag_pipeline('test question', 'Credit card')

            assert required_keys.issubset(result.keys()), \
                f'Missing keys: {required_keys - result.keys()}'

    def test_avg_similarity_is_mean_of_chunks(self, mock_chunks, mock_generation):
        """avg_similarity equals mean of chunk similarities."""
        import numpy as np

        with patch('rag_pipeline.retrieve', return_value=mock_chunks), \
             patch('rag_pipeline.generate', return_value=mock_generation):

            from rag_pipeline import rag_pipeline
            result = rag_pipeline('test question', 'Credit card')

            expected_avg = round(float(np.mean([c['similarity'] for c in mock_chunks])), 4)
            assert result['avg_similarity'] == expected_avg

    def test_min_similarity_is_min_of_chunks(self, mock_chunks, mock_generation):
        """min_similarity equals minimum of chunk similarities."""
        with patch('rag_pipeline.retrieve', return_value=mock_chunks), \
             patch('rag_pipeline.generate', return_value=mock_generation):

            from rag_pipeline import rag_pipeline
            result = rag_pipeline('test question', 'Credit card')

            expected_min = round(min(c['similarity'] for c in mock_chunks), 4)
            assert result['min_similarity'] == expected_min

    def test_latency_is_positive_float(self, mock_chunks, mock_generation):
        """Latency must be a positive number."""
        with patch('rag_pipeline.retrieve', return_value=mock_chunks), \
             patch('rag_pipeline.generate', return_value=mock_generation), \
             patch('rag_pipeline.time.time', side_effect=[0.0, 0.1]):  # Simulate 0.1 second latency

            from rag_pipeline import rag_pipeline
            result = rag_pipeline('test question', 'Credit card')

            assert isinstance(result['latency_sec'], float)
            assert result['latency_sec'] > 0

    def test_question_and_category_passed_through(self, mock_chunks, mock_generation):
        """question and product_category in result match input."""
        with patch('rag_pipeline.retrieve', return_value=mock_chunks), \
             patch('rag_pipeline.generate', return_value=mock_generation):

            from rag_pipeline import rag_pipeline
            result = rag_pipeline(
                question         = 'What are common fraud complaints?',
                product_category = 'Money transfers'
            )

            assert result['question']         == 'What are common fraud complaints?'
            assert result['product_category'] == 'Money transfers'

    def test_invalid_category_propagates_error(self):
        """ValueError from retrieve() propagates through rag_pipeline()."""
        from rag_pipeline import rag_pipeline
        with pytest.raises(ValueError):
            rag_pipeline('test', 'Not A Category')