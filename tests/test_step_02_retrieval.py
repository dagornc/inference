"""Tests unitaires pour le module step_02_retrieval.

Ce module teste le retrieval hybride (dense, sparse, fusion)
avec la configuration v2.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from inference_project.steps.step_02_retrieval import (
    DenseRetriever,
    SparseRetriever,
    HybridFusion,
    process_retrieval,
)


@pytest.fixture
def mock_config_v2():
    """Configuration v2 simulée pour retrieval."""
    return {
        "dense_retrieval": {
            "enabled": True,
            "vector_db": "chromadb",
            "collection_name": "test_collection",
            "top_k": 10,
            "model": {
                "provider": "sentence_transformers",
                "model_name": "BAAI/bge-m3",
                "embedding_dim": 1024,
            },
        },
        "sparse_retrieval": {
            "enabled": True,
            "method": "bm25",
            "index_dir": "/tmp/test_index",
            "top_k": 10,
            "params": {"k1": 1.5, "b": 0.75},
        },
        "fusion": {
            "enabled": True,
            "method": "rrf",
            "rrf_k": 60,
            "weights": {"dense": 0.6, "sparse": 0.4},
        },
        "retrieval": {
            "top_k": 10,
            "final_top_k": 10,
            "fusion": {"enabled": True},
        },
    }


@pytest.fixture
def mock_query_embeddings():
    """Mock query embeddings."""
    # 2 queries, 1024 dimensions
    return np.random.rand(2, 1024).astype(np.float32)


@pytest.fixture
def mock_queries_text():
    """Mock query texts."""
    return ["What is AI?", "How does machine learning work?"]


# =============================================================================
# DENSE RETRIEVER TESTS
# =============================================================================


@patch("chromadb.Client")
def test_dense_retriever_init_chromadb(mock_chromadb_client, mock_config_v2):
    """Test DenseRetriever initialization with ChromaDB."""
    mock_client = MagicMock()
    mock_chromadb_client.return_value = mock_client

    retriever = DenseRetriever(mock_config_v2)

    assert retriever.vector_db == "chromadb"
    assert retriever.collection_name == "test_collection"
    assert retriever.client is not None
    mock_chromadb_client.assert_called_once()


def test_dense_retriever_disabled(mock_config_v2):
    """Test DenseRetriever when disabled."""
    mock_config_v2["dense_retrieval"]["enabled"] = False

    retriever = DenseRetriever(mock_config_v2)

    assert retriever.client is None


@patch("chromadb.Client")
def test_dense_retriever_search(mock_chromadb_client, mock_config_v2, mock_query_embeddings):
    """Test DenseRetriever search functionality."""
    # Setup mock
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chromadb_client.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    # Mock search results
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Document 1 content", "Document 2 content"]],
        "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
        "distances": [[0.1, 0.2]],
    }

    retriever = DenseRetriever(mock_config_v2)
    results = retriever.search(mock_query_embeddings, top_k=2)

    # Verify results structure
    assert isinstance(results, list)
    # Should have results for first query
    if len(results) > 0:
        assert "id" in results[0] or len(results[0]) > 0


# =============================================================================
# SPARSE RETRIEVER TESTS
# =============================================================================


@patch("pyserini.search.lucene.LuceneSearcher")
def test_sparse_retriever_init(mock_lucene_searcher, mock_config_v2):
    """Test SparseRetriever initialization."""
    mock_searcher = MagicMock()
    mock_lucene_searcher.return_value = mock_searcher

    retriever = SparseRetriever(mock_config_v2)

    assert retriever.method == "bm25"
    assert retriever.searcher is not None
    mock_lucene_searcher.assert_called_once_with("/tmp/test_index")


def test_sparse_retriever_disabled(mock_config_v2):
    """Test SparseRetriever when disabled."""
    mock_config_v2["sparse_retrieval"]["enabled"] = False

    retriever = SparseRetriever(mock_config_v2)

    assert retriever.searcher is None


@patch("pyserini.search.lucene.LuceneSearcher")
def test_sparse_retriever_search(mock_lucene_searcher, mock_config_v2, mock_queries_text):
    """Test SparseRetriever search functionality."""
    # Setup mock
    mock_searcher = MagicMock()
    mock_lucene_searcher.return_value = mock_searcher

    # Mock search results
    mock_hit1 = Mock()
    mock_hit1.docid = "doc1"
    mock_hit1.score = 10.5
    mock_hit1.raw = "Document 1 content"

    mock_hit2 = Mock()
    mock_hit2.docid = "doc2"
    mock_hit2.score = 8.3
    mock_hit2.raw = "Document 2 content"

    mock_searcher.search.return_value = [mock_hit1, mock_hit2]

    retriever = SparseRetriever(mock_config_v2)
    results = retriever.search(mock_queries_text, top_k=2)

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 2  # 2 queries
    assert len(results[0]) == 2  # 2 results per query
    assert results[0][0]["id"] == "doc1"
    assert results[0][0]["score"] == 10.5
    assert results[0][0]["source"] == "sparse"


# =============================================================================
# HYBRID FUSION TESTS
# =============================================================================


def test_hybrid_fusion_init(mock_config_v2):
    """Test HybridFusion initialization."""
    fusion = HybridFusion(mock_config_v2)

    assert fusion.method == "rrf"


def test_hybrid_fusion_rrf():
    """Test RRF fusion method."""
    config = {
        "fusion": {
            "method": "rrf",
            "rrf_k": 60,
        }
    }

    fusion = HybridFusion(config)

    # Mock results
    dense_results = [
        [
            {"id": "doc1", "score": 0.9, "document": "Doc 1"},
            {"id": "doc2", "score": 0.7, "document": "Doc 2"},
        ]
    ]

    sparse_results = [
        [
            {"id": "doc2", "score": 10.0, "document": "Doc 2"},
            {"id": "doc3", "score": 8.0, "document": "Doc 3"},
        ]
    ]

    fused = fusion.fuse(dense_results, sparse_results)

    # Verify fusion
    assert isinstance(fused, list)
    assert len(fused) == 1  # 1 query
    assert len(fused[0]) >= 2  # At least 2 unique docs
    # doc2 should rank high (appears in both)
    doc_ids = [r["id"] for r in fused[0]]
    assert "doc2" in doc_ids


def test_hybrid_fusion_weighted():
    """Test weighted fusion method."""
    config = {
        "fusion": {
            "method": "weighted",
            "weights": {"dense": 0.6, "sparse": 0.4},
        }
    }

    fusion = HybridFusion(config)

    dense_results = [
        [
            {"id": "doc1", "score": 0.9, "document": "Doc 1"},
        ]
    ]

    sparse_results = [
        [
            {"id": "doc1", "score": 10.0, "document": "Doc 1"},
        ]
    ]

    fused = fusion.fuse(dense_results, sparse_results)

    assert isinstance(fused, list)
    assert len(fused[0]) >= 1


# =============================================================================
# PROCESS RETRIEVAL TESTS
# =============================================================================


@patch("inference_project.steps.step_02_retrieval.SparseRetriever")
@patch("inference_project.steps.step_02_retrieval.DenseRetriever")
@patch("inference_project.steps.step_02_retrieval.HybridFusion")
def test_process_retrieval(
    mock_fusion_class,
    mock_dense_class,
    mock_sparse_class,
    mock_config_v2,
    mock_query_embeddings,
    mock_queries_text,
):
    """Test process_retrieval orchestration."""
    # Setup mocks
    mock_dense = MagicMock()
    mock_dense.config = {"enabled": True}
    mock_dense.search.return_value = [
        [{"id": "doc1", "score": 0.9}],
        [{"id": "doc2", "score": 0.8}],
    ]
    mock_dense_class.return_value = mock_dense

    mock_sparse = MagicMock()
    mock_sparse.config = {"enabled": True}
    mock_sparse.search.return_value = [
        [{"id": "doc1", "score": 10.0}],
        [{"id": "doc3", "score": 9.0}],
    ]
    mock_sparse_class.return_value = mock_sparse

    mock_fusion = MagicMock()
    mock_fusion.fuse.return_value = [
        [{"id": "doc1", "score": 0.95}],
        [{"id": "doc2", "score": 0.85}],
    ]
    mock_fusion_class.return_value = mock_fusion

    # Run
    results = process_retrieval(
        mock_query_embeddings, mock_queries_text, mock_config_v2
    )

    # Verify
    assert isinstance(results, list)
    assert len(results) == 2  # 2 queries
    mock_dense.search.assert_called_once()
    mock_sparse.search.assert_called_once()
    mock_fusion.fuse.assert_called_once()


@patch("inference_project.steps.step_02_retrieval.SparseRetriever")
@patch("inference_project.steps.step_02_retrieval.DenseRetriever")
def test_process_retrieval_fusion_disabled(
    mock_dense_class,
    mock_sparse_class,
    mock_config_v2,
    mock_query_embeddings,
    mock_queries_text,
):
    """Test process_retrieval with fusion disabled."""
    # Disable fusion
    mock_config_v2["retrieval"]["fusion"]["enabled"] = False

    # Setup mocks
    mock_dense = MagicMock()
    mock_dense.config = {"enabled": True}
    mock_dense_results = [
        [{"id": "doc1", "score": 0.9}],
        [{"id": "doc2", "score": 0.8}],
    ]
    mock_dense.search.return_value = mock_dense_results
    mock_dense_class.return_value = mock_dense

    mock_sparse = MagicMock()
    mock_sparse.config = {"enabled": True}
    mock_sparse.search.return_value = [[{"id": "doc3"}]]
    mock_sparse_class.return_value = mock_sparse

    # Run
    results = process_retrieval(
        mock_query_embeddings, mock_queries_text, mock_config_v2
    )

    # Should return dense results only
    assert results == mock_dense_results
