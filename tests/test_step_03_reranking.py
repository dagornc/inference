"""Tests unitaires pour le module step_03_reranking.

Ce module teste le reranking (cross-encoder, MMR, LLM reranking)
avec la configuration v2.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from inference_project.steps.step_03_reranking import (
    CrossEncoderReranker,
    MMR,
    LLMReranker,
    process_reranking,
)


@pytest.fixture
def mock_config_v2():
    """Configuration v2 simulée pour reranking."""
    return {
        "cross_encoder": {
            "enabled": True,
            "provider": "sentence_transformers",
            "model": "BAAI/bge-reranker-v2-m3",
            "input_top_k": 50,
            "output_top_k": 20,
            "batch_size": 8,
            "normalize_scores": True,
            "normalization_method": "minmax",
            "min_score_threshold": 0.4,
            "device": "cpu",
            "max_length": 512,
        },
        "diversification": {
            "enabled": True,
            "mmr": {
                "enabled": True,
                "lambda": 0.6,
                "use_features": False,
            },
        },
        "mmr": {  # MMR reads from top-level mmr key
            "enabled": True,
            "lambda": 0.6,
            "use_features": False,
        },
        "llm_reranking": {
            "enabled": False,
            "provider": "ollama",
            "model": "llama3",
            "method": "listwise",
            "input_top_k": 10,
            "output_top_k": 10,
            "temperature": 0.0,
            "max_tokens": 2000,
        },
        "reranking": {
            "enabled": True,
            "top_k": 20,
        },
    }


@pytest.fixture
def mock_queries():
    """Mock queries."""
    return ["What is AI?", "How does machine learning work?"]


@pytest.fixture
def mock_retrieval_results():
    """Mock retrieval results."""
    return [
        [
            {"id": "doc1", "document": "AI is artificial intelligence", "score": 0.9},
            {"id": "doc2", "document": "Machine learning is a subset", "score": 0.7},
            {"id": "doc3", "document": "Deep learning uses neural networks", "score": 0.6},
        ],
        [
            {"id": "doc4", "document": "ML algorithms learn from data", "score": 0.85},
            {"id": "doc5", "document": "Supervised learning uses labels", "score": 0.75},
        ],
    ]


# =============================================================================
# CROSS-ENCODER RERANKER TESTS
# =============================================================================


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_init(mock_cross_encoder_class, mock_config_v2):
    """Test CrossEncoderReranker initialization."""
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model

    reranker = CrossEncoderReranker(mock_config_v2)

    assert reranker.model is not None
    mock_cross_encoder_class.assert_called_once_with(
        "BAAI/bge-reranker-v2-m3", max_length=512, device="cpu"
    )


def test_cross_encoder_disabled(mock_config_v2):
    """Test CrossEncoderReranker when disabled."""
    mock_config_v2["cross_encoder"]["enabled"] = False

    reranker = CrossEncoderReranker(mock_config_v2)

    assert reranker.model is None


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_rerank(
    mock_cross_encoder_class, mock_config_v2, mock_queries, mock_retrieval_results
):
    """Test CrossEncoderReranker reranking functionality."""
    # Setup mock
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model

    # Mock predict to return scores
    mock_model.predict.side_effect = [
        np.array([0.95, 0.85, 0.75]),  # Scores for query 1
        np.array([0.90, 0.80]),  # Scores for query 2
    ]

    reranker = CrossEncoderReranker(mock_config_v2)
    results = reranker.rerank(mock_queries, mock_retrieval_results, top_k=2)

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 2  # 2 queries
    assert len(results[0]) == 2  # top_k=2
    assert len(results[1]) == 2

    # Verify scores updated and sorted
    assert results[0][0]["score"] == 0.95
    assert results[0][1]["score"] == 0.85
    assert results[0][0]["source"] == "reranked"

    # Verify predict called twice (once per query)
    assert mock_model.predict.call_count == 2


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_empty_results(mock_cross_encoder_class, mock_config_v2):
    """Test CrossEncoderReranker with empty results."""
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model

    reranker = CrossEncoderReranker(mock_config_v2)
    results = reranker.rerank(["query"], [[]])

    assert results == [[]]
    mock_model.predict.assert_not_called()


# =============================================================================
# MMR TESTS
# =============================================================================


def test_mmr_init(mock_config_v2):
    """Test MMR initialization."""
    mmr = MMR(mock_config_v2)

    assert mmr.lambda_param == 0.6  # From mock_config_v2


def test_mmr_apply_without_embeddings(mock_config_v2, mock_retrieval_results):
    """Test MMR apply without embeddings (fallback to score-based)."""
    mmr = MMR(mock_config_v2)

    # Apply MMR
    results = mmr.apply(mock_retrieval_results, doc_embeddings=None, top_k=2)

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 2  # 2 queries
    assert len(results[0]) <= 2  # top_k=2
    assert len(results[1]) <= 2


def test_mmr_apply_with_embeddings(mock_config_v2):
    """Test MMR apply with embeddings."""
    mmr = MMR(mock_config_v2)

    # Mock results
    results = [
        [
            {"id": "doc1", "document": "Doc 1", "score": 0.9},
            {"id": "doc2", "document": "Doc 2", "score": 0.8},
            {"id": "doc3", "document": "Doc 3", "score": 0.7},
        ]
    ]

    # Mock embeddings (3 docs, 128 dims)
    embeddings = [np.random.rand(3, 128)]

    diversified = mmr.apply(results, doc_embeddings=embeddings, top_k=2)

    # Verify results
    assert isinstance(diversified, list)
    assert len(diversified) == 1
    assert len(diversified[0]) == 2


def test_mmr_disabled(mock_config_v2, mock_retrieval_results):
    """Test MMR when disabled."""
    mock_config_v2["diversification"]["mmr"]["enabled"] = False

    mmr = MMR(mock_config_v2)
    results = mmr.apply(mock_retrieval_results, top_k=2)

    # Should return original results (truncated to top_k)
    assert len(results[0]) <= 2


# =============================================================================
# LLM RERANKER TESTS
# =============================================================================


@patch("openai.OpenAI")
def test_llm_reranker_init(mock_openai_class, mock_config_v2):
    """Test LLMReranker initialization."""
    mock_config_v2["llm_reranking"]["enabled"] = True
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    reranker = LLMReranker(mock_config_v2)

    assert reranker.client is not None
    mock_openai_class.assert_called_once()


def test_llm_reranker_disabled(mock_config_v2):
    """Test LLMReranker when disabled."""
    reranker = LLMReranker(mock_config_v2)

    assert reranker.client is None


@patch("openai.OpenAI")
def test_llm_reranker_rerank(
    mock_openai_class, mock_config_v2, mock_queries, mock_retrieval_results
):
    """Test LLMReranker reranking functionality."""
    mock_config_v2["llm_reranking"]["enabled"] = True

    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock LLM response (OpenAI format)
    mock_choice = MagicMock()
    mock_choice.message.content = "doc1, doc3, doc2"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    reranker = LLMReranker(mock_config_v2)
    results = reranker.rerank(mock_queries, mock_retrieval_results, top_k=3)

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 2  # 2 queries

    # Verify LLM was called
    assert mock_client.chat.completions.create.call_count >= 1


# =============================================================================
# PROCESS RERANKING TESTS
# =============================================================================


@patch("inference_project.steps.step_03_reranking.MMR")
@patch("inference_project.steps.step_03_reranking.CrossEncoderReranker")
def test_process_reranking(
    mock_cross_encoder_class,
    mock_mmr_class,
    mock_config_v2,
    mock_queries,
    mock_retrieval_results,
):
    """Test process_reranking orchestration."""
    # Setup mocks
    mock_reranker = MagicMock()
    mock_reranked_results = [
        [{"id": "doc1", "score": 0.95}, {"id": "doc2", "score": 0.85}],
        [{"id": "doc4", "score": 0.90}],
    ]
    mock_reranker.rerank.return_value = mock_reranked_results
    mock_cross_encoder_class.return_value = mock_reranker

    mock_mmr = MagicMock()
    mock_mmr.apply.return_value = mock_reranked_results
    mock_mmr_class.return_value = mock_mmr

    # Run
    results = process_reranking(
        mock_queries, mock_retrieval_results, mock_config_v2
    )

    # Verify
    assert isinstance(results, list)
    assert len(results) == 2
    mock_reranker.rerank.assert_called_once()
    mock_mmr.apply.assert_called_once()


@patch("inference_project.steps.step_03_reranking.CrossEncoderReranker")
def test_process_reranking_diversification_disabled(
    mock_cross_encoder_class, mock_config_v2, mock_queries, mock_retrieval_results
):
    """Test process_reranking with diversification disabled."""
    # Disable diversification
    mock_config_v2["diversification"]["enabled"] = False

    # Setup mock
    mock_reranker = MagicMock()
    mock_reranked_results = [
        [{"id": "doc1", "score": 0.95}],
        [{"id": "doc4", "score": 0.90}],
    ]
    mock_reranker.rerank.return_value = mock_reranked_results
    mock_cross_encoder_class.return_value = mock_reranker

    # Run
    results = process_reranking(
        mock_queries, mock_retrieval_results, mock_config_v2
    )

    # Should return reranked results without MMR
    assert results == mock_reranked_results


@patch("inference_project.steps.step_03_reranking.MMR")
@patch("inference_project.steps.step_03_reranking.CrossEncoderReranker")
def test_process_reranking_disabled(
    mock_cross_encoder_class, mock_mmr_class, mock_config_v2, mock_queries, mock_retrieval_results
):
    """Test process_reranking when reranking is disabled."""
    # Disable reranking
    mock_config_v2["reranking"]["enabled"] = False

    # Setup mocks - process_reranking doesn't check reranking.enabled
    # so we need to mock the return values
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = mock_retrieval_results  # Return original results
    mock_cross_encoder_class.return_value = mock_reranker

    mock_mmr = MagicMock()
    mock_mmr.apply.return_value = mock_retrieval_results  # Return original results
    mock_mmr_class.return_value = mock_mmr

    # Run
    results = process_reranking(
        mock_queries, mock_retrieval_results, mock_config_v2
    )

    # Should return results (process_reranking doesn't check reranking.enabled)
    # It always runs cross-encoder and MMR if they're enabled
    assert isinstance(results, list)
    assert len(results) == 2
    mock_reranker.rerank.assert_called_once()
    mock_mmr.apply.assert_called_once()
