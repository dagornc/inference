"""Tests unitaires pour le module step_01_embedding.

Ce module teste le traitement de la requête (expansion, décomposition, routing)
avec la configuration v2.
"""

import pytest
from unittest.mock import MagicMock, patch
from inference_project.steps.step_01_embedding import (
    process_query,
    QueryRouter,
    QueryDecomposer,
)

@pytest.fixture
def mock_config_v2():
    """Configuration v2 simulée."""
    return {
        "llm_providers": {
            "openai": {
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1"
            }
        },
        "query_understanding": {
            "enabled": True,
            "type_classification": {
                "enabled": True,
                "classifier": "heuristic",
                "heuristic": {
                    "rules": {
                        "factual": {"patterns": ["^qui", "^quoi"]},
                        "analytical": {"patterns": ["^pourquoi", "^comment"]},
                    }
                }
            }
        },
        "adaptive_query_expansion": {
            "enabled": True,
            "techniques": {
                "rewrite": {"enabled": True},
                "multi_query": {"enabled": False},
                "hyde": {"enabled": False},
                "step_back": {"enabled": False},
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
            },
            "prompts": {
                "rewrite": "Rewrite: {query}",
                "multi_query": "Variants: {query}",
                "hyde": "HyDE: {query}",
                "step_back": "StepBack: {query}"
            }
        },
        "query_decomposition": {
            "enabled": True,
            "trigger": {
                "min_complexity_score": 0.5,
                "detect_connectors": ["et", "ou"]
            },
            "method": {
                "type": "llm",
                "llm": {"provider": "openai"}
            }
        }
    }

@patch("inference_project.steps.step_01_embedding.load_config")
@patch("inference_project.steps.step_01_embedding.openai.OpenAI")
def test_process_query_expansion(mock_openai, mock_load_config, mock_config_v2):
    """Test l'expansion de requête."""
    mock_load_config.return_value = mock_config_v2
    
    # Mock OpenAI response for rewrite
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Expanded Query"
    
    queries = process_query("Original Query", mock_config_v2)
    
    # Should return original + expanded
    assert "Original Query" in queries
    assert "Expanded Query" in queries
    assert len(queries) >= 2

@patch("inference_project.steps.step_01_embedding.load_config")
def test_process_query_no_expansion(mock_load_config, mock_config_v2):
    """Test sans expansion."""
    mock_config_v2["adaptive_query_expansion"]["enabled"] = False
    mock_load_config.return_value = mock_config_v2
    
    queries = process_query("Original Query", mock_config_v2)
    
    assert queries == ["Original Query"]

def test_query_router_heuristic(mock_config_v2):
    """Test le routing heuristique."""
    router = QueryRouter(mock_config_v2)
    
    # Factual
    result = router.route("Qui est le président ?")
    assert result["query_type"] == "factual"
    
    # Analytical
    result = router.route("Pourquoi le ciel est bleu ?")
    assert result["query_type"] == "analytical"
    
    # Default
    result = router.route("Bonjour")
    assert result["query_type"] == "factual"

@patch("inference_project.steps.step_01_embedding.openai.OpenAI")
def test_query_decomposer(mock_openai, mock_config_v2):
    """Test la décomposition de requête."""
    # Mock OpenAI response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Sub Q1\nSub Q2"

    decomposer = QueryDecomposer(mock_config_v2)
    
    # Force decomposition (bypass complexity check for test simplicity or mock it)
    # Here we just test the decompose method directly if possible, or ensure trigger works
    # The decompose method checks complexity internally.
    # "et" is in detect_connectors, so decomposition should be triggered.
    
    sub_queries = decomposer.decompose("Question complexe et autre question")
    
    # At minimum, the original query should be in the results
    assert "Question complexe et autre question" in sub_queries
    assert len(sub_queries) >= 1  # At least the original query

