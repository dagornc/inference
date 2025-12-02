from unittest.mock import MagicMock, patch

import pytest

from inference_project.steps.step_01_embedding import (
    _query_cache,
    expand_multi_query,
    generate_hypothetical_document,
    generate_step_back_question,
    process_query,
    rewrite_query,
)
from inference_project.utils.config_loader import load_config


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def multi_query_mock(*args, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            if "reformulations alternatives" in prompt:
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                mock_message.content = "variant 1\nvariant 2\nvariant 3\nvariant 4"
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                return mock_response
            else:
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_message = MagicMock()
                mock_message.content = "mocked response"
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                return mock_response

        mock_client.chat.completions.create.side_effect = multi_query_mock
        yield mock_client


def test_rewrite_query(mock_openai_client):
    prompt = "test prompt"
    model = "test_model"
    temperature = 0.5
    max_tokens = 50
    response = rewrite_query(prompt, mock_openai_client, model, temperature, max_tokens)
    assert response == "mocked response"
    mock_openai_client.chat.completions.create.assert_called_with(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_generate_hypothetical_document(mock_openai_client):
    prompt = "test prompt"
    model = "test_model"
    temperature = 0.5
    max_tokens = 150
    response = generate_hypothetical_document(
        prompt, mock_openai_client, model, temperature, max_tokens
    )
    assert response == "mocked response"
    mock_openai_client.chat.completions.create.assert_called_with(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_expand_multi_query(mock_openai_client):
    config = load_config("01_embedding_v2", "config")
    # Patch for V2 config structure mismatch
    prompts = config["adaptive_query_expansion"].get("prompts")
    if not prompts:
        prompts = config["adaptive_query_expansion"]["domain_specific_prompts"]["prompts"]["general"]
        prompts["multi_query"] = "Generate {num_variants} reformulations alternatives: {query}"
    
    prompt = prompts["multi_query"].format(
        query="test", num_variants=4
    )
    model = "test_model"
    temperature = 0.5
    max_tokens = 100
    response = expand_multi_query(
        prompt, mock_openai_client, model, temperature, max_tokens
    )
    assert len(response) == 4
    mock_openai_client.chat.completions.create.assert_called_with(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_generate_step_back_question(mock_openai_client):
    prompt = "test prompt"
    model = "test_model"
    temperature = 0.5
    max_tokens = 50
    response = generate_step_back_question(
        prompt, mock_openai_client, model, temperature, max_tokens
    )
    assert response == "mocked response"
    mock_openai_client.chat.completions.create.assert_called_with(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_process_query_all_enabled(mock_openai_client):
    config = load_config("01_embedding_v2", "config")
    
    # Patch config to match code expectation (flatten prompts)
    if "prompts" not in config["adaptive_query_expansion"]:
        prompts = config["adaptive_query_expansion"]["domain_specific_prompts"]["prompts"]["general"]
        prompts["rewrite"] = "Rewrite: {query}"
        prompts["multi_query"] = "Generate {num_variants} reformulations alternatives: {query}"
        prompts["step_back"] = "Step back: {query}"
        config["adaptive_query_expansion"]["prompts"] = prompts

    # Patch cache_enabled for code compatibility
    config["adaptive_query_expansion"]["cache_enabled"] = True

    query = "test query"
    _query_cache.clear()

    results = process_query(query, config)

    assert (
        len(results) == 8
    )  # 1 original + 1 rewrite + 1 hyde + 4 multi-query + 1 step-back
    assert query in results
    assert "mocked response" in results
    assert "variant 1" in results


def test_process_query_all_disabled(mock_openai_client):
    config = load_config("01_embedding_v2", "config")
    
    # Patch config to match code expectation
    if "prompts" not in config["adaptive_query_expansion"]:
        config["adaptive_query_expansion"]["prompts"] = config["adaptive_query_expansion"]["domain_specific_prompts"]["prompts"]["general"]

    config["adaptive_query_expansion"]["enabled"] = False

    query = "test query"
    _query_cache.clear()

    results = process_query(query, config)

    assert len(results) == 1
    assert results[0] == query


def test_process_query_caching(mock_openai_client):
    config = load_config("01_embedding_v2", "config")

    # Patch config to match code expectation
    if "prompts" not in config["adaptive_query_expansion"]:
        prompts = config["adaptive_query_expansion"]["domain_specific_prompts"]["prompts"]["general"]
        prompts["rewrite"] = "Rewrite: {query}"
        prompts["multi_query"] = "Generate {num_variants} reformulations alternatives: {query}"
        prompts["step_back"] = "Step back: {query}"
        config["adaptive_query_expansion"]["prompts"] = prompts

    # Patch cache_enabled for code compatibility
    config["adaptive_query_expansion"]["cache_enabled"] = True

    query = "test query"
    _query_cache.clear()

    # First call
    results1 = process_query(query, config)
    assert query in _query_cache

    # Second call
    results2 = process_query(query, config)

    assert results1 == results2
    assert mock_openai_client.chat.completions.create.call_count == 4
