"""Property-based tests for query processing using Hypothesis."""

from hypothesis import given
from hypothesis import strategies as st

from inference_project.steps.step_01_embedding import _query_cache
from inference_project.utils.config_loader import load_config


def test_process_query_property_based(mock_openai_client):
    """Property-based test for process_query with various text inputs."""
    from inference_project.steps.step_01_embedding import process_query

    @given(st.text(min_size=1))
    def run_test(query: str):
        config = load_config("01_embedding", "config")
        _query_cache.clear()

        results = process_query(query, config)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert query in results

    run_test()
