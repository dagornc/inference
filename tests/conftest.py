from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
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
