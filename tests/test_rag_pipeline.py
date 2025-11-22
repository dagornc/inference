import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipelines")))

# Mock open_webui module before importing rag_pipeline
mock_open_webui = MagicMock()
sys.modules["open_webui"] = mock_open_webui
sys.modules["open_webui.utils"] = MagicMock()
sys.modules["open_webui.utils.chat"] = MagicMock()
sys.modules["open_webui.constants"] = MagicMock()

from pipelines.rag_pipeline import Pipe

class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pipe = Pipe()
        self.pipe.valves.RAG_ENABLED = True
        self.pipe.valves.CONFIG_DIR = "/tmp/config" # Dummy path

    @patch("pipelines.rag_pipeline.generate_chat_completion", new_callable=AsyncMock)
    @patch("pipelines.rag_pipeline.load_config")
    @patch("pipelines.rag_pipeline.process_query")
    @patch("pipelines.rag_pipeline.process_embeddings")
    @patch("pipelines.rag_pipeline.process_retrieval")
    @patch("pipelines.rag_pipeline.process_reranking")
    @patch("pipelines.rag_pipeline.process_compression")
    @patch("pipelines.rag_pipeline.PromptConstructor")
    async def test_pipe_flow(self, MockPromptConstructor, mock_compression, mock_reranking, 
                       mock_retrieval, mock_embeddings, mock_query, mock_load_config, 
                       mock_generate):
        
        # Setup Mocks
        mock_load_config.return_value = {}
        mock_query.return_value = ["expanded query"]
        mock_embeddings.return_value = {"dense_embeddings": [[0.1, 0.2]]}
        mock_retrieval.return_value = [[{"id": "doc1", "document": "content1"}]]
        mock_reranking.return_value = [[{"id": "doc1", "document": "content1", "score": 0.9}]]
        mock_compression.return_value = [{"id": "doc1", "document": "compressed content1"}]
        
        mock_prompt_constructor_instance = MagicMock()
        MockPromptConstructor.return_value = mock_prompt_constructor_instance
        mock_prompt_constructor_instance.build_prompt.return_value = ("System Prompt", "User Prompt with Context")
        
        mock_generate.return_value = "Generated Response"

        # Input Body
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo"
        }
        user = {"id": "user1"}

        # Run Pipe
        result = await self.pipe.pipe(body, user)

        # Assertions
        mock_query.assert_called()
        mock_embeddings.assert_called()
        mock_retrieval.assert_called()
        mock_reranking.assert_called()
        mock_compression.assert_called()
        mock_prompt_constructor_instance.build_prompt.assert_called()
        
        # Check generate call
        args, kwargs = mock_generate.call_args
        called_body = args[1]
        
        self.assertEqual(called_body["model"], "llama3.2:latest")
        self.assertEqual(called_body["messages"][0]["role"], "system")
        self.assertEqual(called_body["messages"][0]["content"], "System Prompt")
        self.assertEqual(called_body["messages"][1]["role"], "user")
        self.assertEqual(called_body["messages"][1]["content"], "User Prompt with Context")
        
        self.assertEqual(result, "Generated Response")

if __name__ == "__main__":
    unittest.main()
