"""
Open WebUI Pipeline for Liquid Glass RAG Inference.
This pipeline integrates the custom RAG steps (Embedding, Retrieval, Reranking, Compression)
and delegates the final generation to the Open WebUI backend for streaming support.
"""

from typing import List, Union, Generator, Iterator, Dict, Any
import os
import sys

# Add src to path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pydantic import BaseModel, Field
from open_webui.utils.chat import generate_chat_completion
from open_webui.constants import TASKS

# Import RAG steps
from inference_project.utils.config_loader import load_config
from inference_project.steps.step_01_embedding import process_query
from inference_project.steps.step_01_embedding_generation import process_embeddings
from inference_project.steps.step_02_retrieval import process_retrieval
from inference_project.steps.step_03_reranking import process_reranking
from inference_project.steps.step_04_compression import process_compression
from inference_project.steps.step_05_generation import PromptConstructor


class Pipe:
    class Valves(BaseModel):
        LLM_MODEL: str = Field(
            default="llama3.2:latest",
            description="The Ollama model to use for generation (e.g., llama3.2:latest)"
        )
        RAG_ENABLED: bool = Field(
            default=True,
            description="Enable or disable the RAG pipeline."
        )
        CONFIG_DIR: str = Field(
            default="/app/pipelines/config",
            description="Path to the configuration directory."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.type = "manifold"
        self.id = "liquid_glass_rag"
        self.name = "Liquid Glass RAG Pipeline"

    def pipes(self) -> List[dict]:
        return [
            {"id": self.id, "name": self.name},
        ]

    async def pipe(self, body: dict, __user__: dict, __request__: Any = None) -> Union[str, Generator, Iterator]:
        print(f"Pipe called with model: {body.get('model')}")
        
        # 1. Extract Query
        messages = body.get("messages", [])
        if not messages:
            return "Error: No messages found."
        
        last_message = messages[-1]
        query = last_message.get("content", "")
        
        if not self.valves.RAG_ENABLED:
            # Bypass RAG, just generate
            body["model"] = self.valves.LLM_MODEL
            return await generate_chat_completion(__request__, body, __user__)

        try:
            # 2. RAG Process
            print(f"Processing RAG for query: {query}")
            config_dir = self.valves.CONFIG_DIR
            
            # Step 1a: Query Processing
            config_01 = load_config("01_embedding_v2", config_dir)
            expanded_queries = process_query(query, config_01)
            
            # Step 1b: Embedding Generation
            embeddings_result = process_embeddings(expanded_queries, config_01)
            query_embeddings = embeddings_result.get("dense_embeddings")
            
            if query_embeddings is None:
                 raise Exception("Failed to generate embeddings")

            # Step 2: Retrieval
            config_02 = load_config("02_retrieval_v2", config_dir)
            retrieval_results = process_retrieval(query_embeddings, expanded_queries, config_02)
            
            # Step 3: Reranking
            config_03 = load_config("03_reranking_v2", config_dir)
            reranked_results = process_reranking(expanded_queries, retrieval_results, config_03)
            
            # Flatten results for compression
            flat_docs = []
            seen_ids = set()
            for query_results in reranked_results:
                for doc in query_results:
                    doc_id = doc.get("id")
                    if doc_id and doc_id not in seen_ids:
                        flat_docs.append(doc)
                        seen_ids.add(doc_id)
            
            # Step 4: Compression
            config_04 = load_config("04_compression_v2", config_dir)
            compressed_docs = process_compression(flat_docs, query)
            
            # 3. Prompt Construction
            config_05 = load_config("05_generation_v2", config_dir)
            prompt_constructor = PromptConstructor(config_05)
            
            # Build system and user prompts
            system_prompt, user_prompt = prompt_constructor.build_prompt(query, compressed_docs)
            
            # Update messages for generation
            # We replace the last message with the constructed user prompt (which includes context)
            # And ensure system prompt is set
            
            new_messages = []
            
            # Add system prompt if not present or replace existing?
            # Usually pipelines should respect existing conversation history, but RAG context 
            # is specific to the last query.
            # Strategy: Keep history, but modify the last message to include context.
            # And prepend system prompt.
            
            # Check if system message exists
            has_system = any(m.get("role") == "system" for m in messages)
            
            if not has_system:
                new_messages.append({"role": "system", "content": system_prompt})
            else:
                # Update existing system prompt? Or just append ours?
                # Let's append ours as a system instruction or replace the first system message.
                # For simplicity, let's add it as the first message.
                new_messages.append({"role": "system", "content": system_prompt})
                
            # Add history (excluding last message and system messages if we handled them)
            for msg in messages[:-1]:
                if msg.get("role") != "system":
                    new_messages.append(msg)
            
            # Add modified last message
            new_messages.append({"role": "user", "content": user_prompt})
            
            body["messages"] = new_messages
            body["model"] = self.valves.LLM_MODEL
            
            print(f"Delegating to Open WebUI with model {self.valves.LLM_MODEL}")
            return await generate_chat_completion(__request__, body, __user__)

        except Exception as e:
            print(f"Error in RAG Pipeline: {e}")
            return f"Error in RAG Pipeline: {e}"
