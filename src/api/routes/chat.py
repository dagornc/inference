from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import os
import numpy as np

from inference_project.utils.config_loader import load_config
from inference_project.steps.step_01_embedding import process_query
from inference_project.steps.step_01_embedding_generation import process_embeddings
from inference_project.steps.step_02_retrieval import process_retrieval
from inference_project.steps.step_03_reranking import process_reranking
from inference_project.steps.step_04_compression import process_compression
from inference_project.steps.step_05_generation import process_generation

from src.api.models import ChatRequest, ChatResponse, Source

router = APIRouter()

# Cache configs to avoid reloading from disk on every request
_CONFIG_CACHE: Dict[str, Any] = {}

def get_config(step_name: str, config_file: str) -> Dict[str, Any]:
    if step_name not in _CONFIG_CACHE:
        # Assuming config dir is at project root / config
        # We need to find the absolute path to config dir
        # Current file is src/api/routes/chat.py
        # Config is at ../../../config
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_dir = os.path.join(base_dir, "config")
        _CONFIG_CACHE[step_name] = load_config(config_file, config_dir)
    return _CONFIG_CACHE[step_name]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.query
        
        # --- Step 1a: Query Processing & Expansion ---
        config_01 = get_config("step_01", "01_embedding_v2")
        # Apply overrides if any (simplified)
        if request.model_config_override:
            pass 
            
        expanded_queries = process_query(query, config_01)
        
        # --- Step 1b: Embedding Generation ---
        # We need embeddings for retrieval
        embeddings_result = process_embeddings(expanded_queries, config_01)
        query_embeddings = embeddings_result.get("dense_embeddings")
        
        if query_embeddings is None:
             raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        # --- Step 2: Retrieval ---
        config_02 = get_config("step_02", "02_retrieval_v2")
        retrieval_results = process_retrieval(query_embeddings, expanded_queries, config_02)
        
        # retrieval_results is a list of results per query. We flatten or use the first?
        # Usually retrieval returns a list of lists (one list of docs per query).
        # We need to flatten for reranking or pass as is?
        # process_reranking signature: queries: List[str], results: List[List[Dict[str, Any]]]
        # So we pass it as is.
        
        # --- Step 3: Reranking ---
        config_03 = get_config("step_03", "03_reranking_v2")
        reranked_results = process_reranking(expanded_queries, retrieval_results, config_03)
        
        # reranked_results is List[List[Dict]]. We probably want to flatten to get a single context for generation?
        # Or maybe process_generation handles it?
        # process_generation signature: query: str, documents: List[Dict[str, Any]]
        # So we need a single list of documents.
        # We should probably take the top results from the reranked list.
        # Let's flatten and deduplicate by ID.
        
        flat_docs = []
        seen_ids = set()
        for query_results in reranked_results:
            for doc in query_results:
                doc_id = doc.get("id")
                if doc_id and doc_id not in seen_ids:
                    flat_docs.append(doc)
                    seen_ids.add(doc_id)
        
        # --- Step 4: Compression ---
        config_04 = get_config("step_04", "04_compression_v2")
        compressed_docs = process_compression(flat_docs, query) # Using original query for compression context
        
        # --- Step 5: Generation ---
        config_05 = get_config("step_05", "05_generation_v2")
        
        # Apply overrides for LLM model if requested
        if request.model_config_override:
             # Example: {"llm_providers": {"ollama": {"model": "llama3"}}}
             # We might need to patch global config or step config.
             # For now, let's assume we patch step config's llm section if present
             pass

        generation_result = process_generation(query, compressed_docs, config_05)
        
        # generation_result is a dict with 'answer', 'sources', 'metadata' etc.
        
        # Map to ChatResponse
        sources = []
        # Assuming generation_result['sources'] is a list of source info or we extract from docs
        # The example output showed "Sources: {response['num_sources']}"
        # Let's look at ResponseFormatter in step 05.
        # It seems it returns a dict.
        
        # We'll try to extract sources from the used documents or the response metadata.
        # For now, let's just map what we have.
        
        answer = generation_result.get("answer", "")
        meta = generation_result.get("metadata", {})
        
        # Construct sources from compressed_docs (which were used)
        # Or better, use what step 5 returns if it filters them.
        # Let's use compressed_docs as sources for now.
        for i, doc in enumerate(compressed_docs[:5]): # Limit to top 5
            sources.append(Source(
                id=str(doc.get("id", i)),
                content=doc.get("document", "")[:200] + "...",
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            ))
            
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata=meta
        )

    except Exception as e:
        # Log error
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
