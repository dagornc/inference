from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    model_config_override: Optional[Dict[str, Any]] = None
    # Example override: {"llm_providers": {"ollama": {"model": "llama3"}}}

class Source(BaseModel):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict[str, Any]
