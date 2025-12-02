
import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import importlib.util
from pydantic import BaseModel
from typing import List, Dict, Any

# This is a mock user object for the pipe
class MockUser:
    def __init__(self):
        self.id = "mock_user"
        self.name = "Mock User"

class ChatBody(BaseModel):
    messages: List[Dict[str, Any]]
    model: str
    stream: bool = False

# Minimal web server to host the pipeline
app = FastAPI(
    title="Open WebUI RAG Pipeline Host",
    description="A minimal server to run a single Open WebUI compatible pipeline.",
)

# Load the pipeline dynamically
pipeline_path = "/app/pipelines/rag_pipeline.py"
spec = importlib.util.spec_from_file_location("rag_pipeline", pipeline_path)
rag_pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_pipeline_module)

# Instantiate the pipeline
pipeline = rag_pipeline_module.Pipe()
print(f"Pipeline '{pipeline.name}' loaded.")

# Add the src directory to the path to allow imports in the pipeline
sys.path.append('/app/src')


@app.get("/")
def read_root():
    return {"message": f"Pipeline '{pipeline.name}' is running."}

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request):
    """
    Mimics the OpenAI chat completions endpoint and routes the request
    to the loaded pipeline.
    """
    print(f"Received request for model: {body.model}")
    
    # The pipeline expects __user__ and __request__
    mock_user = MockUser()

    try:
        # Call the pipe method
        response_generator = await pipeline.pipe(
            body=body.dict(),
            __user__=mock_user,
            __request__=request
        )

        # Handle streaming and non-streaming responses
        if hasattr(response_generator, "__aiter__"):
            # This is a generator for streaming
            return StreamingResponse(response_generator, media_type="application/x-ndjson")
        else:
            # This is a single response
            # The pipeline might return a simple string or a dict.
            # We'll wrap it in a format similar to OpenAI's non-streaming response.
            final_response = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response_generator),
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            return JSONResponse(content=final_response)

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099)
