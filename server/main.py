from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from server.model_service import MODEL_NAME, ModelService
from server.schemas import GenerateRequest, GenerateResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = ModelService()
    service.load()
    app.state.model_service = service
    yield


app = FastAPI(title="Local LLM Inference Server", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    service: ModelService = request.app.state.model_service
    return {
        "status": "healthy" if service.is_loaded else "unhealthy",
        "model_loaded": service.is_loaded,
        "device": str(service.device),
        "model": MODEL_NAME,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, request: Request) -> dict[str, object]:
    service: ModelService = request.app.state.model_service
    try:
        return service.generate(body.prompt, body.max_new_tokens)
    except Exception as exc:
        # Do not include the submitted prompt in logs or error details.
        raise HTTPException(status_code=500, detail="Model generation failed") from exc


@app.get("/gpu")
def gpu(request: Request) -> dict[str, object]:
    service: ModelService = request.app.state.model_service
    return service.gpu_stats()
