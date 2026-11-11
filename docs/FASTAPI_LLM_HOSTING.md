# Hosting a local LLM with FastAPI

## Startup and request path

Uvicorn imports `server.main:app` and enters FastAPI's lifespan startup. The server creates one `ModelService`, reads tokenizer and model files from `E:\Hosted LLM\models\smollm2-135m` (or `LLM_MODEL_PATH`), and places the FP16 model on `cuda:0`. Loading is local-only, so startup does not download weights. The model stays in evaluation mode in GPU memory until the process shuts down.

`POST /generate` validates the prompt and token limit, then FastAPI runs the synchronous handler in its thread pool. A lock permits one generation at a time. The tokenizer applies the model's chat template, the model generates with `use_cache=True`, and the response decodes only tokens added after the input. CUDA synchronization brackets generation timing so asynchronous GPU work is included.

## Weights and KV cache

Model weights are persistent process state. During generation, the KV cache stores attention keys and values for the current prompt and generated sequence, avoiding recomputation of prior tokens. It is request-specific and becomes reclaimable when generation tensors go out of scope; PyTorch may keep freed blocks in its allocator's reserved pool for reuse. No per-request `empty_cache()` is needed.

## Monitoring and worker count

Use `nvidia-smi` in a terminal to inspect GPU utilization and memory while requests run. `/gpu` reports PyTorch's device total, allocated, and reserved memory; reserved memory can remain higher than allocated memory because of the caching allocator.

Start with one Uvicorn worker. Each worker is a separate process and would load another model copy, consuming additional VRAM. One worker also keeps this initial single-GPU serving path straightforward to observe.

Run from the project directory:

```powershell
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs`. With the server running, the smoke check is `python scripts-train/test-server.py`.
