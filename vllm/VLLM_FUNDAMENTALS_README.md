# vLLM Fundamentals — 10 Concepts to Remember

> **Learning lab:** SmolLM2-135M-Instruct · NVIDIA RTX 5050 (~8 GiB VRAM) · Ubuntu 24.04 in WSL2 · vLLM 0.33.0  
> **Goal:** Explain how a model is served, where GPU memory goes, how requests are scheduled, and how to measure performance.

## The mental model (memorize this)

```text
Client / Windows PowerShell
          |
          | HTTP (OpenAI-compatible API)
          v
     vLLM API server
          |
          v
    Engine + Scheduler ----> waiting / running requests
          |
          v
     GPU Model Runner
          |
          +---- Shared model weights (VRAM)
          +---- KV cache block pool (VRAM)
          +---- Temporary compute/workspace (VRAM)
          |
          v
     Generated tokens -> response
```

The **model** decides token probabilities. The **inference engine** decides *how to execute requests efficiently*.

## 1. Model weights

**Remember:** Weights are parameters learned during training. The serving process loads the model from disk into GPU memory once per model replica, then reuses those weights across requests.

- Disk: `E:\Hosted LLM\models\smollm2-135m` (WSL path: `/mnt/e/Hosted LLM/models/smollm2-135m`).
- GPU: RTX 5050; the standalone FP16 model used about **257 MiB of live model tensors** in our earlier PyTorch test.
- Ten users **do not mean ten copies** of weights in one engine instance. Multiple worker processes / replicas can load separate copies.

**Interview line:** “Weights are shared; per-request generation state is not.”

## 2. Prefill

**Remember:** Prefill processes the input prompt and constructs its initial KV cache, typically processing many prompt tokens in parallel.

```text
Prompt tokens -> embeddings -> Transformer layers -> prompt KV cache -> first next-token logits
```

- Longer prompts generally add prefill work.
- **TTFT** (time to first token) is especially affected by prompt processing **and queueing**.
- **Chunked prefill** can split large prefills into scheduled portions.

**Interview line:** “Prefill turns the prompt into model state and produces the first-token prediction.”

## 3. Decode

**Remember:** After prefill, the model generates output autoregressively, usually advancing one token position per active request per decode step.

```text
Previous token + cached K/V -> new logits -> next token -> repeat
```

- Decoding reuses old K/V rather than recomputing the entire history.
- At small batch sizes it is frequently **memory-bandwidth-bound**; the exact bottleneck depends on model/hardware/batch.
- Performance is tracked by **TPOT** (time per output token) and inter-token latency.

**Interview line:** “Prefill processes the prompt; decode extends the answer token by token.”

## 4. KV cache

**Remember:** K/V tensors are **runtime data**, not trained weights. They store attention keys and values of previous processed token positions separately for each Transformer layer and logical sequence.

For **our SmolLM2** (30 layers, 3 KV heads, head dimension 64, BF16/FP16 = 2 bytes):

```text
KV bytes = 2 (K and V) × layers × KV heads × head dim × tokens × bytes/value
         = 2 × 30 × 3 × 64 × tokens × 2
         = 23,040 bytes per cached token per sequence
```

| Cached token positions | Raw KV tensor size, one sequence |
|---:|---:|
| 100 | ~2.20 MiB |
| 1,000 | ~21.97 MiB |
| 2,048 | ~45 MiB |
| 8,192 | ~180 MiB |

- More cached tokens and more active sequences generally require more KV memory.
- A completed request's blocks can be reused; a **reserved KV pool may remain in VRAM**.
- **GQA** uses fewer KV heads (3) than query heads (9), reducing KV storage compared with a same-width design using 9 KV heads.

**Interview line:** “Weights are stable during inference; KV state grows with sequence length and active requests.”

## 5. Scheduler

**Remember:** The scheduler picks which waiting/running requests and how much token work can execute within the available token budget and KV capacity.

- Your `--max-num-seqs 4` limits the number of sequences simultaneously scheduled, **not** the number of HTTP clients that can connect.
- Requests may wait when active capacity is full.
- Memory pressure can lead to preemption; specific preemption/recompute behavior depends on engine/version/configuration.
- It is **not simply CPU round-robin**: scheduling is token- and KV-memory-aware.

**Interview line:** “The scheduler manages request admission and GPU token work without exceeding execution or cache budgets.”

## 6. Continuous batching

**Remember:** The active batch can change at each scheduling step. Finished sequences leave and new ones can join without waiting for the longest response.

```text
Step 1: [A B C]
Step 2: [A B C]
Step 3: [B C D]    # A finished, D joined
```

- Batches can mix **prefill work and decode work** depending on scheduler configuration.
- Improves **aggregate GPU utilization and throughput**, but does not guarantee lower latency for every individual request.
- Each request still maintains distinct logical KV state.

**Interview line:** “Continuous batching dynamically schedules active sequences instead of keeping a fixed batch together until all are done.”

## 7. PagedAttention / paged KV allocation

**Remember:** vLLM manages the KV cache in reusable **blocks**. A sequence's logical blocks can map to noncontiguous physical blocks.

```text
Sequence A logical: [A0] [A1] [A2]
GPU KV pool:       [B0] [A1] [free] [A0] [B1] [A2] ...
```

- Reduces waste from reserving large, contiguous per-request cache regions.
- Supports efficient reuse and block management as sequences grow/shrink.
- **Prefix caching:** identical eligible prefixes may reuse cached computations; reuse must respect correct cache identity/isolation.
- **Paging does not remove the cost of long contexts**: their K/V still consume memory.

**Interview line:** “Paged KV allocation improves memory flexibility and utilization; it doesn't change attention's mathematical purpose.”

## 8. Eager mode, CUDA Graphs, and compilation

**Remember:** These are **execution optimizations**, separate from batching and KV caching.

| Concept | What to remember |
|---|---|
| GPU kernel | A function/program executed on the GPU |
| Eager execution | Dispatches operations using the normal runtime path |
| Compilation | Can optimize/fuse operations and select better kernels |
| CUDA Graph | Captures GPU work for efficient replay, reducing repeated launch overhead |

- We launched with `--enforce-eager` to simplify compatibility/debugging.
- In our vLLM 0.33.0 logs, this **disabled `torch.compile`, CUDA Graphs, and JIT warmup**.
- **Eager does not disable KV caching or the scheduler.**
- Optimized mode can improve speed but may increase startup/warmup time and memory use.

**Interview line:** “Eager is a clean baseline; compiled kernels and CUDA Graph replay target execution overhead.”

## 9. Quantization

**Remember:** Quantization represents weights and sometimes activations/KV cache with fewer bits to reduce memory bandwidth/storage, potentially trading off quality or hardware compatibility.

Approximate raw storage per **1 billion weights**:

| Format | Raw weight size |
|---|---:|
| FP32 | ~4 GB |
| FP16 / BF16 | ~2 GB |
| INT8 | ~1 GB |
| 4-bit | ~0.5 GB |

- Real deployments also need scales/metadata, buffers and sometimes unquantized layers.
- **Weight quantization does not automatically quantize KV cache.**
- Lower precision isn't guaranteed to run faster: kernel and GPU support matter.
- For an 8 GB GPU, quantization can help fit larger models, but first validate quality and real VRAM usage.

**Interview line:** “Quantization trades representation precision for lower memory requirements; benchmark the actual kernel and quality.”

## 10. Performance metrics

**Remember:** Evaluate **both user latency and system throughput**, plus queue/cache pressure.

| Metric | Practical meaning |
|---|---|
| TTFT | Request arrival → first output token, includes queue + prefill |
| TPOT | Average interval per output token after first (definition depends on benchmark) |
| ITL | Individual time gap between streamed tokens |
| End-to-end latency | Request start → complete response |
| Output throughput | Output tokens/sec across requests |
| Prompt throughput | Input/prefill tokens/sec across requests |
| Running / waiting | Active scheduled work vs queued requests |
| KV cache usage | Fraction of KV block capacity currently occupied (not necessarily VRAM allocation) |
| GPU utilization | Compute activity; high % alone doesn't prove good user latency |

**Important:** vLLM's periodic log line showing `Avg generation throughput` is an **interval-wide engine metric**. It may be low when the server is idle; it is **not** automatically a per-request benchmark.

**Interview line:** “I track TTFT/TPOT and tail latency for user experience; throughput, queue size and KV usage for capacity planning.”

---

## Our actual vLLM run: how to interpret the logs

```text
vLLM: 0.33.0
Model: SmolLM2-135M-Instruct (resolved LlamaForCausalLM)
GPU: NVIDIA RTX 5050 (WSL2)
max_model_len: 2048
max_num_seqs: 4
GPU KV cache budget: 4.86 GiB
GPU KV cache capacity: 226,640 token positions
Estimated max 2048-token contexts from KV capacity: 110.66x
```

**Do not confuse capacity with concurrency.** `110.66x` is a *theoretical cache-memory estimate*, not 110 simultaneously scheduled requests. Our `--max-num-seqs 4` and actual throughput impose other limits. Cache capacity is distributed across the engine's pool; model, temporary allocations, and scheduler constraints still apply.

```text
API Server  ->  Engine Core/Scheduler  ->  GPU Model Runner
                                      ->  Shared weights + KV block pool
```

The server successfully exposed `/v1/chat/completions` and was callable from **Windows PowerShell** using `http://127.0.0.1:8000`.

### Commands worth remembering

Start Ubuntu (Windows PowerShell):

```powershell
wsl -d Ubuntu-24.04
```

Inside Ubuntu, activate the existing environment and start the server:

```bash
cd ~/llm-vllm-env
source .venv/bin/activate

vllm serve "/mnt/e/Hosted LLM/models/smollm2-135m" \
  --served-model-name smollm2 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 4 \
  --enforce-eager
```

Check health and models:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Test inference:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Explain attention simply."}],"max_tokens":100,"temperature":0}'
```

Inspect runtime metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | grep '^vllm:' | head -40
watch -n 1 nvidia-smi
```

Stop vLLM: `Ctrl+C` in the server terminal. To shut down all WSL2 distros from Windows PowerShell:

```powershell
wsl --shutdown
```

This also stops other WSL2 services (including Docker's WSL backend), so use it when appropriate.

## Five questions you should be able to answer without notes

1. **Weights vs KV cache?** Weights are shared trained parameters; KV holds request-specific, per-layer attention state.
2. **Prefill vs decode?** Prefill processes prompt tokens; decode generates additional tokens while reusing cached K/V.
3. **Why continuous batching?** Requests join/leave dynamically to improve aggregate GPU utilization.
4. **Why paged KV blocks?** Efficient assignment/reuse of cache storage, reducing contiguous-allocation waste.
5. **Why run eager first?** Easier compatibility/debugging baseline before testing compilation/CUDA Graph optimizations.

## What to learn next (hands-on order)

1. **Streaming + TTFT:** Compare 50-token vs 1,000-token prompts, measure first-token latency.
2. **Concurrent clients:** Send 1, 2, 4, then 8 requests; inspect running/waiting counts and throughput.
3. **KV pool and prefix caching:** Repeat prompts; compare cache hits and occupancy.
4. **Eager vs optimized:** Remove `--enforce-eager` *only after* stable baseline; benchmark startup, memory and generation.
5. **Quantization:** Try a suitable larger quantized model and check actual GPU fit/quality.

**One-sentence summary:** **vLLM keeps the model resident on the GPU, schedules token work across active requests, manages their KV cache in reusable blocks, and uses optimized GPU execution to improve serving efficiency.**
