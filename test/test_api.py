
import requests
import time

URL = "http://127.0.0.1:8000/generate"

payload = {
    "prompt": "Explain how transformers work in large language models in detail.",
    "max_new_tokens": 1000
}

start = time.perf_counter()

response = requests.post(
    URL,
    json=payload,
    timeout=180
)

response.raise_for_status()
data = response.json()

print("\n=== GENERATED TEXT ===")
print(data["response"])

print("\n=== PERFORMANCE ===")
print("Input tokens:", data["input_tokens"])
print("Output tokens:", data["output_tokens"])
print("Inference time:", data["inference_time_seconds"], "seconds")
print("Tokens/second:", round(data["tokens_per_second"], 2))
print("Total API time:", round(time.perf_counter() - start, 2), "seconds")
