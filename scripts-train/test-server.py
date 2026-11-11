"""Smoke check a running local inference server (no model initialization here)."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"


def get_json(path: str) -> dict:
    with urlopen(BASE_URL + path, timeout=60) as response:
        return json.loads(response.read())


def post_json(path: str, payload: dict) -> dict:
    request = Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    health_before = get_json("/health")
    assert health_before["model_loaded"] is True
    first = post_json("/generate", {"prompt": "Explain artificial intelligence", "max_new_tokens": 32})
    gpu = get_json("/gpu")
    second = post_json("/generate", {"prompt": "What is a neural network?", "max_new_tokens": 32})
    health_after = get_json("/health")

    assert first["response"].strip() and first["output_tokens"] > 0
    assert second["response"].strip() and second["output_tokens"] > 0
    assert health_after["model_loaded"] is True
    assert health_before["model"] == health_after["model"]
    assert gpu["total_vram_mb"] > 0
    print("Server smoke check passed: two generations, model remains loaded.")
    print("GPU:", gpu)


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        raise SystemExit(f"API returned HTTP {exc.code}: {exc.read().decode()}") from exc
