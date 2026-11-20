import os

import httpx
from langchain_ollama import ChatOllama


def test_local_inference() -> None:
    url = os.environ["OLLAMA_URL"]
    model = os.environ["OLLAMA_MODEL"]
    with httpx.Client(base_url=url) as client:
        response = client.get("/api/tags")
        response.raise_for_status()
        if not any(item["name"] == model for item in response.json()["models"]):
            print(f"Downloading {model} into Ollama.")
            response = client.post("/api/pull", json={"model": model, "stream": False}, timeout=1800)
            response.raise_for_status()
            assert response.json()["status"] == "success", response.text

    response = ChatOllama(
        base_url=url, model=model, reasoning=False, temperature=0,
        num_predict=64, client_kwargs={"timeout": 120},
    ).invoke("Say hello in one short sentence.").content
    assert isinstance(response, str) and response.strip(), "Ollama returned no text"
    print(f"Ollama replied: {response}")
