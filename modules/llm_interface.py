# Module: Local LLM API interface
import requests
import json


def query_local_llm(
    prompt: str, model_name="kcc", ollama_url="http://localhost:11434/api/chat"
) -> str:
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}

    response = requests.post(ollama_url, json=payload, stream=True)

    if response.status_code != 200:
        print(f"❌ Error querying local LLM: {response.status_code}")
        return ""

    output = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    output += data["message"]["content"]
            except json.JSONDecodeError:
                continue

    return output
