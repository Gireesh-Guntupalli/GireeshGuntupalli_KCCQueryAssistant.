# Module: Live Internet Search fallback
import requests


def perform_live_search(
    query: str,
    top_k=5,
    serpapi_api_key="6fb5953aa0005416f5307922637a89b395a05e7208c7b66ce2171b30d3df4e80",
):
    params = {"engine": "google", "q": query, "api_key": serpapi_api_key}
    serp_response = requests.get("https://serpapi.com/search", params=params)

    if serp_response.status_code != 200:
        print(f"Error fetching from SerpAPI: {serp_response.status_code}")
        return []

    organic_results = serp_response.json().get("organic_results", [])
    if not organic_results:
        return []

    snippets = [r.get("snippet", "") for r in organic_results[:top_k]]
    return snippets
