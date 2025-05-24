# Module: Retrieval Augmented Generation pipeline & LLM integration
from scipy.spatial.distance import cosine

from modules.llm_interface import query_local_llm
from modules.live_search import perform_live_search


# def answer_with_local_llm_rag(query, model, collection, threshold=0.6, top_k=5):
#     # Embed query
#     query_embedding = model.encode([query])[0]

#     # Search ChromaDB
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         include=["documents", "metadatas", "embeddings"],
#     )

#     docs = results["documents"][0]
#     embeddings = results["embeddings"][0]

#     similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]
#     relevant_chunks = [doc for doc, sim in zip(docs, similarities) if sim >= threshold]

#     if relevant_chunks:
#         print(f"\n✅ Found {len(relevant_chunks)} relevant chunks from DB.")
#         context = "\n\n".join(relevant_chunks[:top_k])
#         full_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

# Context:
# {context}

# Question:
# {query}

# Answer:"""
#     else:
#         print("\n⚠️ No relevant local chunks found.")
#         print("🌐 Performing a live Internet search...")

#         snippets = perform_live_search(query, top_k)
#         if not snippets:
#             print("No search results found.")
#             return None

#         context = "\n\n".join(snippets)
#         full_prompt = f"""You are a helpful assistant. Use the following search snippets to answer the user's question.

# Search Snippets:
# {context}

# Question:
# {query}

# Answer:"""

#     print("\n🧠 Sending prompt to local LLM...")
#     answer = query_local_llm(full_prompt)
#     print("\n🤖 LLM Response:\n")
#     print(answer)
#     return answer


def answer_with_local_llm_rag(query, model, collection, threshold=0.6, top_k=5):
    import requests
    from scipy.spatial.distance import cosine
    import json

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    embeddings = results["embeddings"][0]

    similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]
    relevant_chunks = [doc for doc, sim in zip(docs, similarities) if sim >= threshold]

    if relevant_chunks:
        context = "\n\n".join(relevant_chunks[:top_k])
        full_prompt = f"""You are a helpful agricultural assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
        source = "local"
    else:
        serpapi_api_key = (
            "6fb5953aa0005416f5307922637a89b395a05e7208c7b66ce2171b30d3df4e80"
        )
        params = {"engine": "google", "q": query, "api_key": serpapi_api_key}
        serp_response = requests.get("https://serpapi.com/search", params=params)

        if serp_response.status_code == 200:
            organic_results = serp_response.json().get("organic_results", [])
            snippets = [r.get("snippet", "") for r in organic_results[:top_k]]
            context = "\n\n".join(snippets)
            full_prompt = f"""You are a helpful assistant. Use the following search results to answer the question.

Search Results:
{context}

Question:
{query}

Answer:"""
            source = "serpapi"
        else:
            return "Error contacting SerpAPI", "error"

    # Query LLM via Ollama
    payload = {
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": full_prompt}],
    }
    try:
        response = requests.post(
            "http://localhost:11434/api/chat", json=payload, stream=True
        )
        answer = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    answer += data["message"]["content"]
        return answer, source
    except Exception as e:
        return f"LLM Error: {e}", "error"
