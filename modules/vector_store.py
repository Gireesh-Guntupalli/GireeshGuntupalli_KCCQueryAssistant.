# import chromadb
# from chromadb.config import Settings
# import chromadb.utils.embedding_functions as embedding_functions


# def initialize_chroma_collection(
#     collection_name="kcc_embeddings", persist_path="./chroma_store"
# ):
#     """
#     Initializes a ChromaDB persistent collection with embedding function.

#     Returns:
#         chroma_client, collection: ChromaDB client and initialized collection.
#     """
#     chroma_client = chromadb.PersistentClient(path=persist_path)

#     try:
#         chroma_client.delete_collection(collection_name)
#         print(f"⚠️ Existing collection '{collection_name}' deleted.")
#     except Exception:
#         print(f"No existing collection named '{collection_name}' found; creating new.")

#     embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )

#     collection = chroma_client.create_collection(
#         name=collection_name, embedding_function=embedding_fn
#     )

#     return chroma_client, collection


# def batch_add_to_chroma(collection, documents, metadatas, ids, batch_size=5000):
#     """
#     Adds documents in batches to a ChromaDB collection.
#     """
#     for i in range(0, len(documents), batch_size):
#         collection.add(
#             documents=documents[i : i + batch_size],
#             metadatas=metadatas[i : i + batch_size],
#             ids=ids[i : i + batch_size],
#         )
#         print(f"✅ Added batch {i} to {min(i + batch_size, len(documents))}")


# modules/vector_store.py

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions


def store_embeddings_in_chroma(
    sample_df,
    collection_name="kcc_embeddings",
    persist_path="./chroma_store",
    batch_size=5000,
):
    chroma_client = chromadb.PersistentClient(path=persist_path)

    try:
        chroma_client.delete_collection(collection_name)
        print(f"⚠️ Existing collection '{collection_name}' deleted.")
    except Exception:
        print(f"No existing collection named '{collection_name}' found; creating new.")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    documents = sample_df["text"].tolist()
    metadatas = sample_df["metadata"].tolist()
    ids = sample_df["doc_id"].tolist()

    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print(f"✅ Added batch {i} to {min(i + batch_size, len(documents))}")

    return chroma_client, collection
