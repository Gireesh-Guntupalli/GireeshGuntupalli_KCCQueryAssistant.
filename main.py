# from modules.data_preprocessing import preprocess_dataframe
# from modules.data_transformation import (
#     create_qa_docs,
#     export_qa_docs,
#     export_cleaned_raw,
# )
# from modules.embedding_generation import generate_embeddings
# from modules.vector_store import (
#     initialize_chroma_collection,
#     batch_add_to_chroma,
# )
# from modules.rag_pipeline import answer_with_local_llm_rag

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer


# def main():
#     print("Loading and preprocessing raw dataset...")
#     raw_df = pd.read_csv(r"D:\Data Science Projects\datasets\Year_2024_dataset.csv")
#     df_cleaned = preprocess_dataframe(raw_df)

#     print("Transforming cleaned data into QA docs...")
#     qa_df = create_qa_docs(df_cleaned)

#     print("Exporting data...")
#     export_qa_docs(qa_df, "data/kcc_qa_clean.json", "data/kcc_qa_clean.csv")
#     export_cleaned_raw(df_cleaned, "data/kcc_cleaned_raw.csv")

#     print("\nGenerating embeddings...")
#     sample_df, embeddings = generate_embeddings(qa_df)

#     print("\nInitializing vector store and adding embeddings...")
#     chroma_client, collection = initialize_chroma_collection()

#     batch_add_to_chroma(
#         collection,
#         documents=sample_df["text"].tolist(),
#         metadatas=sample_df["metadata"].tolist(),
#         ids=sample_df["doc_id"].tolist(),
#     )

#     # Load embedding model for querying
#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Example: Run RAG pipeline with a sample query
#     query = "What crops are recommended for rainy season?"
#     print(f"\nUser query: {query}")
#     answer_with_local_llm_rag(query, model, collection)


# if __name__ == "__main__":
#     main()


from modules.data_preprocessing import preprocess_dataframe
from modules.data_transformation import (
    create_qa_docs,
    export_qa_docs,
    export_cleaned_raw,
)
from modules.embedding_generation import generate_embeddings
from modules.vector_store import (
    initialize_chroma_collection,
    batch_add_to_chroma,
)
from modules.rag_pipeline import answer_with_local_llm_rag

import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    print("🔄 Loading and preprocessing raw dataset...")
    raw_df = pd.read_csv(r"D:\Data Science Projects\datasets\Year_2024_dataset.csv")
    df_cleaned = preprocess_dataframe(raw_df)

    print("🔧 Transforming cleaned data into QA docs...")
    qa_df = create_qa_docs(df_cleaned)

    print("💾 Exporting data...")
    export_qa_docs(qa_df, "data/kcc_qa_clean.json", "data/kcc_qa_clean.csv")
    export_cleaned_raw(df_cleaned, "data/kcc_cleaned_raw.csv")

    print("🧠 Generating embeddings...")
    sample_df, embeddings = generate_embeddings(qa_df)

    print("📦 Initializing vector store and adding embeddings...")
    chroma_client, collection = initialize_chroma_collection()

    batch_add_to_chroma(
        collection,
        documents=sample_df["text"].tolist(),
        metadatas=sample_df["metadata"].tolist(),
        ids=sample_df["doc_id"].tolist(),
    )

    print("✅ Vector store ready.")

    # Load model once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 🔁 Query loop
    print("\n🤖 Welcome to the KCC Query Assistant!")
    while True:
        query = input(
            "\n💬 Enter your agricultural query (or type 'exit' to quit):\n> "
        )
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting KCC Query Assistant. Goodbye!")
            break

        answer_with_local_llm_rag(query, model, collection)


if __name__ == "__main__":
    main()
