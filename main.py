from modules.data_preprocessing import preprocess_dataframe
from modules.data_transformation import (
    create_qa_docs,
    export_qa_docs,
    export_cleaned_raw,
)
from modules.embedding_generation import generate_embeddings
from modules.vector_store import store_embeddings_in_chroma
from modules.rag_pipeline import answer_with_local_llm_rag
from sentence_transformers import SentenceTransformer
import pandas as pd


def main():
    # Data loading and preprocessing
    print("🔄 Loading and preprocessing raw dataset...")
    raw_df = pd.read_csv(r"D:\Data Science Projects\datasets\Year_2024_dataset.csv")
    df_cleaned = preprocess_dataframe(raw_df)

    # Data transformation
    print("🔧 Transforming cleaned data into QA docs...")
    qa_df = create_qa_docs(df_cleaned)

    # Data export
    print("💾 Exporting data...")
    export_qa_docs(qa_df, "data/kcc_qa_clean.json", "data/kcc_qa_clean.csv")
    export_cleaned_raw(df_cleaned, "data/kcc_cleaned_raw.csv")

    # Embedding generation (still needed for other potential uses)
    print("🧠 Generating embeddings...")
    sample_df, _ = generate_embeddings(qa_df)

    # Vector store setup
    print("📦 Initializing vector store and adding embeddings...")
    chroma_client, collection = store_embeddings_in_chroma(sample_df)
    print("✅ Vector store ready.")

    # Load embedding model for RAG pipeline
    print("🔧 Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Interactive query loop
    print("\n🤖 Welcome to the KCC Query Assistant!")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query = input("💬 Enter your agricultural query:\n> ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting KCC Query Assistant. Goodbye!")
                break

            print("\n🤖 Assistant:")
            full_response = ""
            for response_chunk, source in answer_with_local_llm_rag(
                query, model, collection
            ):
                print(response_chunk, end="", flush=True)
                full_response += response_chunk

            print(f"\n\n[Source: {source}]")
            print("-" * 50)  # Visual separator

        except KeyboardInterrupt:
            print("\n🛑 Session interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
