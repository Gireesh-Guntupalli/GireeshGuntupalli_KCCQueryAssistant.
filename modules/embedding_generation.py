# Module: Embedding generation & vector store ingestion
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def generate_embeddings(
    qa_df: pd.DataFrame, sample_size: int = 20000, model_name: str = "all-MiniLM-L6-v2"
):
    """
    Generate embeddings for sampled QA pairs from the dataframe.

    Args:
        qa_df (pd.DataFrame): DataFrame containing 'query' and 'answer' columns.
        sample_size (int): Number of samples to take for embedding.
        model_name (str): SentenceTransformer model name.

    Returns:
        sample_df (pd.DataFrame): Sampled DataFrame with combined text column.
        embeddings (np.ndarray): Generated embeddings as numpy array.
    """
    # Sample the dataframe
    sample_df = qa_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Combine query and answer text
    sample_df["text"] = sample_df["query"] + " " + sample_df["answer"]

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(sample_df["text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings)

    print(f"✅ Embeddings shape: {embeddings.shape}")

    return sample_df, embeddings
