import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from modules.vector_store import store_embeddings_in_chroma
from modules.data_preprocessing import preprocess_dataframe
from modules.data_transformation import create_qa_docs
from modules.embedding_generation import generate_embeddings
from modules.rag_pipeline import (
    answer_with_local_llm_rag,
)

st.set_page_config(page_title="KCC Query Assistant", layout="wide")

st.title("🌾 KCC Query Assistant")

# Header image
image_url = "https://analyticsindiamag.com/wp-content/uploads/2023/04/kissan-got.jpg"
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="{image_url}" width="400" />
        <p>🍃 The KisaanGPT your KCC Query Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# System Setup
@st.cache_resource
def setup_system():
    raw_df = pd.read_csv("data/kcc_cleaned_raw.csv")
    df_cleaned = preprocess_dataframe(raw_df)
    qa_df = create_qa_docs(df_cleaned)
    sample_df, _ = generate_embeddings(qa_df)
    _, collection = store_embeddings_in_chroma(sample_df)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model, collection


model, collection = setup_system()

# Input field
query = st.text_input("💬 Ask your agricultural question:")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response_placeholder = st.empty()
        streamed_answer = ""

        # Use the streaming generator
        for chunk, source in answer_with_local_llm_rag(query, model, collection):
            if source == "error":
                st.error(chunk)
                break
            streamed_answer += chunk
            response_placeholder.markdown("### Answer:\n" + streamed_answer)

        # Source tag
        if source == "local":
            st.success("✅ Answer from Historical KCC data")
        elif source == "serpapi":
            st.info("🌐 Answer from Internet search")
