# 🌾 KCC Query Assistant

**KCC Query Assistant** is an AI-powered tool built using Retrieval-Augmented Generation (RAG) to help Indian farmers get accurate and informative responses to agricultural queries. It leverages historical Kisan Call Center (KCC) data and can fall back to live web search when needed.

## 📌 Features

- ✅ Query Indian agricultural data using natural language
- 🧠 Uses local LLM via Ollama for offline response generation
- 🔍 Retrieval-Augmented Generation using ChromaDB vector store
- 🔄 Fallback to SerpAPI for web search when no relevant data is found
- 🖥️ Streamlit web app for easy interaction
- 🛠️ Data preprocessing, QA formatting, embedding, and persistent vector store setup

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/kcc-query-assistant.git
   cd kcc-query-assistant
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Dependencies

Here are the key libraries used:

- `streamlit` - UI for the app
- `sentence-transformers` - For generating text embeddings
- `chromadb` - Vector store for document retrieval
- `scipy`, `pandas`, `requests` - Utility & data processing
- `ollama` - Interface with local LLM (`gemma3:1b`)
- `serpapi` - Fallback for web search

Your `requirements.txt` includes all needed packages.

---

## 🚀 Quickstart: Launch the App

1. **Ensure Ollama is running**
   - Install [Ollama](https://ollama.com/)
   - Run the model:  
     ```bash
     ollama run gemma3:1b
     ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Interact via Web UI**
   - Ask your agricultural queries
   - Get answers from local dataset or SerpAPI

---

## 🧩 Project Structure

```bash
kcc-query-assistant/
│
├── app.py                          # Streamlit frontend
├── main.py                         # CLI utility to process & index data
├── requirements.txt                # Python dependencies
│
├── data/                           # Processed CSV/JSON files
├── modules/                        # Modularized project code
│   ├── data_preprocessing.py       # Cleaning raw data
│   ├── data_transformation.py      # Creating QA pairs
│   ├── embedding_generation.py     # Generate text embeddings
│   ├── vector_store.py             # ChromaDB collection handling
│   ├── rag_pipeline.py             # LLM querying and RAG logic
│   └── live_search.py              # SerpAPI integration (optional)
│
└── chroma_store/                   # Persistent vector store
```

---

## ⚙️ Offline & Online Answering Logic

- **If relevant context exists in KCC embeddings** → Answer using local LLM (`gemma3:1b`)
- **If no relevant context is found** → Fallback to web search via [SerpAPI](https://serpapi.com/)
- Real-time streaming of LLM responses using the Ollama `stream=True` API

---

## 🧪 Example Prompt

> 💬 “What are common pests in sugarcane crops in Uttar Pradesh during monsoon?”

---

## 👨‍🌾 Prompt Customization (Used Internally)

```text
You are the Kisan Call Center (KCC) Query Assistant, an expert AI specialized in agricultural advice for Indian farmers. 
You provide clear, accurate, and helpful responses about crops, pests, weather, farming techniques, and related queries based on the KCC dataset. 
Always use any provided context to ground your answers. If no context is available, answer based on your general knowledge in agriculture. 
Be polite, concise, and informative.
```

---

## 🧼 Preprocessing Details

- Lowercasing and trimming whitespace from important fields: `Crop`, `DistrictName`, `StateName`, `Sector`, etc.
- Conversion of QA pairs from dataset rows
- Embedding each QA pair with `all-MiniLM-L6-v2`
- Storing in persistent ChromaDB collection

---

## 🧠 Parameters

You can tune the LLM response style using:

- `temperature = 1` – Higher creativity
- `top_p = 0.8` – Nucleus sampling (diverse choices)

---

## 🔒 API Keys

To use **SerpAPI**, export your API key:

```bash
export SERPAPI_API_KEY="your_key_here"
```

---

## 📬 Feedback & Contribution

Pull requests, issues, and feedback are welcome!

---

## 📜 License

MIT License – Free to use and modify.

