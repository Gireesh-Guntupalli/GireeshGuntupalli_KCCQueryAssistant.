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
   git clone https://github.com/Gireesh-Guntupalli/GireeshGuntupalli_KCCQueryAssistant..git
   cd kcc-query-assistant
   ```

2. **Create Virtual Environment**
   ```bash
    conda create --name kcc python=3.10
    conda activate kcc
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
- `pandas`, `requests` - Utility & data processing
- `ollama` - Interface with local LLM (`gemma3:1b`)
- `serpapi` - Fallback for web search

The `requirements.txt` includes all needed packages.

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

## 🧪 Example Prompts

> 💬 “What pest-control methods are recommended for paddy in Tamil Nadu?”
> 💬 “How to manage drought stress in groundnut cultivation?”
> 💬 “What issues do sugarcane farmers in Maharashtra commonly face?”
> 💬 “My tomato plants are wilting even after watering regularly. What could be the problem?”
> 💬 “What are the recommended pest control methods for cotton in Maharashtra?”
> 💬 “How can I increase the yield of sugarcane using organic practices?”
> 💬 “Which is the best fertilizer for paddy during the flowering stage?”
> 💬 “What are common pests in sugarcane crops in Uttar Pradesh during monsoon?”
> 💬 “How does agriculatrual practices in Germany differ from that India?” # triggers web search
> 💬 “What are the government subsidies for installing solar-powered irrigation pumps in 2025?”
> 💬 “Which startup provides AI-based crop disease detection using drones?” # triggers web search

---

## 👨‍🌾 Prompt Customization (Used Internally), (Refer Modelfile)

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

These parameters are tuned to prioritize **accuracy and consistency** over creativity:

- `temperature = 0.3` – Lower creativity, more focused and factual responses
- `top_p = 0.7` – Restricts sampling to the most probable tokens, reducing randomness

---

## 🔒 API Keys

To use **SerpAPI**, export your API key:

```bash
export SERPAPI_API_KEY= "6fb5953aa0005416f5307922637a89b395a05e7208c7b66ce2171b30d3df4e80"
```

---

## 📬 Feedback & Contribution

Pull requests, issues, and feedback are welcome!

---

## 📜 License

MIT License 
