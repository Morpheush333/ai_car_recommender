# AI Car Recommender 🚗

A **Streamlit-based AI car recommendation app** that suggests cars based on user preferences using **OpenAI embeddings** and **FAISS** for retrieval-augmented generation (RAG). The AI also provides friendly explanations with pros and cons for each recommended car.

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.30-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Features

- Input preferences like budget, car type, fuel type, brand, and special features.
- Uses **OpenAI embeddings** to match user queries with car inventory.
- Retrieves top matching cars using **FAISS vector search**.
- Provides **AI-generated recommendations** with pros and cons.
- Interactive Streamlit interface for real-time user input.

---

## How to run

Run the Streamlit app:

```bash
streamlit run main.py
```


## Usage

1. Enter your car preferences in the form.  
2. Click **Find Cars**.  
3. View recommended cars displayed side-by-side.  
4. Read AI-generated explanations for each recommendation.  

## How It Works

- **Embedding Cars:** Each car entry is converted into a vector using OpenAI embeddings.  
- **FAISS Index:** Vectors are stored in a FAISS index for fast similarity search.  
- **User Query:** User preferences are converted into embeddings and used to retrieve top matches.  
- **RAG & Explanation:** The AI generates friendly recommendations using context from the retrieved cars.  

## Dependencies

- `streamlit` – Web app framework  
- `openai` – OpenAI API client  
- `pandas` – Data manipulation  
- `numpy` – Numerical operations  
- `faiss-cpu` – Vector similarity search  
- `python-dotenv` – Load environment variables  