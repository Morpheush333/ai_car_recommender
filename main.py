"""
Streamlit Demo: AI Car Recommender
Dependencies:
    pip install streamlit openai pandas faiss-cpu numpy dotenv os
Set your OpenAI API key:
    export OPENAI_API_KEY="your_key_here"
Run:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from openai import OpenAI

# ----------------- Initialize OpenAI ---------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ----------------- Load Car Dataset ---------------------
cars = pd.read_csv("cars.csv")

# Normalize text columns for consistent filtering
cars["type"] = cars["type"].str.lower()
cars["fuel"] = cars["fuel"].str.lower()
cars["model"] = cars["model"].str.lower()
cars["features"] = cars["features"].str.lower()

# ----------------- Embedding & FAISS Setup ---------------------
embedding_model = "text-embedding-3-small"
embed_dim = 1536

@st.cache_resource
def load_faiss_index():
    if os.path.exists("car_index.faiss") and os.path.exists("car_embeddings.npy"):
        index = faiss.read_index("car_index.faiss")
        embeddings = np.load("car_embeddings.npy")
        return index, embeddings

    # Build texts for all cars
    car_texts = [
        f"{row['model']}, Type: {row['type']}, Fuel: {row['fuel']}, "
        f"Price: {row['price']}, Features: {row['features']}"
        for _, row in cars.iterrows()
    ]

    # Batch request embeddings
    response = client.embeddings.create(
        model=embedding_model,
        input=car_texts
    )
    car_embeddings = np.array([e.embedding for e in response.data], dtype="float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embed_dim)
    index.add(car_embeddings)

    # Save to disk for reuse
    faiss.write_index(index, "car_index.faiss")
    np.save("car_embeddings.npy", car_embeddings)

    return index, car_embeddings

faiss_index, car_embeddings = load_faiss_index()

# ----------------- Streamlit UI ---------------------
st.set_page_config(page_title="AI Car Recommender", layout="wide")
st.title("🚗 Advanced AI Car Recommender")

st.markdown("""
Welcome! Enter your car preferences below, and the AI will recommend the best matches with explanations.
""")

# ---- User Input ----
with st.form("preferences_form"):
    budget = st.number_input("Maximum budget (USD)", min_value=0, step=1000, value=40000)
    car_type = st.selectbox("Car type", ["", "sedan", "suv", "truck"])
    fuel_type = st.selectbox("Fuel type", ["", "petrol", "hybrid", "electric"])
    features = st.text_input("Specific features (e.g., AWD, autopilot)").lower()
    brand = st.text_input("Preferred brand").lower()
    submitted = st.form_submit_button("Find Cars")

# ----------------- RAG ---------------------
def get_embedding(text: str):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding

def retrieve_cars(preferences, top_k=5):
    query_text = ", ".join(f"{k}: {v}" for k, v in preferences.items() if v)
    query_emb = get_embedding(query_text)
    D, I = faiss_index.search(np.array([query_emb], dtype="float32"), top_k)

    retrieved = cars.iloc[I[0]].copy()

    # Strict filtering with vectorized conditions
    mask = pd.Series(True, index=retrieved.index)
    if preferences.get("budget"):
        mask &= retrieved["price"] <= preferences["budget"]
    if preferences.get("type"):
        mask &= retrieved["type"].str.contains(preferences["type"])
    if preferences.get("fuel"):
        mask &= retrieved["fuel"].str.contains(preferences["fuel"])
    if preferences.get("brand"):
        mask &= retrieved["model"].str.contains(preferences["brand"])
    if preferences.get("features"):
        mask &= retrieved["features"].str.contains(preferences["features"])

    return retrieved[mask]

def generate_recommendation(preferences, retrieved_cars):
    if retrieved_cars.empty:
        return "Sorry, no cars match your preferences exactly. Try adjusting your budget or features."

    context_text = "\n".join(
        [f"{row['model']} | {row['type']} | {row['fuel']} | ${row['price']} | Features: {row['features']}"
         for _, row in retrieved_cars.iterrows()]
    )

    prompt = f"""
You are a helpful car sales assistant.
User preferences: {preferences}
Candidate cars:\n{context_text}

Provide a friendly recommendation with pros and cons for each car, explaining why it fits the user's needs.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful car sales assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ----------------- Display Results ---------------------
if submitted:
    user_prefs = {
        "budget": budget,
        "type": car_type.lower(),
        "fuel": fuel_type.lower(),
        "features": features,
        "brand": brand
    }
    retrieved_cars = retrieve_cars(user_prefs)

    st.subheader("Recommended Cars")
    if retrieved_cars.empty:
        st.warning("No cars matched your preferences.")
    else:
        cols = st.columns(len(retrieved_cars))
        for col, (_, car) in zip(cols, retrieved_cars.iterrows()):
            col.markdown(f"**{car['model'].title()}**")
            col.markdown(f"- Type: {car['type'].title()}")
            col.markdown(f"- Fuel: {car['fuel'].title()}")
            col.markdown(f"- Price: ${car['price']:,}")
            col.markdown(f"- Features: {car['features']}")

        st.subheader("AI Recommendations Explanation")
        explanation = generate_recommendation(user_prefs, retrieved_cars)
        st.markdown(explanation)
