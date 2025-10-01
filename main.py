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
# 1. Load environment variables from .env (if exists)
load_dotenv()

# 2. Get API key from env
api_key = os.getenv("OPENAI_API_KEY")

# 3. Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ----------------- Sample Car Dataset ---------------------
# Replace with real inventory later
cars = pd.read_csv("cars.csv")

# ----------------- Embed Cars for RAG ---------------------
embedding_model = "text-embedding-3-small"
embed_dim = 1536
car_embeddings = []


def get_embedding(text: str, model=embedding_model):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


for _, row in cars.iterrows():
    text = f"{row['model']}, Type: {row['type']}, Fuel: {row['fuel']}, Price: {row['price']}, Features: {row['features']}"
    emb = get_embedding(text)
    car_embeddings.append(np.array(emb, dtype="float32"))

car_embeddings = np.vstack(car_embeddings)
faiss_index = faiss.IndexFlatL2(embed_dim)
faiss_index.add(car_embeddings)

# ----------------- Streamlit App ---------------------
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
    features = st.text_input("Specific features (e.g., AWD, autopilot)")
    brand = st.text_input("Preferred brand")
    submitted = st.form_submit_button("Find Cars")


# ----------------- RAG ---------------------
def retrieve_cars(preferences, top_k=5):
    query_text = ", ".join(f"{k}: {v}" for k, v in preferences.items() if v)
    query_emb = get_embedding(query_text, model=embedding_model)
    D, I = faiss_index.search(np.array([query_emb], dtype="float32"), top_k)

    retrieved = cars.iloc[I[0]].copy()

    # strict filtering
    def match(car):
        if preferences.get("budget") and car["price"] > preferences["budget"]:
            return False
        if preferences.get("type") and preferences["type"] not in car["type"].lower():
            return False
        if preferences.get("fuel") and preferences["fuel"] not in car["fuel"].lower():
            return False
        if preferences.get("brand") and preferences["brand"] not in car["model"].lower():
            return False
        if preferences.get("features") and preferences["features"] not in car["features"].lower():
            return False
        return True

    return retrieved[retrieved.apply(match, axis=1)]


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


# ----------------- Display Recommendations ---------------------
if submitted:
    user_prefs = {
        "budget": budget,
        "type": car_type,
        "fuel": fuel_type,
        "features": features.lower(),
        "brand": brand.lower()
    }
    retrieved_cars = retrieve_cars(user_prefs)

    st.subheader("Recommended Cars")
    if retrieved_cars.empty:
        st.warning("No cars matched your preferences.")
    else:
        # Side-by-side display
        cols = st.columns(len(retrieved_cars))
        for col, (_, car) in zip(cols, retrieved_cars.iterrows()):
            col.markdown(f"**{car['model']}**")
            col.markdown(f"- Type: {car['type']}")
            col.markdown(f"- Fuel: {car['fuel']}")
            col.markdown(f"- Price: ${car['price']}")
            col.markdown(f"- Features: {car['features']}")

        # LLM explanation
        st.subheader("AI Recommendations Explanation")
        explanation = generate_recommendation(user_prefs, retrieved_cars)
        st.markdown(explanation)
