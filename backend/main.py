from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = FastAPI()

# Allow all origins for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Load data
df = pd.read_csv("medicine_dataset.csv", low_memory=False)
df = df.fillna("")

# Prepare combined column (once)
df["combined"] = (
    df["name"].astype(str) + " " +
    df[["use0", "use1", "use2", "use3", "use4"]].astype(str).agg(" ".join, axis=1) + " " +
    df[[col for col in df.columns if col.startswith("sideEffect")]].astype(str).agg(" ".join, axis=1)
)

# Vectorize entire dataset once
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

def find_best_match(user_input):
    names = df["name"].tolist()
    matches = difflib.get_close_matches(user_input.lower(), names, n=1, cutoff=0.3)
    return names.index(matches[0]) if matches else None

@app.get("/recommend")
def recommend_medicine(medicine: str = Query(...)):
    names = df["name"].tolist()
    medicine_lower = medicine.lower().strip()

    # Try exact match first (case-insensitive)
    exact_matches = [i for i, name in enumerate(names) if name.lower() == medicine_lower]
    if exact_matches:
        idx = exact_matches[0]
        message = f"Exact match found for '{names[idx]}'."
    else:
        # Find closest matches (up to 3) with cutoff 0.3 or higher for caution
        close_matches = difflib.get_close_matches(medicine_lower, [n.lower() for n in names], n=3, cutoff=0.3)
        if close_matches:
            idx = names.index(next(name for name in names if name.lower() == close_matches[0]))
            message = (
                f"Exact medicine not found. Showing closest match: '{names[idx]}'. "
                "Please verify if this is the medicine you intended."
            )
        else:
            return {"error": "Medicine not found. Please check the name and try again."}

    input_name = df.loc[idx, "name"]

    # Full input medicine record (all relevant columns)
    cols = [
        "id", "name",
        "substitute0", "substitute1", "substitute2", "substitute3", "substitute4",
        *[f"sideEffect{i}" for i in range(42)],
        "use0", "use1", "use2", "use3", "use4",
        "Chemical Class", "Habit Forming", "Therapeutic Class", "Action Class"
    ]
    input_medicine = df.loc[idx, cols].to_dict()

    # Substitutes for input medicine (non-empty)
    substitutes = [s for s in df.loc[idx, ["substitute0", "substitute1", "substitute2", "substitute3", "substitute4"]] if s]

    # Side effects for input medicine (non-empty)
    side_effect_cols = [f"sideEffect{i}" for i in range(42)]
    side_effects = [se for se in df.loc[idx, side_effect_cols] if se]

    # Recommendations (similar medicines excluding the input medicine)
    user_vector = vectorizer.transform([df.loc[idx, "combined"]])
    cosine_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = cosine_scores.argsort()[-6:][::-1]  # top 6 including self
    top_indices = [i for i in top_indices if i != idx][:5]  # remove self and limit to 5

    recommendations = df.iloc[top_indices][cols].to_dict(orient="records")

    return {
        "message": message,
        "input_medicine": input_medicine,
        "substitutes": substitutes,
        "side_effects": side_effects,
        "recommendations": recommendations
    }






from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


from pathlib import Path

# ...
import os
# Assuming this is main.py inside /backend/
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "index.html")
