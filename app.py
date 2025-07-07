from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import StringIO
import joblib

app = FastAPI()

# Load your entire pipeline (feature engineering + preprocessing + model)
pipeline = joblib.load("catboost_model.pkl")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Normalize column names (lowercase, strip spaces)
    df.columns = [col.strip().lower() for col in df.columns]

    if "sk_id_curr" not in df.columns:
        raise HTTPException(
            status_code=400, detail="CSV must contain 'sk_id_curr' column"
        )

    sk_ids = df["sk_id_curr"].copy()

    try:
        # Predict probabilities directly using the pipeline
        probs = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    target = (probs >= 0.5).astype(int)

    for sk_id, t in zip(sk_ids, target):
        print(f"ID: {sk_id}, Target: {t}")

    results = [
        {"sk_id_curr": int(sk_id), "target": int(t)} for sk_id, t in zip(sk_ids, target)
    ]
    return {"predictions": results}
