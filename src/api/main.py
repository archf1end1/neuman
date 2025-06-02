# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List
import shap

app = FastAPI()

# --- Define the input data model using Pydantic ---
class WaferData(BaseModel):
    data: List[List[float]]

# --- Load the trained XGBoost model ---
try:
    model_path = '../model/model_smote.pkl'  # Or '../model/model.pkl' if you prefer the non-SMOTE one
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# --- Define the /score endpoint ---
@app.post("/score")
async def predict_yield(wafer_data: WaferData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(wafer_data.data)

        # Ensure the input data has the same number of features as the model expects
        if input_df.shape[1] != model.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Expected {model.n_features_in_} features, got {input_df.shape[1]}")

        # Make predictions
        predictions = model.predict_proba(input_df)[:, 1].tolist() # Get probability of class 1 (fail)

        # For SHAP values (optional but good for MVP)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df).tolist()
            return {"predictions": predictions, "shap_values": shap_values}
        except Exception as shap_e:
            print(f"Error generating SHAP values: {shap_e}")
            return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)