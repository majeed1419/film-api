from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
scaler = joblib.load('scaler.joblib')
dbscan = joblib.load('dbscan_model.joblib')
app = FastAPI()
@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"


# Define the request body structure
class Data(BaseModel):
    Year: int
    Metascore: float
    Duration_in_minutes: float
    Score: float

@app.post('/predict')
def predict(data: Data):
    # Convert the input data to a DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Extract features
    features = ['Year', 'Metascore', 'Duration_in_minutes', 'Score']
    X = df[features]
    
    # Standardize the features
    X_scaled = scaler.transform(X)
    
    # Predict the cluster
    cluster_label = dbscan.fit_predict(X_scaled)
    
    # Return the cluster label
    return {"Cluster": int(cluster_label[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
