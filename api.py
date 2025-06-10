from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from infar_model import predict
import numpy as np


app = FastAPI()
class InputData(BaseModel):
    features: List[float]
    
@app.post("/predict")
def make_prediction(input_data: InputData):
    input_array = np.array(input_data.features, dtype=np.float32).reshape(1, -1)
    
    prediction = predict(input_array)
    
    return {"prediction": int(prediction)}