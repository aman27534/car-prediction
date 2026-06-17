from fastapi import FastAPI, HTTPException
from app.schema import CarFeature, PredictionResponse
from app.model import load_artifact, predict_price

app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    load_artifact()

@app.get('/')
def startup():
    return {"message": "Car Price Prediction API is running"}

@app.post('/predict', response_model=PredictionResponse)
def predict(car_feature: CarFeature):
    try:
        prediction = predict_price(car_feature.dict())
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))