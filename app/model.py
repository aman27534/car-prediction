from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib


CAR_PRICE_API_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = CAR_PRICE_API_DIR / "random_forest_model.pkl"
COLS_PATH = CAR_PRICE_API_DIR / "feature_columns.pkl"

_model = None
_feature_columns = None

def load_artifact():
    global _model, _feature_columns
    if _model is None: 
        _model = joblib.load(MODEL_PATH)
    if _feature_columns is None:
        _feature_columns = joblib.load(COLS_PATH)



def preprocess(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df = pd.get_dummies(df)
    if _feature_columns is not None:
        df = df.reindex(columns=_feature_columns, fill_value=0)
    return df

def predict_price(payload: dict) -> float:
    df = preprocess(payload)
    prediction = _model.predict(df)[0]
    return float(prediction)