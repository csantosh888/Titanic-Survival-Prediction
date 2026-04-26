import joblib
import pandas as pd

from config import MODEL_PATH
from features import create_features


def predict_survival(input_data):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])
    df = create_features(df)

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability