from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import onnxruntime
import numpy as np
from mlflow_utils import log_model_and_metrics

app = FastAPI()

# Определяем глобальную ONNX-сессию
session = onnxruntime.InferenceSession("random_forest_model.onnx")

class FeaturesInput(BaseModel):
    features: List[List[float]]
    y_true: List[int]

@app.post("/predict-and-log")
def predict_and_log(input_data: FeaturesInput):
    try:
        # Преобразуем входные данные
        features = np.array(input_data.features, dtype=np.float32)
        y_true = input_data.y_true

        print("Features:", features)
        print("True labels:", y_true)

        # Выполняем предсказания
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: features})[0]
        predictions = np.round(predictions)

        print("Predictions:", predictions)

        # Логируем результаты в MLflow
        result = log_model_and_metrics(features, y_true, predictions, model_path="random_forest_model.onnx")

        return {"status": "success", "predictions": predictions.tolist(), "roc_auc": result.get("roc_auc"), "class_report": result.get("class_report")}
    
    except Exception as e:
        print("Error occurred:", str(e))
        return {"status": "error", "message": str(e)}