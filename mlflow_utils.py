import mlflow
import mlflow.onnx
import onnxruntime
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import onnx

# Настройка MLflow
mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_experiment("RandomForestExperiment")

def log_model_and_metrics(X_test, y_test, predictions, model_path="random_forest_model.onnx"):
    # Загрузка модели для анализа графа
    try:
        model = onnx.load(model_path)  # Загружаем модель как объект
        print("Model structure loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model structure: {e}")
        raise e

    # Проверка наличия разных классов
    if len(np.unique(y_test)) < 2:
        print("Warning: Only one class present in y_true. Skipping ROC AUC calculation.")
        roc_auc = None
    else:
        roc_auc = roc_auc_score(y_test, predictions)

    # Генерация отчёта классификации
    class_report = classification_report(y_test, predictions, output_dict=True)

    # Логирование в MLflow
    with mlflow.start_run():
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        for label, metrics in class_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)

        # Логирование модели как объекта
        mlflow.onnx.log_model(
            onnx_model=model,  # Передаём объект модели
            artifact_path="model",
            registered_model_name="RandomForestONNX"
        )

    return {"roc_auc": roc_auc, "class_report": class_report}