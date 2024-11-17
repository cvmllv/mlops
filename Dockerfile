FROM python:3.9-slim

# Устанавливаем утилиту unzip для разархивирования
RUN apt-get update && apt-get install -y unzip && apt-get clean

# Копируем зависимости
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем файлы приложения
COPY server.py /app/server.py
COPY mlflow_utils.py /app/mlflow_utils.py
COPY random_forest_model.onnx.zip /app/random_forest_model.onnx.zip

# Устанавливаем рабочую директорию
WORKDIR /app

# Разархивируем модель
RUN unzip random_forest_model.onnx.zip && rm random_forest_model.onnx.zip

# Запуск FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]