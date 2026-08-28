FROM python:3.11-slim

WORKDIR /app

# system deps needed by torchvision/Pillow for image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

# model artifacts needed at inference time (trained model + vocab)
COPY models/base_resnet_lstm/best_model.pt ./models/base_resnet_lstm/best_model.pt
COPY data/processed/vocab.json ./data/processed/vocab.json

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]