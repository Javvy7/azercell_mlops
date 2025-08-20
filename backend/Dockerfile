FROM python:3.12-slim

WORKDIR /app

COPY backend/ ./
COPY models/model.pkl ./models/model.pkl

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
