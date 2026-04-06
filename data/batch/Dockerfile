FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY batch_pipeline.py .
CMD ["python", "batch_pipeline.py"]
