# TB Resistance Predictor — Hugging Face Space (Docker SDK)
FROM python:3.11-slim

# build tools + zlib in case VCF parsing (pysam/htslib) needs to compile;
# harmless if unused.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code + package + slim models + reports (cv_metrics.csv) + web frontend
COPY . .

ENV MODELS_DIR=models \
    REPORTS_DIR=reports \
    PYTHONUNBUFFERED=1

# HF Spaces expects the app on 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
