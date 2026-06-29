# Reproducible environment for the full pipeline (CLIP inference, Grad-CAM,
# MLflow tracking). The dashboard runs in the same image for local dev;
# Vercel deployment uses its own Python runtime per vercel.json and does
# NOT use this Dockerfile (Vercel builds from requirements.txt directly).
#
# Build: docker build -t clever-hans-clip .
# Run pipeline:  docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs clever-hans-clip python scripts/run_pipeline.py
# Run dashboard: docker run --rm -p 8050:8050 -v $(pwd)/outputs:/app/outputs clever-hans-clip python src/dashboard/app.py
# Run tests:     docker run --rm clever-hans-clip pytest tests/unit tests/integration -v

FROM python:3.10-slim

# OpenCV and some torch deps need these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["python", "src/dashboard/app.py"]
