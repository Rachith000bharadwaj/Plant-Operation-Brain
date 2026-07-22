# Plant Operations Brain — single-container deployment.
# Build:  docker build -t plant-brain .
# Run:    docker run -p 8501:8501 --env-file .env plant-brain
# (For GPU acceleration use the nvidia runtime: --gpus all)

FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF and audio handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 espeak-ng && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
