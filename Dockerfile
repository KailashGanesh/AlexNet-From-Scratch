FROM python:3.12-slim
RUN apt-get update && apt-get install -y curl zlib1g-dev --no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/models && \
    curl -L -o /app/models/best_model.pth "https://drive.google.com/uc?export=download&id=1bTpFE5yqvxwHuq5jfegSoTp9dP4lmB7j"
COPY . .
EXPOSE 3000
CMD ["python", "server.py"]