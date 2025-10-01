# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps needed for some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for layer caching
COPY requirements.txt .

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

EXPOSE 8000

# run uvicorn pointing to the FastAPI app module
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
