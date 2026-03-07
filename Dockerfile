FROM python:3.12-slim

WORKDIR /app

# Copy requirement files first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies (We use --no-cache-dir to keep image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn (binds to Render's dynamic $PORT or 8000 locally)
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
