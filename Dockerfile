FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create log directories
RUN mkdir -p logs/production logs/experiments/temp_sweep \
    logs/experiments/critic_ablation \
    logs/experiments/prompt_robustness \
    logs/experiments/calibration

# Default: run FastAPI
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
