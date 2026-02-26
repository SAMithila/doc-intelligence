FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install -e .

# Copy app files
COPY configs/ configs/
COPY eval_data/ eval_data/
COPY app/ app/

# Expose ports
EXPOSE 8000 8501

# Default command (FastAPI)
CMD ["uvicorn", "docint.api.main:app", "--host", "0.0.0.0", "--port", "8000"]