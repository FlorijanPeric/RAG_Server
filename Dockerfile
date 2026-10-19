FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Models normally run in the `models` container. LOCAL_MODELS=true adds PyTorch +
# sentence-transformers so EMBEDDING_MODE=local / LLM_MODE=local work in here too.
ARG LOCAL_MODELS=false
RUN if [ "$LOCAL_MODELS" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-local.txt; \
    fi

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
