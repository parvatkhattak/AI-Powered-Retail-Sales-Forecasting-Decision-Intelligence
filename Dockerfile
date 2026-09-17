# ─────────────────────────────────────────────────────────────────────────────
# Retail AI — Sales Forecasting & Decision Intelligence
# One-command run:
#   docker build -t retail-ai .
#   docker run -p 8501:8501 --env-file .env retail-ai
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# System deps (gcc needed for some Python C-extensions like LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer-cached) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Streamlit configuration ───────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit && printf '\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
base = "dark"\n\
' > /root/.streamlit/config.toml

# ── Build the database (idempotent — skips if retail.db already exists) ───────
# The pipeline reads train.csv and store.csv from data/ and writes data/retail.db.
# If you volume-mount a pre-built retail.db this step becomes a no-op.
RUN python -c "\
import os, pathlib; \
db = pathlib.Path('data/retail.db'); \
train = pathlib.Path('data/train.csv'); \
if not db.exists() and train.exists(): \
    import src.data_pipeline; \
    src.data_pipeline.run_pipeline() \
"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
