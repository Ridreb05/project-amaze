FROM python:3.11-slim

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

# Create empty assets dir
RUN mkdir -p assets && chown appuser:appuser assets

USER appuser

EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "-m", "server.app"]
