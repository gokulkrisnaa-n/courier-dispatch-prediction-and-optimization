# Multi-stage build.
#   - `serve`  : slim image for the FastAPI scoring service (requirements-serve.txt)
#   - `runner` : full image for the producer + dispatcher stream services
# Deps are installed before the source is copied so code changes don't bust the
# (expensive) dependency layer.

# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# --------------------------------------------------------------------------- #
FROM base AS serve
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt
COPY pyproject.toml ./
COPY config ./config
COPY src ./src
RUN pip install -e . --no-deps
EXPOSE 8000
# Model + reference profile are mounted at /app/artifacts by docker-compose.
CMD ["uvicorn", "dispatch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------------- #
FROM base AS runner
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY config ./config
COPY src ./src
RUN pip install -e . --no-deps
# Default command; docker-compose overrides per service (producer / dispatcher).
CMD ["python", "-m", "dispatch.stream.dispatcher"]
