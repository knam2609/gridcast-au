# syntax=docker/dockerfile:1
# The GridCast-AU API (docs/plans/m5_plan.md §2.6). Build and start it with `make api-up`,
# which passes the commit and whether the files copied here differ from it.
FROM python:3.12-slim

# uv as on the host; libgomp1 for LightGBM's OpenMP.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
# The locked dependencies first (a cached layer until uv.lock changes), then the project,
# installed editable so gridcast.config.PROJECT_ROOT is /app, as the checkout is on the host.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY configs ./configs
COPY src ./src
RUN uv sync --frozen --no-dev && python -m compileall -q src

# Non-root. /data is the data root: only <data root>/models is mounted, read-only, by
# docker-compose.models.yml.
RUN useradd --uid 10001 --no-log-init --create-home --home-dir /home/api api && mkdir /data
USER api

ARG GIT_SHA=""
ARG GIT_DIRTY=""
LABEL org.opencontainers.image.revision=$GIT_SHA
ENV PATH=/app/.venv/bin:$PATH \
    GRIDCAST_DATA_ROOT=/data \
    GRIDCAST_LOG_FORMAT=json \
    GRIDCAST_BUILD_GIT_SHA=$GIT_SHA \
    GRIDCAST_BUILD_DIRTY=$GIT_DIRTY

EXPOSE 8000
# One worker: the model state takes about 1.1 GB per process, and model requests are serialized.
CMD ["uvicorn", "gridcast.api.app:serve", "--factory", "--host", "0.0.0.0", "--port", "8000"]
