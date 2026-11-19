FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv

WORKDIR /workspace
ENV PATH="/workspace/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/src" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY src ./src
COPY tests ./tests
