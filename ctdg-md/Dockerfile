FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install .
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "ctdg_md.api:app", "--host", "0.0.0.0", "--port", "8000"]
