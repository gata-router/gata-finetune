# Copyright 2024 - 2026 Dave Hall, Skwashd Services https://gata.works, MIT License

FROM public.ecr.aws/chainguard/python:latest-dev AS builder

ENV LANG=C.UTF-8
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app:/app/venv/lib/python3.14/site-packages:/usr/lib/python3.14"
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT="/app/venv"
ENV UV_PYTHON_DOWNLOADS=0

COPY . /app

USER root

WORKDIR /app

RUN set -ex; \
    apk add --no-cache \
        libgomp \
        openblas-dev; \
    uv sync --no-dev --frozen --compile-bytecode --directory /app; \
    rm -rf /var/cache/apk/*;\
    ln -s /app/train.py /app/train; \
    echo "model_name = 'bert-base-uncased'; from transformers import AutoModelForSequenceClassification, AutoTokenizer; model = AutoModelForSequenceClassification.from_pretrained(model_name); tokenizer = AutoTokenizer.from_pretrained(model_name);" | python3


FROM public.ecr.aws/chainguard/wolfi-base:latest AS packages
RUN set -ex; \
    apk add --no-cache \
        libgomp \
        openblas; \
    rm -rf /var/cache/apk/*


FROM public.ecr.aws/chainguard/python:latest

LABEL org.opencontainers.image.source=https://github.com/gata-router/gata-finetune
LABEL org.opencontainers.image.description="Gata router ticket model fine tuning task."
LABEL org.opencontainers.image.licenses=MIT

ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="/app:/app/venv/lib/python3.14/site-packages:/usr/lib/python3.14"
ENV PYTHONUNBUFFERED=1
ENV SM_MODEL_DIR="/opt/ml/model"
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1 
ENV UV_NO_SYNC=1

USER root

COPY --from=builder /app /app
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY --from=packages /usr/lib/libgomp.so.* /usr/lib/libopenblas* /usr/lib/
COPY --from=packages /var/lib/db/sbom/libgomp-*.spdx.json /var/lib/db/sbom/openblas-*.spdx.json /var/lib/db/sbom/

WORKDIR /app

ENTRYPOINT ["python3"]
CMD ["train"]
