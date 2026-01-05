ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.05-py3-igpu
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt \
    && rm -rf /usr/local/lib/python3.10/dist-packages/cv2/typing

COPY inference_service.py /app/inference_service.py

EXPOSE 8001

CMD ["python3", "inference_service.py"]
