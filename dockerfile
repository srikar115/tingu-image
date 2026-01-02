FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir runpod diffusers transformers accelerate

COPY handler.py .

CMD ["python3", "handler.py"]
