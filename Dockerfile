FROM python:3.10-slim

WORKDIR /app

COPY api.py .
COPY infar_model.py .
COPY models .

RUN pip install fastapi uvicorn onnx onnxruntime

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0", "--port", "8000", "--reload"]
# --reload is used for development purposes, remove it in production

