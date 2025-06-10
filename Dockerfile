FROM  python 3.10-slim

WORKDIR /app

COPY api.py .
COPY infer_model.py .
COPY models .

RUN pip install fastapi uvicorn onnx onnxruntime


