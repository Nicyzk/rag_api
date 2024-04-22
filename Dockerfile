# Use a specific tag for a more controlled environment
FROM python:3.10-slim

# By default, listen on port 5000
EXPOSE 5000/tcp

WORKDIR /var/task

# Define the FUNCTION_DIR variable as /var/task
ENV FUNCTION_DIR="/var/task/"
ENV PYTHONPATH=${FUNCTION_DIR}:$PYTHONPATH

# Combine COPY instructions to reduce layers
COPY requirements.txt ${FUNCTION_DIR}
COPY src/ ${FUNCTION_DIR}src/

# Install dependencies and clean up in a single layer
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && rm -rf /root/.cache 

ENV OPENAI_API_KEY=sk-proj-fcs5dEkmdLGrzG30a5A6T3BlbkFJkCWedk7pTxUgHcE6Prga \
    WEAVIATE_URL=https://infinitus-62ojfc40.weaviate.network \
    WEAVIATE_API_KEY=LfkKf5Xk5vjGWLTdL9CSNNgfB17M78APiiPe \
    REDIS_URL=redis://default:U8dceI32PJobdGvgbJvrNQuJsjywaEO2@redis-18197.c30833.us-east-1-mz.ec2.cloud.rlrcp.com:18197 \
    AWS_SECRET_ID=AKIA3FLDXIK76QSXCNRL \
    AWS_SECRET_KEY=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 \
    AWS_DEFAULT_REGION=us-east-2

CMD ["python3", "-m", "uvicorn", "src.base:app", "--host", "0.0.0.0", "--port", "5001"]