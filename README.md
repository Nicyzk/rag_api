## Description
This API leverages semi-structured RAG to provide an LLM (ChatGPT-4) with context and attributable sources. 

## RAG WorkFlow
We use Unstructured to parse both text and tables. Unstructured uses: 
- tesseract for Optical Character Recognition (OCR)
- poppler for PDF rendering and processing

Unstructured.partition_pdf generates chunks of text that we categorize into text or table types. 

We use MultiVectorRetrieval: 
- Vector store using Weaviate that stores the embedded summaries
- Document store using Redis that stores raw text and tables


## Development Set-up
AWS ec2 for cloud-hosting, nginx as reverse proxy, systemd
Dynamodb for storage of documents
Weaviate for vector database
Redis for raw text/tables database


## 3. Run the development server
Note: 
- To run the API, you will need to supply the keys as specified in /src/.env
- Please ensure you have Python3.10.X

Set up virtual env and install requirements:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
! brew install tesseract
! brew install poppler
source setup.sh
```

Start the server:
```
uvicorn src.base:app --host 0.0.0.0 --port 5000 // For dev
```

The webserver will be hosted on http://localhost:5001.

Features include: 
- PDF document parsing using Unstructured