## Description 
This API leverages semi-structured RAG to provide an LLM (ChatGPT-4) with context and attributable sources. 

Features include: 
- PDF document parsing using Unstructured
- Using semi-structured RAG to provide context for an LLM (gpt-4). Allows for processing of both tables and text
- Retrieval with LLM - LLM uses PDF context to generate answers to the user
- A chat based web interface with functionalities such as: querying the LLM, creating databases, adding files to databases for 
- User can view on the UI exactly which chunks are most relevant to their query

Services used: 
- AWS EC2 for cloud-hosting, nginx as reverse proxy, systemd
- AWS dynamodb for storage of database, documents, and chunks relationships
- Weaviate for vector database
- Redis for raw text/tables key-value store

## RAG WorkFlow
We use Unstructured to parse both text and tables. Unstructured uses: 
- tesseract for Optical Character Recognition (OCR)
- poppler for PDF rendering and processing

Unstructured.partition_pdf generates chunks of text that we categorize into text or table types. 

We use MultiVectorRetrieval: 
- Vector store using Weaviate that stores the embedded summaries
- Document store using Redis that stores raw text and tables

RAG pipeline: 
We provide a prompt template and the context to gpt-4, which generates accurate answers. 

## Limitations
- database is manually cleared for now, but this can be resolved in the future
- latency is an issue as the example template takes approximately 3 minutes to process


## Accessing the API

### Method 1: Public URL

**Important note**: Large pdf files may take a long time to process. For instance, the example file takes around 3 minutes to process. Chrome has a timeout of ~1.5 min which causes the page to be reloaded, please wait a little longer for changes to be reflected, the server should be working fine. 

The api docs are hosted at: http://18.118.140.126/docs

### Method 2: Development Set-up
**Note:** 
- To run the API, you will need to supply the keys as specified in /src/.env
- Please ensure you have Python3.10.X

To set up virtual env and install requirements:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
! brew install tesseract
! brew install poppler
source setup.sh
```

To start the server:
```
uvicorn src.base:app --host 0.0.0.0 --port 5000 // For dev
```

The webserver will be hosted on http://localhost:5000.