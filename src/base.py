# Standard library imports
import json
import logging
import os
import uuid
import shutil
import pickle as pkl
import time
from typing import Dict, List, Tuple, Any
import asyncio
import uvicorn
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

# Third-party library imports
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Body, HTTPException, APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.retrievers import (
    MultiVectorRetriever,
)
from langchain.vectorstores.weaviate import Weaviate
from langchain_cohere import ChatCohere
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from weaviate import Client
from weaviate.auth import AuthApiKey

from src.models import QueryModel
from botocore.exceptions import ClientError
from uuid import uuid4
import boto3
from boto3.dynamodb.conditions import Key
from sec_edgar_downloader import Downloader

# Our libraries
from src.redis_connect import VF_Redis

# set_debug(True)
# set_verbose(True)

DUMMY_UUID = "00000000-0000-0000-0000-000000000000"

# Set up environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

print("OpenAI Key: " + str(OPENAI_API_KEY))
print("Weaviate Url: " + str(WEAVIATE_URL))
print("Weaviate API Key: " + str(WEAVIATE_API_KEY))

# Set up the FastAPI app
app = FastAPI()
upload_router = APIRouter(prefix="/upload")
database_router = APIRouter(prefix="/database")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Set up logging
logging.basicConfig(level=logging.INFO)

BaseModel.model_config["arbitrary_types_allowed"] = True

# Set up the Weaviate client
auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
)
vectorstore = Weaviate(
    client=client,
    embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    index_name="Text_tables",
    text_key="page_content",
    attributes=["doc_id", "owner_uuid", "chunk_id", "data_source"],
    by_text=False,
)
ID_KEY = "chunk_id"
store = VF_Redis(redis_url=os.environ["REDIS_URL"])
# store = InMemoryStore()
multi_vec_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=ID_KEY,
    k=5,
)


# set up in chunk similarity models
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and performs well for similarity tasks
ST_model = SentenceTransformer(model_name).cpu()

# AWS DynamoDB
session = boto3.Session(
    aws_access_key_id=os.environ["AWS_SECRET_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
    region_name=os.environ["AWS_DEFAULT_REGION"]
)
dynamodb = session.resource('dynamodb', region_name=os.environ["AWS_DEFAULT_REGION"])
table_data_sources_id_name = dynamodb.Table("data_source_id_name")
table_doc_id_data_source_id = dynamodb.Table("doc_id_data_source_id")


class Element(BaseModel):
    type: str
    text: Any


@upload_router.post("/file_upload/")
async def upload_and_process_pdf(
    file: UploadFile = File(...), data_source_name: str = Form(...)
) -> JSONResponse:
    print(WEAVIATE_API_KEY, WEAVIATE_URL)
    logging.info("Starting PDF upload and process")
    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    file_name = os.path.basename(file.filename)
    with open(tmp_path, "wb") as buffer:
        buffer.write(await file.read())
    logging.info("PDF uploaded to temporary path: %s", tmp_path)
    logging.info("File name: %s", file_name)

    if file is not None:
        _, categorized_elements, category_counts = process_pdf_file(tmp_path)

    logging.info("File processing complete")

    table_summaries, tables, text = consolidate_elements(categorized_elements)
    logging.info("Consolidation of elements complete")

    upload_to_stores(
        table_summaries, tables, text, DUMMY_UUID, data_source_name, file_name
    )
    logging.info("Uploaded summaries to stores")

    if file is not None:
        os.remove(tmp_path)
        logging.info("Temporary file %s removed", tmp_path)
    else:
        pass

    return JSONResponse(
        content={
            "message": "Processing and upload successful",
            "category_counts": category_counts,
        }
    )


def process_pdf_file(path: str) -> Tuple[List[any], List[Element], Dict[str, int]]:
    logging.info("Partitioning PDF file...")
    raw_pdf_elements = partition_pdf(
        filename=path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1000,
        new_after_n_chars=500,
        combine_text_under_n_chars=500,
        image_output_dir_path=os.path.dirname(path),
    )

    category_counts = {}
    for element in raw_pdf_elements:
        category = str(type(element))
        category_counts[category] = category_counts.get(category, 0) + 1

    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    return raw_pdf_elements, categorized_elements, category_counts

def consolidate_elements(
    categorized_elements: List[Element],
) -> Tuple[List[str], List[str], List[str]]:
    prompt_text = """You are an assistant tasked with summarizing tables. \
    Give a concise summary of the table. Table:\n{element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatCohere(temperature=0, model="command-r")
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    tables = [i.text for i in categorized_elements if i.type == "table"]
    texts = [i.text for i in categorized_elements if i.type == "text"]

    logging.info("Summarizing tables...")
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 10})

    return table_summaries, tables, texts


def upload_to_stores(
    table_summaries: List[str],
    tables,
    texts: List[str],
    caller_uuid: str,
    data_source_name: str,
    file_name: str,
):
    # Define constant strings
    TABLE_SUMMARY_SOURCE = "upload-pdf-table-summary"
    TEXT2VEC_OPENAI = "text2vec-openai"

    # Assume texts consists of excerpts from one document
    doc_id = str(uuid.uuid4())

    # add to dynmo
    create_document(doc_id, data_source_name)

    # Upload text chunks to Weaviate
    logging.info("Uploading text chunks to Weaviate...")
    chunk_ids = [str(uuid.uuid4()) for _ in texts]
    text_chunks = [
        Document(
            page_content=s,
            metadata={
                "doc_id": doc_id,
                ID_KEY: chunk_ids[i],
                "owner_uuid": caller_uuid,
                "data_source": data_source_name,
                "vectorizer": TEXT2VEC_OPENAI,
                "file_name": file_name,
            },
        )
        for i, s in enumerate(texts)
    ]
    logging.info("Uploading texts summaries to Weaviate...")
    multi_vec_retriever.vectorstore.add_documents(text_chunks)
    logging.info("Adding texts to Docstore...")
    multi_vec_retriever.docstore.mset(list(zip(chunk_ids, text_chunks)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(
            page_content=s,
            metadata={
                "doc_id": doc_id,
                ID_KEY: table_ids[i],
                "owner_uuid": caller_uuid,
                "data_source": data_source_name,
                "vectorizer": TEXT2VEC_OPENAI,
            },
        )
        for i, s in enumerate(table_summaries)
    ]
    logging.info("Uploading table summaries to Weaviate...")
    multi_vec_retriever.vectorstore.add_documents(summary_tables)
    logging.info("Adding tables to Docstore...")
    multi_vec_retriever.docstore.mset(list(zip(table_ids, tables)))


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        # If doc has 'page_content' attribute, use it directly.
        if hasattr(doc, "page_content"):
            formatted_docs.append(doc.page_content)
        # If doc is a bytes object, decode it to a string.
        elif isinstance(doc, bytes):
            try:
                decoded_content = doc.decode("utf-8")  # Adjust encoding if necessary
                formatted_docs.append(decoded_content)
            except UnicodeDecodeError:
                # Handle decoding error, if necessary
                logging.warning("Failed to decode bytes content.")
        elif isinstance(doc, str):
            # just naive append
            formatted_docs.append(doc)
        else:
            # Handle other unexpected types
            logging.warning("Unexpected type encountered: %s", type(doc))
    return "\n\n".join(formatted_docs)


def find_split_point(s, start, size):
    # Attempt to find a punctuation or space before the size limit
    end = start + size
    if end > len(s):
        return len(
            s
        )  # Return the end of the string if the calculated end exceeds string length

    # Move backwards from the end point to find a space or punctuation
    while end > start and not s[end - 1].isspace() and s[end - 1] not in ".!?,;":
        end -= 1

    # If no suitable break point is found, just use the maximum chunk size
    if end == start:
        end = start + size if start + size < len(s) else len(s)

    return end


def convert_chunks_readable(chunk, query, model):

    is_serialized = False
    try:
        chunk = json.loads(chunk)
        serial = chunk
        chunk = chunk["chunk"]
        is_serialized = True
        print(f"CHUNK: {type(chunk)} -- {str(chunk)}")
        print(f"SERIAL: {type(serial)} -- {str(serial)}")
    except json.JSONDecodeError as _:
        print("failed to decode chunk -- continuing as raw string")

    # Remove apostrophes from the chunk for comparison
    chunk_no_apostrophes = chunk.replace("'", "")

    # Adjusting the length and overlap based on the query
    sub_chunk_size = min(int(1.5 * len(query)), len(chunk))
    overlap = sub_chunk_size // 2

    return serial if is_serialized else chunk


def parrellel_chunk_process(chunks, query, model):
    # Assuming `chunks` is a list of chunks and `query` is defined
    processed_chunks = []

    print("3. (BEFORE) PROCESSED CHUNKS ARE OF TYPE: " + str(type(chunks[0])))

    for chunk in chunks:
        # Process each chunk sequentially in a for loop
        processed_chunk = convert_chunks_readable(chunk, query, model)
        processed_chunks.append(processed_chunk)

    print("4. (AFTER) PROCESSED CHUNKS ARE OF TYPE : " + str(type(processed_chunks[0])))
    return processed_chunks


def process_retrieved_chunks(context):
    retrieved_chunks = []
    for chunk in context:
        # Check if chunk has 'page_content' attribute
        if hasattr(chunk, "page_content"):
            print("2. SERIALIZED AS A JSON")
            generated_json = json.dumps(
                {"chunk": chunk.page_content, "metadata": chunk.metadata}
            )
            retrieved_chunks.append(generated_json)
        # If chunk is a bytes object, decode it to a string
        elif isinstance(chunk, bytes):
            try:
                decoded_content = chunk.decode("utf-8")
                retrieved_chunks.append(decoded_content)
            except UnicodeDecodeError:
                # Handle decoding error, if necessary
                logging.warning("Failed to decode bytes content.")
        elif isinstance(chunk, str):
            print("2. NOT SERIALIZED, PASSED AS A STR")
            retrieved_chunks.append(chunk)
        else:
            # Log a warning for unexpected types
            logging.warning("Unexpected type encountered: %s", type(chunk))
    return retrieved_chunks


async def generate_chunks(
    query, rag_chain_with_source, start_time,
):
    context_serialized = False
    for chunk in rag_chain_with_source.stream(query):
        if "context" in chunk and not context_serialized:
            context_documents = chunk["context"][: min(50, len(chunk["context"]))]
            print("1. CONTEXT DOCUMENTS RAW FORM " + str(context_documents))
            context_chunks = process_retrieved_chunks(context_documents)

            context_chunks = parrellel_chunk_process(
                context_chunks, query, ST_model
            )

            for context_chunk in context_chunks:
                if not isinstance(context_chunk, dict):
                    print("Not a dict, just a str.")
                    yield json.dumps(
                        {
                            "context": json.dumps(
                                {"chunk": context_chunk, "metadata": {}}
                            )
                        }
                    ) + "\n"
                elif isinstance(context_chunk, dict) and isinstance(
                    context_chunk["chunk"], dict
                ):
                    yield json.dumps(
                        {
                            "context": json.dumps(
                                {
                                    "chunk": context_chunk["chunk"]["chunk"],
                                    "metadata": context_chunk["chunk"]["metadata"],
                                }
                            )
                        }
                    ) + "\n"
                else:
                    yield json.dumps({"context": json.dumps(context_chunk)}) + "\n"
                await asyncio.sleep(0.0001)
            context_serialized = True

        if "answer" in chunk:
            answer_text = chunk["answer"]
            yield json.dumps({"answer": answer_text}) + "\n"
            await asyncio.sleep(0.0001)

    end_time = time.time()

    print(f"Query time: {end_time - start_time} seconds")


@app.post("/query")
async def query_chunks_and_generate(
    query: str, params: QueryModel = Body(...)
) -> StreamingResponse:
    print(params)
    datasources = params.datasources
    most_similar_section = params.most_similar_section
    filters = params.filters
    start_time = time.time()

    retriever = multi_vec_retriever

    # Reset filter
    retriever.search_kwargs["where_filter"] = None

    source_filter = {
        "operator": "Or",
        "operands": [
            {"path": "data_source", "operator": "Equal", "valueString": data_source}
            for data_source in datasources
        ],
    }

    retriever.search_kwargs["where_filter"] = source_filter


    template = """Answer the question based only on the following context, which can include text and tables
    This is the retrieved context from internal data, use this to answer the question
    {{context}}
    Question: {{question}}
    """
    template = template.format()
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4")

    cohere_model = ChatCohere(model="command-r", temperature=0)

    # RAG pipeline
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | cohere_model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": multi_vec_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return StreamingResponse(
        generate_chunks(query, rag_chain_with_source, start_time,),
        media_type="application/x-ndjson",
    )


@database_router.get("/data_source_file_names/", response_model=dict[str, str])
async def data_source_file_names(data_source_name: str) -> dict[str, str]:

    mapping = get_data_source_mapping()

    doc_ids = []

    try:
        response = table_doc_id_data_source_id.query(
            KeyConditionExpression="data_source_id = :ds_id",
            ExpressionAttributeValues={":ds_id": mapping[data_source_name]},
        )
        # Collect all doc_ids from the query response
        doc_ids = [item["doc_id"] for item in response.get("Items", [])]

        # Handle potential paginated results
        while "LastEvaluatedKey" in response:
            response = table_doc_id_data_source_id.query(
                KeyConditionExpression="data_source_id = :ds_id",
                ExpressionAttributeValues={":ds_id": mapping[data_source_name]},
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            doc_ids.extend([item["doc_id"] for item in response.get("Items", [])])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error while returning file list. ",
        )

    final_dict = {}
    try:
        for doc_id in doc_ids:
            response = (
                client.query.get("Text_tables", ["file_name"])
                .with_where(
                    {
                        "path": ["doc_id"],
                        "operator": "Equal",
                        "valueText": doc_id,
                    }
                )
                .do()
            )
            final_dict[doc_id] = response["data"]["Get"]["Text_tables"][0]["file_name"]

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error while polling Weaviate.")

    return final_dict


@database_router.get("/get_data_source/")
async def get_data_source(data_source_name: str):
    mapping = get_data_source_mapping()
    data_source_id = mapping[data_source_name]
    response = table_data_sources_id_name.query(
        KeyConditionExpression=Key("data_source_id").eq(data_source_id),
        Limit=1,
    )
    if not response["Items"]:
        raise HTTPException(
            status_code=404,
            detail='The data source "' + data_source_name + '" does not exist.',
        )
    return response["Items"][0]

@database_router.post("/create_data_source/")
async def create_data_source(data_source_name: str, description: str = None):
    try:
        item_id = str(uuid4())

        response = table_data_sources_id_name.scan(
            FilterExpression=Key("data_source_name").eq(data_source_name),
            Limit=1,
        )
        if response["Items"]:  # There already exists a data source with this name.
            raise HTTPException(
                status_code=403,
                detail='The data source "' + data_source_name + '" already exists.',
            )
        else:
            creation_date = str(datetime.now(tz=timezone.utc))
            table_data_sources_id_name.put_item(
                Item={
                    "data_source_id": item_id,
                    "data_source_name": data_source_name,
                    "created_on": creation_date,
                    "last_updated": creation_date,
                    "allowlist": {},
                    "description": description,
                }
            )
        return {
            "id": item_id,
            "data_source_name": data_source_name,
            "response": response,
        }
    except ClientError as e:
        raise HTTPException(status_code=400, detail=str(e))


@database_router.get("/get_data_sources/")
async def get_data_sources():
    try:
        mapping = get_data_source_mapping()
        response = table_data_sources_id_name.scan()
        items = response.get("Items", [])
        contents = [item["data_source_name"] for item in items]
        counts = []
        created_on = []
        desc = []
        for ds_name in contents:
            response = table_doc_id_data_source_id.query(
                KeyConditionExpression="data_source_id = :ds_id",
                ExpressionAttributeValues={":ds_id": mapping[ds_name]},
                Select="COUNT",
            )
            counts.append(response["Count"])
            response = table_data_sources_id_name.query(
                KeyConditionExpression="data_source_id = :ds_id",
                ExpressionAttributeValues={":ds_id": mapping[ds_name]},
                Select="SPECIFIC_ATTRIBUTES",
                ProjectionExpression="created_on, description",
            )
            created_on.append(response["Items"][0]["created_on"])
            try:
                desc.append(response["Items"][0]["description"])
            except KeyError as _:
                desc.append("")

        return {
            "data_source_names": contents,
            "counts": counts,
            "created_on": created_on,
            "description": desc,
        }
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_document(doc_id: str, data_source_name: str):
    mapping = get_data_source_mapping()
    response = table_doc_id_data_source_id.put_item(
        Item={"doc_id": doc_id, "data_source_id": mapping[data_source_name]}
    )
    response = table_data_sources_id_name.update_item(
        Key={
            "data_source_id": mapping[data_source_name],
            "data_source_name": data_source_name,
        },
        UpdateExpression="SET #attr = :val",
        ExpressionAttributeNames={"#attr": "last_updated"},
        ExpressionAttributeValues={":val": str(datetime.now(tz=timezone.utc))},
        ReturnValues="UPDATED_NEW",
    )

    return response


# NOTE - This replaces the old value with the new value and does not append
@database_router.post("/update_data_source/")
async def update_data_source(data_source_name: str, column_name: str, new_value: str):
    mapping = get_data_source_mapping()
    data_source_id = mapping[data_source_name]

    response = table_data_sources_id_name.update_item(
        Key={
            "data_source_id": data_source_id,
            "data_source_name": data_source_name,
        },
        UpdateExpression="SET #attr = :val",
        ExpressionAttributeNames={"#attr": column_name},
        ExpressionAttributeValues={":val": new_value},
        ReturnValues="UPDATED_NEW",
    )

    return response


def get_data_source_mapping():
    data_source_mapping = {}
    try:
        response = table_data_sources_id_name.scan()
        items = response.get("Items", [])
        for item in items:
            data_source_name = item.get("data_source_name")
            data_source_id = item.get("data_source_id")
            data_source_mapping[data_source_name] = data_source_id
    except ClientError as e:
        print(f"An error occurred: {e}")
        return None

    return data_source_mapping


def delete_document_ddb(data_source_name: str, doc_id: str):
    # Delete document from table_doc_id_data_source_id
    mapping = get_data_source_mapping()
    response = table_doc_id_data_source_id.delete_item(
        Key={"doc_id": doc_id, "data_source_id": mapping[data_source_name]}
    )
    return response


def delete_vectors_by_metadata_attribute(attribute_name, attribute_value):
    class_name = "Text_tables"
    # Query to find all object IDs where the metadata attribute has the specified value
    query = f"""
    {{
        Get {{
            {class_name}(
                where: {{
                    path: ["{attribute_name}"]
                    operator: Equal
                    valueString: "{attribute_value}"
                }}
            ) {{
                uuid
            }}
        }}
    }}
    """

    result = client.query.raw(query)
    object_ids = [obj["uuid"] for obj in result["data"]["Get"][class_name]]

    # Delete each object by ID
    for object_id in object_ids:
        client.data_object.delete(object_id, class_name)

    logging.info("Deleted %d objects from class '%s'.", len(object_ids), class_name)


app.include_router(upload_router)
app.include_router(database_router)


if __name__ == "__main__":
    uvicorn.run("base:app", host="127.0.0.1", port=int(5001), reload=False)
