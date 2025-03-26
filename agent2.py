from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# from langchain_mongodb import MongoDBAtlasVectorSearch
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from main import RAG
from dotenv import dotenv_values
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from gpt4all import GPT4All
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os

from utils import get_mongo_client, load_data_from_pdf

config = dotenv_values(".env")
MONGO_URI = config["MONGO_URI"]

DB_NAME = "RAG_DEMO"
COLLECTION_NAME = "NRMA_PDF"

client = get_mongo_client(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

llm = Ollama(model="llama3.1:8b-instruct-q8_0", request_timeout=150.0, temperature=0.1)
embed_model = HuggingFaceEmbedding("mixedbread-ai/mxbai-embed-large-v1")

Settings.llm = llm
Settings.embed_model = embed_model
vector_store = MongoDBAtlasVectorSearch(
    client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    vector_index_name="vector_index",
)

# data_src = "nrma-car-pds-spds007-1023-nsw-act-qld-tas.pdf"
# texts = load_data_from_pdf(data_src)

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine_test = index.as_query_engine(similarity_top_k=5)

response = query_engine_test.retrieve("What is NRMA?")

for i in response:
    print(i)
