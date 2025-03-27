from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer

NUM_DOC_LIMIT = 250  # the number of documents to process and generate embeddings.

config = dotenv_values(".env")

# Set up MongoDB connection
MONGO_URI = config["MONGO_URI"]
DB_NAME = "RAG_DEMO"
COLLECTION_NAME = "NRMA_PDF"
MODEL_PATH = "mixedbread-ai/mxbai-embed-large-v1"

DATA_SRC = [
    "datasets/nrma-car-pds-spds007-1023-nsw-act-qld-tas.pdf",
    "datasets/POL011BA.pdf",
]


def get_embedding(text):
    # model_path = "local_model/"  # downloaded from SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    # model = SentenceTransformer(model_path)
    model = SentenceTransformer(MODEL_PATH)
    return model.encode(text).tolist()


def load_data_from_pdf(src):
    """Set up document loading and splitting"""

    loader = PyPDFLoader(src)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts
