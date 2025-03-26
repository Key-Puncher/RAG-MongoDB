import uuid
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def get_mongo_client(mongo_uri):
    """Establish and validate connection to MongoDB."""

    client = MongoClient(mongo_uri)

    # Validate the connection
    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        # Connection successful
        print("Connection to MongoDB successful")
        return client
    print("Connection to MongoDB failed")
    return None


def load_data_from_pdf(src):
    """Set up document loading and splitting"""

    loader = PyPDFLoader(src)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


def ingest_data(db, corpus=None, corpus_collection_name=""):
    """Ingest data into MongoDB collections."""

    if corpus and corpus_collection_name:
        corpus_docs = [
            {
                "_id": str(uuid.uuid4()),
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in corpus
        ]
        db[corpus_collection_name].insert_many(corpus_docs)
        print(f"Ingested {len(corpus_docs)} documents into {corpus_collection_name}")


def clear_data(collection):
    """Delete all data from a collection."""

    collection.delete_many({})
