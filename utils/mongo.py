import time
from pymongo.operations import SearchIndexModel
import uuid
from pymongo import MongoClient


from gpt4all import GPT4All

from utils.common import NUM_DOC_LIMIT, load_data_from_pdf, get_embedding
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import ollama


INDEX_DEFINITION = {
    "fields": [
        {
            "numDimensions": 1024,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector",
        },
        # {"type": "filter", "path": "metadata"},
        # {"type": "filter", "path": "text"},
    ]
}


def create_embeddings(collection):
    # Filters for only documents with a summary field and without an embeddings field
    # Creates embeddings for documents without them
    filter = {
        "$and": [
            {"text": {"$exists": True, "$nin": [None, ""]}},
            {"embedding": {"$exists": False}},
        ]
    }
    updated_doc_count = 0
    for document in collection.find(filter).limit(NUM_DOC_LIMIT):
        text = document["text"]
        embedding = get_embedding(text)
        collection.update_one(
            {"_id": document["_id"]}, {"$set": {"embedding": embedding}}, upsert=True
        )
        updated_doc_count += 1
    print("Documents updated: {}".format(updated_doc_count))


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


def ingest_data(collection, corpus=None):
    """Ingest data into MongoDB collections."""

    if corpus:
        corpus_docs = [
            {
                "_id": str(uuid.uuid4()),
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in corpus
        ]
        collection.insert_many(corpus_docs)
        print(f"Ingested {len(corpus_docs)} documents into {collection}")


def clear_data(collection):
    """Delete all data from a collection."""

    collection.delete_many({})


def setup_vector_search_index(collection, index_definition, index_name="vector_index"):
    """
    Setup a vector search index for a MongoDB collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index")
    """
    search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
        type="vectorSearch",
    )
    result = collection.create_search_index(model=search_index_model)
    print("New search index named " + result + " is building.")
    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate = None
    if predicate is None:
        predicate = lambda index: index.get("queryable") is True  # noqa: E731
    while True:
        indices = list(collection.list_search_indexes(result))
        if len(indices) and predicate(indices[0]):
            break
        time.sleep(5)
    print(result + " is ready for querying.")


def load_data_from_src(collection, data_src):
    clear_data(collection)

    for src in data_src:
        texts = load_data_from_pdf(src)
        ingest_data(collection, texts)
    create_embeddings(collection)

    if "vector_index" not in [
        idx["name"] for idx in list(collection.list_search_indexes())
    ]:
        setup_vector_search_index(collection, INDEX_DEFINITION)
