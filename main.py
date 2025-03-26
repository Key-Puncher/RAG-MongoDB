from dotenv import dotenv_values

from gpt4all import GPT4All

from vector_index import setup_vector_search_index, INDEX_DEFINITION
from utils import get_mongo_client, ingest_data, clear_data, load_data_from_pdf
from search import get_embedding, create_embeddings
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import ollama

config = dotenv_values(".env")

# Set up MongoDB connection
MONGO_URI = config["MONGO_URI"]
DB_NAME = "RAG_DEMO"
COLLECTION_NAME = "NRMA_PDF"

client = get_mongo_client(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

DATA_LOADED = True
# if we haven't got data loaded: ingest data into mongodb'
if not DATA_LOADED:
    clear_data(collection)
    data_src = ["nrma-car-pds-spds007-1023-nsw-act-qld-tas.pdf", "POL011BA.pdf"]
    for src in data_src:
        texts = load_data_from_pdf(src)
        ingest_data(db, texts, COLLECTION_NAME)
    create_embeddings(collection)


if "vector_index" not in [
    idx["name"] for idx in list(collection.list_search_indexes())
]:
    setup_vector_search_index(db[COLLECTION_NAME], INDEX_DEFINITION)


class RAG:
    def __init__(self, collection=None, generator=None):
        """
        Initialize the RAG system with a retriever and a generator.

        :param retriever: Function or class for retrieving relevant documents.
        :param generator: Function or class for generating responses.
        """
        self.collection = collection
        self.generator = generator

    def retrieve_documents(self, query):
        """
        Retrieve relevant documents based on the query.

        :param query: User's input query.
        :return: List of retrieved documents.
        """
        query_embedding = get_embedding(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = collection.aggregate(pipeline)
        array_of_results = []
        for doc in results:
            array_of_results.append(doc)
        return array_of_results

    def generate_response(self, query, retrieved_docs):
        """
        Generate a response using retrieved documents.

        :param query: User's input query.
        :param retrieved_docs: Documents retrieved for context.
        :return: Generated response text.
        """
        text_documents = ""
        for doc in retrieved_docs:
            text = doc.get("text", "")
            string = f"Summary: {text}\n"
            text_documents += string

        prompt = f"""Use the following information from NRMA insurance to answer the question at the end.
            {text_documents}
            Question: {query}
            """
        # response = self.generator.generate(prompt)
        # cleaned_response = response.replace("\\n", "\n")
        response = ollama.chat(
            model="llama3.2", messages=[{"role": "user", "content": prompt}]
        )
        cleaned_response = response["message"]["content"]
        return cleaned_response

    def answer_query(self, query):
        """
        Complete RAG pipeline: Retrieve documents and generate a response.

        :param query: User's input query.
        :return: Generated response text.
        """
        retrieved_docs = self.retrieve_documents(query)
        response = self.generate_response(query, retrieved_docs)
        return response


if __name__ == "__main__":
    # Example usage
    # local_llm_path = "./mistral-7b-openorca.gguf2.Q4_0.gguf"
    # local_llm = GPT4All(local_llm_path)

    local_llm = Ollama(model="llama3.2", request_timeout=150.0, temperature=0.1)

    rag = RAG(collection, local_llm)
    print(
        rag.answer_query(
            "Tell me the best NRMA insurance to get for a single mother with a young child."
        )
    )

    print("\n\n")

# Close the MongoDB connection when done
client.close()
