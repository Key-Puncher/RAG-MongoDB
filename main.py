from dotenv import dotenv_values


from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from gpt4all import GPT4All
from llama_index.llms.openai import OpenAI

from vector_index import setup_vector_search_index, INDEX_DEFINITION
from utils import get_mongo_client, ingest_data, clear_data, load_data_from_pdf

config = dotenv_values(".env")
OPENAI_KEY = config["OPENAI_KEY"]
DATA_LOADED = True

# Set up MongoDB connection
MONGO_URI = config["MONGO_URI"]
DB_NAME = "RAG_DEMO"
COLLECTION_NAME = "NRMA_PDF"
NUM_DOC_LIMIT = 250  # the number of documents to process and generate embeddings.

client = get_mongo_client(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# if we haven't got data loaded: ingest data into mongodb'
if not DATA_LOADED:
    data_src = "nrma-car-pds-spds007-1023-nsw-act-qld-tas.pdf"
    texts = load_data_from_pdf(data_src)
    ingest_data(db, texts, COLLECTION_NAME)

model_path = "local_model/"  # downloaded from SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
model = SentenceTransformer(model_path)


# Define function to generate embeddings
def get_embedding(text):
    return model.encode(text).tolist()


def create_embeddings(collection):
    # Filters for only documents with a summary field and without an embeddings field
    # Creates embeddings for documents without them
    filter = {
        "$and": [
            {"text": {"$exists": True, "$nin": [None, ""]}},
            {"embeddings": {"$exists": False}},
        ]
    }
    updated_doc_count = 0
    for document in collection.find(filter).limit(NUM_DOC_LIMIT):
        text = document["text"]
        embedding = get_embedding(text)
        collection.update_one(
            {"_id": document["_id"]}, {"$set": {"embeddings": embedding}}, upsert=True
        )
        updated_doc_count += 1
    print("Documents updated: {}".format(updated_doc_count))


create_embeddings(collection)

setup_vector_search_index(db[COLLECTION_NAME], INDEX_DEFINITION)

# Set up RAG pipeline


local_llm_path = "./mistral-7b-openorca.gguf2.Q4_0.gguf"
local_llm = GPT4All(local_llm_path)


def get_query_results(query):
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
                "_id": 0,
                "summary": 1,
                "listing_url": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = collection.aggregate(pipeline)
    array_of_results = []
    for doc in results:
        array_of_results.append(doc)
    return array_of_results


query_tool = FunctionTool.from_defaults(name="get_nrma_info", fn=get_query_results)

llm = OpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_tool], llm=llm, verbose=True
)
agent = AgentRunner(agent_worker)


if __name__ == "__main__":
    # Example usage
    question = "Can you recommend an NRMA insurance plan for a single mother with a young child?"
    documents = get_query_results(question)
    text_documents = ""
    for doc in documents:
        text = doc.get("text", "")
        string = f"Summary: {text}\n"
        text_documents += string
    prompt = f"""Use the following information from NRMA insurance to answer the question at the end.
        {text_documents}
        Question: {question}
        """
    response = local_llm.generate(prompt)
    cleaned_response = response.replace("\\n", "\n")
    print(cleaned_response)

    print("\n\n")

    response = agent.query(
        "Can you recommend an NRMA insurance plan for a single mother with a young child?"
    )
    print(response)

# Close the MongoDB connection when done
client.close()
