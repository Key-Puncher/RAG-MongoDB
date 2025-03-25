# from langchain_mongodb import MongoDBAtlasVectorSearch
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from main import RAG
from dotenv import dotenv_values
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from gpt4all import GPT4All
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os

from utils import get_mongo_client

config = dotenv_values(".env")
OPENAI_KEY = config["OPENAI_KEY"]


class AgenticRAG:
    def __init__(self, collection=None, generator=None, agents=[]):
        pass

    def get_agent(self, client, agent_func=None):
        OPENAI_KEY = config["OPENAI_KEY"]
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY

        llm = OpenAI(model="gpt-4o", temperature=0)

        vector_store = MongoDBAtlasVectorSearch(
            client,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            vector_index_name="vector_index",
        )

        index = VectorStoreIndex.from_vector_store(vector_store)
        query_engine = index.as_query_engine(similarity_top_k=5)  # , llm=llm)

        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="knowledge_base",
                description=(
                    "Provides information about NRMA car insurance."
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )

        agent_worker = FunctionCallingAgentWorker.from_tools(
            [query_engine_tool], llm=llm, verbose=True
        )
        agent = agent_worker.as_agent()

        response = agent.chat(
            "Tell me the best NRMA insurance to get for a single mother with a young child."
        )
        print(str(response))

        return agent


if __name__ == "__main__":
    config = dotenv_values(".env")
    DATA_LOADED = True

    # Set up MongoDB connection
    MONGO_URI = config["MONGO_URI"]

    DB_NAME = "RAG_DEMO"
    COLLECTION_NAME = "NRMA_PDF"

    client = get_mongo_client(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    local_llm_path = "./mistral-7b-openorca.gguf2.Q4_0.gguf"
    local_llm = GPT4All(local_llm_path)

    rag = AgenticRAG(collection, local_llm)
    agent = rag.get_agent(client)
