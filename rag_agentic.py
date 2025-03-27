from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from utils.common import MONGO_URI, DB_NAME, COLLECTION_NAME, MODEL_PATH
from utils.mongo import get_mongo_client
from evaluator import Evaluator


class AgenticRAG:
    def __init__(self, client, agents=[]):
        """
        Initialize the RAG system with a database connection and agents to use.

        :param client: A connection to a MongoDB instance.
        :param agents: A list of agent tools to provide to the LLM.
        """
        self.client = client
        llm = Ollama(model="llama3.2", request_timeout=150.0, temperature=0)
        embed_model = HuggingFaceEmbedding(MODEL_PATH)

        Settings.llm = llm
        Settings.embed_model = embed_model
        self.agent_worker = self._get_agent_tools(llm, agents)

    def _get_agent_tools(self, llm, agents):
        """
        Private function to initialise the tools that an agent can use.

        Available agents:

        query_engine_tool - agent that sends a query to a knowledge base
        """
        available_agents = {"query_engine_tool": self._get_query_engine_tool}

        tools = []
        for agent in agents:
            if agent in available_agents:
                print(f"Setting up {agent}")
                tools.append(available_agents[agent](llm))
            else:
                print(f"Cannot find tool {agent}")
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools, llm=llm, verbose=True
        )
        print("Agent created.")
        return agent_worker

    def _get_vector_index(self):
        """
        Private function that uses a MongoDB collection as knowledge base for a query engine tool
        """
        vector_store = MongoDBAtlasVectorSearch(
            self.client,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            vector_index_name="vector_index",
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index

    def _get_query_engine_tool(self, llm):
        """
        Private function that defines a query engine tool from a provided vector index.
        """
        index = self._get_vector_index()
        query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
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
        return query_engine_tool

    def retrieve_documents(self, query):
        """
        Using the query engine tool, retrieve relevant documents based on the query.

        :param query: User's input query.
        :print: List of retrieved documents.
        """
        index = self._get_vector_index()
        query_engine_test = index.as_query_engine(similarity_top_k=5)
        response = query_engine_test.retrieve(query)

        return [node.get_text() for node in response]

    def answer_query(self, query):
        """
        Complete Agentic RAG pipeline: Use available tools and generate a response.

        :param query: User's input query.
        :return: Generated response text.
        """
        agent = self.agent_worker.as_agent()
        return str(agent.chat(query))


if __name__ == "__main__":
    # Example Usage
    client = get_mongo_client(MONGO_URI)
    query = "Tell me the best NRMA insurance to get for a single mother."
    rag = AgenticRAG(client, agents=["query_engine_tool"])
    response = rag.answer_query(query)

    print("\n\n")
    input("Press enter to continue.")

    print("Evaluate answer relevance:")
    grade = Evaluator.relevance(query, response)
    print(grade)

    print("\n\n")

    print("Evaluate answer groundedness:")
    retrieved_docs = rag.retrieve_documents(query)
    grade = Evaluator.groundedness(response, retrieved_docs)
    print(grade)

    print("\n\n")

    print("Evaluate retrieval relevance:")
    grade = Evaluator.retrieval_relevance(query, retrieved_docs)
    print(grade)

    # Close the MongoDB connection when done
    client.close()
