import ollama

from utils.common import get_embedding, MONGO_URI, DB_NAME, COLLECTION_NAME
from utils.mongo import get_mongo_client
from evaluator import Evaluator


class RAG:
    def __init__(self, client):
        """
        Initialize the RAG system with a retriever and a generator.

        :param client: A connection to a MongoDB instance.
        """
        self.collection = client[DB_NAME][COLLECTION_NAME]

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
        results = self.collection.aggregate(pipeline)
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
    client = get_mongo_client(MONGO_URI)

    query = "Tell me the best NRMA insurance to get for a single mother."

    rag = RAG(client)
    response = rag.answer_query(query)
    print(response)
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
