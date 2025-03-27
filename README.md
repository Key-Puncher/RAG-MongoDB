# Retrieval Augmented Generation

This repo contains code for running Retrieval Augmented Generation with a vector database.

First, read data from pdf files and generate embeddings from them. Then RAG is done over these embeddings. 

To start, install the dependencies and process data. I use MongoDB, and create a vector search index via utilities from `utils/mongo.py`. 

To run the RAG pipelines, run one of the files `rag.py` or `rag_agentic.py`.
