from sentence_transformers import SentenceTransformer


NUM_DOC_LIMIT = 250  # the number of documents to process and generate embeddings.

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
