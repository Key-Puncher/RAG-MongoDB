import time
from pymongo.operations import SearchIndexModel


INDEX_DEFINITION = {
    "fields": [
        {
            "numDimensions": 1024,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector",
        }
    ]
}


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
