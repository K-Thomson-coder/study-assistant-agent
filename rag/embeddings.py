from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embeddings() :
    return HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2"
        )