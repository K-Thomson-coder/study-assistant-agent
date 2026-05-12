from langchain_community.vectorstores import FAISS

def create_vectorstores(chunks, embeddings) :
    faiss = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return faiss