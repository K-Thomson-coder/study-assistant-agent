from langchain_community.vectorstores import FAISS

def create_vectorstores(chunks, embeddings) :
    db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db