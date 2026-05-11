from langchain_community.vectorstores import Chroma

def create_vectorstores(chunks, embeddings) :
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db