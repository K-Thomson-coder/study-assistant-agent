from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def create_qa_chain(retriever) :

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
        )
    
    def format_docs(docs) :
        return "\n\n".join(d.page_content for d in docs)
    
    prompt = ChatPromptTemplate.from_template(
        """
        Use ONLY the context below.                
          
        Context :
        {context}
          
        Question :
        {question}
          """)
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain