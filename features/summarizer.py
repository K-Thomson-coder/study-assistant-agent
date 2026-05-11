from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_text(text) :
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2
    )

    prompt = f"Summarize this clearly:\n{text}"

    response = llm.invoke(prompt)

    return response.content