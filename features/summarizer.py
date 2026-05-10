from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_text(text) :
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = f"Summarize this clearly:\n{text}"

    response = llm.invoke(prompt)

    return response.content