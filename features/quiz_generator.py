from langchain_google_genai import ChatGoogleGenerativeAI

def generate_quiz(text) :
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = f"Generate 5 quiz questions from:\n{text}"

    response = llm.invoke(prompt)

    return response.content