from langchain_google_genai import ChatGoogleGenerativeAI

def generate_quiz(text) :
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.5
    )

    prompt = f"Generate 5 MCQ questions from:\n{text}"
    prompt = f"""

I want you to generate multiple-choice questions based on this text: [Insert Text]. Follow this exact format:
Example :
Try these questions :

Q1. What is the first phase of SDLC?

A) Design B) Testing C) Requirement Analysis D) Maintenance

Q2. Which model is best for small projects?

A) Waterfall B) Spiral C) V-Model D) Agile

....

Key : 1. C, 2. A, .....

Now, generate 5 questions in this same format from {text}.

"""

    response = llm.invoke(prompt)

    return response.content