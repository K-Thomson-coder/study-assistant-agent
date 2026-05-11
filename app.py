import streamlit as st
import tempfile
import os

from dotenv import load_dotenv

from rag.loader import load_pdf
from rag.chunker import split_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstores
from rag.qa_chain import create_qa_chain

from features.summarizer import summarize_text
from features.quiz_generator import generate_quiz

load_dotenv()

st.set_page_config(
    page_title="AI Study Copilot",
    page_icon="01",
    layout="wide"
)

st.title("Study Assistant Agent")

@st.cache_resource(show_spinner=False)
def load_embeddings() :
    get_embeddings()

@st.cache_resource(show_spinner=False)
def build_vectorstores(chunks, embeddings) :
    create_vectorstores(chunks, embeddings)

if "processed" not in st.session_state :
    st.session_state.processed = False
if "documents" not in st.session_state :
    st.session_state.documents = False
if "vectorstore" not in st.session_state :
    st.session_state.vectorstore = False
if "qa_chain" not in st.session_state :
    st.session_state.qa_chain = False
if "current_file" not in st.session_state :
    st.session_state.current_file = None

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file :

    file_id = (uploaded_file.name, uploaded_file.size)
    
    if file_id != st.session_state.current_file :

        st.session_state.processed = False

        st.session_state.documents = None
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None

        st.session_state.current_file = file_id

    if not st.session_state.processed :

        with st.spinner("Processing PDF...") :

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file :
                
                tmp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = tmp_file.name

            documents = load_pdf(temp_path)
            chunks = split_documents(documents)

            embeddings = load_embeddings()
            retriever = (build_vectorstores(chunks, embeddings)).as_retriever(search_kwargs={'k': 4})

            qa_chain = create_qa_chain(retriever)

            st.session_state.documents = documents
            st.session_state.vectorstore = retriever
            st.session_state.qa_chain = qa_chain

            st.session_state.processed = True

            os.remove(temp_path)
        st.toast("PDF processed successfully!")

    option = st.selectbox(
        "Choose Feature",
        [
            "Ask Questions",
            "Summarize",
            "Generate Quiz"
        ]
    )

    if option == "Ask Questions" :
        question = st.text_input("Ask")

        if question :
            
            with st.spinner("Generating answer...") :
                response = st.session_state.qa_chain.invoke(question)
                answer = response.content

                st.write(answer)

    elif option == "Summarize" :
        
        if st.button("Generate Summary") :
            
            with st.spinner("Generating summary...") :
                text = " ".join(
                    [
                        doc.page_content for doc in st.session_state.documents[:5]
                    ]
                )

                summary = summarize_text(text)

                st.write(summary['text'])

    elif option == "Generate Quiz" :

        if st.button("Generate Quiz") :

            with st.spinner("Generating quiz...") :
                text = " ".join(
                    [
                        doc.page_content for doc in st.session_state.documents[:5]
                    ]
                )

                quiz = generate_quiz(text)

                st.write(quiz[0]['text'])