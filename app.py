import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure Gemini API
genai.configure(api_key='Your Key')

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

# Initialize sentence transformer model for embedding
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
embedding_size = 384  # Size of the embeddings from 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(embedding_size)
texts = []  # To store the original texts


def embed_text(text):
    return sentence_model.encode(text)


def add_to_faiss(text, embedding):
    global index, texts
    index.add(np.array([embedding]))
    texts.append(text)


def search_faiss(query_embedding, top_k=5):
    if index.ntotal == 0:
        return []

    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [texts[i] for i in indices[0]]


def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()


def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2)


def process_uploaded_file(file):
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        text = extract_text_from_excel(file)
    elif file.type == "application/json":
        text = extract_text_from_json(file)
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    else:
        st.error(f"Unsupported file type: {file.type}")
        return

    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]  # Simple chunking
    for chunk in chunks:
        embedding = embed_text(chunk)
        add_to_faiss(chunk, embedding)
    st.success("File processed and added to the knowledge base.")


def get_gemini_response(prompt, context=""):
    if context:
        full_prompt = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"
    else:
        full_prompt = f"Human: {prompt}\n\nAssistant:"

    response = model.generate_content(full_prompt)
    return response.text


def main():
    st.set_page_config(layout="wide")

    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Upload")
        uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, Excel, JSON, or TXT)",
                                         type=["pdf", "docx", "xlsx", "json", "txt"])
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)

    # Main chat interface
    st.title("Gemini Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get relevant context from FAISS if documents have been uploaded
        query_embedding = embed_text(prompt)
        relevant_context = search_faiss(query_embedding)
        context_text = "\n".join(relevant_context)

        # Generate Gemini response
        response = get_gemini_response(prompt, context_text)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()