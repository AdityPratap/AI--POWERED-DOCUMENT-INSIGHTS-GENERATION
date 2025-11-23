<<<<<<< HEAD
import os
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import streamlit as st
import numpy as np

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE = faiss.IndexFlatL2(768)  # Assuming embedding dimension is 768

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to generate embeddings
def generate_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Function to store embeddings in Faiss index
def store_embeddings(embeddings):
    VECTOR_STORE.add(embeddings.reshape(1, -1))

# Function to search for similar documents using Faiss
def search_similar_documents(query_embedding, top_k=5):
    D, I = VECTOR_STORE.search(query_embedding.reshape(1, -1), k=top_k)
    return D, I

# Streamlit Interface with Chatbot
def streamlit_interface():
    st.title("PDF Comparison and Analysis")

    pdf_directory = st.text_input("Enter the PDF directory path", "pdfs/")
    if not os.path.exists(pdf_directory):
        st.error("The provided directory does not exist.")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("No PDF files found in the directory.")
        return

    if st.button("Extract and Analyze All PDFs"):
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            embeddings = generate_embeddings(text)
            store_embeddings(embeddings)
        st.success("Embeddings for all PDFs generated and stored successfully.")

    st.subheader("Chat with the Bot")
    query = st.text_input("Enter your query")
    if query:
        query_embedding = generate_embeddings(query)
        D, I = search_similar_documents(query_embedding, top_k=3)
        st.write("Top 3 similar documents:")
        for idx, distance in zip(I.ravel(), D.ravel()):
            st.write(f"Document: {pdf_files[idx]} (Distance: {distance:.4f})")
        
        # Display the text of the most similar document as a response
        most_similar_doc = pdf_files[I[0][0]]
        pdf_path = os.path.join(pdf_directory, most_similar_doc)
        most_similar_text = extract_text_from_pdf(pdf_path)
        st.subheader(f"Response from {most_similar_doc}")
        st.write(most_similar_text)

# Main function
def main():
    streamlit_interface()

if __name__ == "__main__":
    main()
=======
import os
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import streamlit as st
import numpy as np

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE = faiss.IndexFlatL2(768)  # Assuming embedding dimension is 768

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to generate embeddings
def generate_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Function to store embeddings in Faiss index
def store_embeddings(embeddings):
    VECTOR_STORE.add(embeddings.reshape(1, -1))

# Function to search for similar documents using Faiss
def search_similar_documents(query_embedding, top_k=5):
    D, I = VECTOR_STORE.search(query_embedding.reshape(1, -1), k=top_k)
    return D, I

# Streamlit Interface with Chatbot
def streamlit_interface():
    st.title("PDF Comparison and Analysis")

    pdf_directory = st.text_input("Enter the PDF directory path", "pdfs/")
    if not os.path.exists(pdf_directory):
        st.error("The provided directory does not exist.")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("No PDF files found in the directory.")
        return

    if st.button("Extract and Analyze All PDFs"):
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            embeddings = generate_embeddings(text)
            store_embeddings(embeddings)
        st.success("Embeddings for all PDFs generated and stored successfully.")

    st.subheader("Chat with the Bot")
    query = st.text_input("Enter your query")
    if query:
        query_embedding = generate_embeddings(query)
        D, I = search_similar_documents(query_embedding, top_k=3)
        st.write("Top 3 similar documents:")
        for idx, distance in zip(I.ravel(), D.ravel()):
            st.write(f"Document: {pdf_files[idx]} (Distance: {distance:.4f})")
        
        # Display the text of the most similar document as a response
        most_similar_doc = pdf_files[I[0][0]]
        pdf_path = os.path.join(pdf_directory, most_similar_doc)
        most_similar_text = extract_text_from_pdf(pdf_path)
        st.subheader(f"Response from {most_similar_doc}")
        st.write(most_similar_text)

# Main function
def main():
    streamlit_interface()

if __name__ == "__main__":
    main()
>>>>>>> 4f5bf9a8a33864fcb24b4b71fdf03d72ca47e619
