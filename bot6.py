import os
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import streamlit as st
import numpy as np

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Update this based on the actual embedding dimension
VECTOR_STORE = faiss.IndexFlatL2(EMBEDDING_DIM)

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
    if len(embeddings.shape) == 1:  # Ensure embeddings have two dimensions
        embeddings = embeddings.reshape(1, -1)
    VECTOR_STORE.add(embeddings)

# Streamlit Interface
def streamlit_interface():
    st.title("PDF Comparison and Analysis")

    pdf_directory = st.text_input("Enter the PDF directory path", "C:/Users/lenovo/OneDrive/Desktop/Alemeno_Project/")
    if not os.path.exists(pdf_directory):
        st.error("The provided directory does not exist.")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("No PDF files found in the directory.")
        return

    selected_doc = st.selectbox("Select a PDF to analyze", pdf_files)
    pdf_path = os.path.join(pdf_directory, selected_doc)

    if st.button("Extract and Analyze"):
        text = extract_text_from_pdf(pdf_path)
        embeddings = generate_embeddings(text)
        store_embeddings(embeddings)
        st.success("Embeddings generated and stored successfully.")
        st.write("Extracted Text:", text)

    # Implement search functionality using Faiss
    query_text = st.text_input("Enter text to search for similar documents")
    if query_text:
        query_embedding = generate_embeddings(query_text)
        D, I = VECTOR_STORE.search(query_embedding.reshape(1, -1), k=5)
        st.write("Top 5 similar documents:")
        for idx, distance in zip(I.ravel(), D.ravel()):
            if idx < len(pdf_files):
                st.write(f"Document: {pdf_files[idx]} (Distance: {distance:.4f})")
            else:
                st.write(f"Document ID: {idx} (Distance: {distance:.4f})")

# Main function
def main():
    streamlit_interface()

if __name__ == "__main__":
    main()
