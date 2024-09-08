import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load the embedding model for document indexing
model = SentenceTransformer('all-MiniLM-L6-v2')

# Document sections (for testing, replace with actual extracted sections)
document_sections = ["Section 1 content", "Section 2 content", "Section 3 content"]

# Create embeddings
embeddings = model.encode(document_sections)

# Initialize FAISS for searching similar sections
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to find the most relevant section and get the answer
def get_answer(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Streamlit interface
st.title("OMC-MDG")
query = st.text_input("Ask a question:")
if query:
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)  # Get top 1 section
    relevant_section = document_sections[I[0][0]]
    answer = get_answer(query, relevant_section)
    st.write(f"Answer: {answer}")