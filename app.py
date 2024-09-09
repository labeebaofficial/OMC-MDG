import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load the embedding model for document indexing
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit App
st.title("OMC-MDG")

# Define the path to the uploaded PDF in your GitHub repo
pdf_path = 'omc-mdg.pdf'

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Add a file uploader if you want users to upload a PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Display the extracted text (for testing purposes)
    st.write("Extracted Text from the PDF:")
    st.write(pdf_text)

    # Break PDF text into sections (for embedding)
    document_sections = pdf_text.split("\n\n")  # Split text into sections by paragraphs

    # Create embeddings
    embeddings = model.encode(document_sections)

    # Initialize FAISS for searching similar sections
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Function to find the most relevant section and get the answer
    def get_answer(query, context):
        result = qa_pipeline(question=query, context=context)
        return result['answer']

    # Get user query
    user_query = st.text_input("Enter your query:")

    if user_query:
        # Encode the query
        query_embedding = model.encode([user_query])

        # Search for the most relevant section
        D, I = index.search(query_embedding, k=1)  # Get top 1 section
        relevant_section = document_sections[I[0][0]]

        # Get the answer from the most relevant section
        answer = get_answer(user_query, relevant_section)
        st.write(f"Answer: {answer}")