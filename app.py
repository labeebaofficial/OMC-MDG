import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

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

# Define the path to the uploaded PDF in your GitHub repo
pdf_path = 'omc-mdg.pdf'

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        text = ''
        # Iterate through each page and extract text
        for page in reader.pages:
            text += page.extract_text()
    return text

# Streamlit App
st.title("OMC-MDG")

# Add a file uploader if you want users to upload a PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Display the extracted text (for testing purposes)
    st.write("Extracted Text from the PDF:")
    st.write(pdf_text)

    # Add further processing or querying here
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        # Simple keyword-based search function (example)
        def search_pdf_content(query, text):
            results = [sentence for sentence in text.split('. ') if query.lower() in sentence.lower()]
            return results

        search_results = search_pdf_content(user_query, pdf_text)
        
        if search_results:
            st.write("Results:")
            for result in search_results:
                st.write(result)
        else:
            st.write("No relevant information found.")
