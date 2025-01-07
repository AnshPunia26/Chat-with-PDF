import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Streamlit Application Title
st.title("Chat with PDF - Interactive Chatbot")

# Sidebar for OpenAI API Key Input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    openai.api_key = api_key


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, max_length=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to create embeddings for text chunks
def create_embeddings(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings

# Function to find the most relevant chunk
def find_relevant_chunk(query, embeddings, text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    most_relevant_index = similarities.argmax()
    return text_chunks[most_relevant_index]

# Function to query OpenAI LLM
def query_openai(context, question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify GPT-3.5
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        return None


# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF file(s):", type=["pdf"], accept_multiple_files=True)

if uploaded_files and api_key:
    with st.spinner("Processing uploaded files..."):
        all_text = ""
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            all_text += pdf_text + "\n"

        # Display extracted text
        st.subheader("Extracted Text from Uploaded PDFs:")
        st.text_area("Extracted Text", all_text[:5000], height=250)

        # Split text into chunks and generate embeddings
        text_chunks = split_text_into_chunks(all_text)
        embeddings = create_embeddings(text_chunks)

        st.success("Files processed successfully!")

    # User query section
    st.subheader("Ask Questions about the Uploaded PDFs")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Generating response..."):
            # Find the most relevant chunk
            relevant_chunk = find_relevant_chunk(user_question, embeddings, text_chunks)

            # Query OpenAI LLM
            answer = query_openai(relevant_chunk, user_question)

            # Display results
            st.markdown(f"### Answer: {answer}")
            with st.expander("Relevant Section from Document:"):
                st.markdown(f"**{relevant_chunk}**")

else:
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key.")
    st.info("Upload one or more PDF files to get started.")

# Footer
st.markdown("---")
st.markdown("Powered by Sentence Transformers, OpenAI, and Streamlit")
