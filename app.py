# Importing dependencies
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API key. Set it in .env or as an environment variable.")

# Custom template to guide LLM model
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow-Up Input: {question}

Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extract text from PDFs
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

# Convert text into chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)   
    return text_splitter.split_text(raw_text)

# Use sentence-transformers and FAISS for vector storage
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Generate conversation chain  
def get_conversationchain(vectorstore):
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    
    return conversation_chain

# Handle user questions and display chat history
def handle_question(question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return
    
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with Multiple PDFs :books:")
    
    # User question input
    question = st.text_input("Ask a question about your document:")
    if question:
        handle_question(question)
    
    # Sidebar file uploader
    with st.sidebar:
        st.subheader("Your Documents")
        docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                
                # Extract text from PDFs
                raw_text = get_pdf_text(docs)
                
                if not raw_text.strip():
                    st.error("No text found in the uploaded PDFs. Please upload a valid document.")
                    return
                
                # Convert text into chunks
                text_chunks = get_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversationchain(vectorstore)

if __name__ == '__main__':
    main()
