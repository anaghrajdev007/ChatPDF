import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API key. Set it in .env or as an environment variable.")

custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow-Up Input: {question}

Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

conversation_chain = None
chat_history = []

def get_pdf_text(files):
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversationchain(vectorstore):
    llm = ChatOpenAI(
        temperature=0.2,
        openai_api_key=openai_api_key,
        base_url="https://api.groq.com/openai/v1",
        model="qwen-2.5-32b",
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
    )

app = Flask(__name__)

@app.route('/')
def manifest():
    return jsonify({
        "name": "PDF Conversational AI",
        "description": "A conversational AI service that answers questions based on the content of uploaded PDFs.",
        "documentation": """
            <h3>Endpoints</h3>
            <ul>
                <li>POST /upload: Upload PDFs to process the text content.</li>
                <li>POST /question: Ask a question based on the uploaded PDFs.</li>
            </ul>
        """
    })

@app.route('/upload', methods=['POST'])
def upload():
    global conversation_chain, chat_history
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    raw_text = get_pdf_text(files)
    if not raw_text.strip():
        return jsonify({"error": "No text found in the uploaded PDFs. Please upload valid documents."}), 400

    text_chunks = get_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversationchain(vectorstore)
    chat_history = []
    return jsonify({"message": "PDFs processed successfully."})

@app.route('/question', methods=['POST'])
def question():
    global conversation_chain, chat_history
    if conversation_chain is None:
        return jsonify({"error": "No conversation chain available. Please upload PDFs first."}), 400

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question_text = data['question']
    response = conversation_chain({"question": question_text})
    chat_history = response.get("chat_history", [])
    
    answer = chat_history[-1].content if chat_history else ""
    return jsonify({
        "answer": answer,
        "chat_history": [msg.content for msg in chat_history]
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8501)
