import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

class PDFConversationalAI:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Missing OpenAI API key. Set it in .env or as an environment variable.")

        self.conversation_chain = None
        self.chat_history = []

        self.prompt_template = PromptTemplate.from_template(
            """Given the following conversation and a follow-up question, rephrase the follow-up question 
            to be a standalone question, in its original language.
            
            Chat History:
            {chat_history}

            Follow-Up Input: {question}

            Standalone question:"""
        )

    def extract_text_from_pdfs(self, files):
        text = ""
        for file in files:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        return text

    def chunk_text(self, raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        return text_splitter.split_text(raw_text)

    def create_vectorstore(self, text_chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        return faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    def setup_conversation_chain(self, vectorstore):
        llm = ChatOpenAI(
            temperature=0.2,
            openai_api_key=self.openai_api_key,
            base_url="https://api.groq.com/openai/v1",
            model="qwen-2.5-32b",
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            condense_question_prompt=self.prompt_template,
            memory=memory,
        )
        self.chat_history = []

    def process_pdfs(self, files):
        raw_text = self.extract_text_from_pdfs(files)
        if not raw_text.strip():
            return {"error": "No text found in the uploaded PDFs. Please upload valid documents."}, 400

        text_chunks = self.chunk_text(raw_text)
        vectorstore = self.create_vectorstore(text_chunks)
        self.setup_conversation_chain(vectorstore)
        return {"message": "PDFs processed successfully."}, 200

    def ask_question(self, question):
        if self.conversation_chain is None:
            return {"error": "No conversation chain available. Please upload PDFs first."}, 400

        response = self.conversation_chain({"question": question})
        self.chat_history = response.get("chat_history", [])

        answer = self.chat_history[-1].content if self.chat_history else ""
        return {
            "answer": answer,
            "chat_history": [msg.content for msg in self.chat_history]
        }, 200
