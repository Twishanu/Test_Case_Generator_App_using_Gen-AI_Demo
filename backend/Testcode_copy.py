import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

import warnings
warnings.filterwarnings('ignore')

class Chatbot:
    load_dotenv()
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self._initialize_components()
    
    def _recreate_rag_chain(self):
        model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", convert_system_message_to_human=True)
        system_prompt = (
            "You are an assistant. "
            "Use only the retrieved context to precisely answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n{context}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answering_chain)
    
    def clear_memory(self):
        try:
            # Clear the vectorstore
            if self.vectorstore:
                self.vectorstore.delete_collection()
            
            # Reset in-memory storage
            self.documents = []
            self.chunks = []
            self.raw_documents = []
            
            # Reinitialize 
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory="db"
            )
            self.retriever = self.vectorstore.as_retriever()
            self._recreate_rag_chain()
            
            return True
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False

    def process_raw_docs(self, raw_files):
        print("hello")
        self.raw_documents = []

        for raw_file in raw_files:
            temp_raw_file_path = f"temp_{raw_file.name}"
            with open(temp_raw_file_path, "wb") as f:
                f.write(raw_file.getbuffer())
            try:
                if raw_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_raw_file_path)
                    self.raw_documents.extend(loader.load())
                elif raw_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(temp_raw_file_path)
                    self.raw_documents.extend(loader.load())
                else:
                    with open(temp_raw_file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        self.raw_documents.append(Document(page_content=text))
            finally:
                if os.path.exists(temp_raw_file_path):
                    os.remove(temp_raw_file_path)
        self.raw_chunks = self.splitter.split_documents(self.raw_documents)
        self.vectorstore = Chroma.from_documents(
            documents=self.raw_chunks,
            embedding=self.embeddings,
            persist_directory="db"
        )
        self.retriever = self.vectorstore.as_retriever()
        model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", convert_system_message_to_human=True)
        system_prompt = (
            "You are an assistant. "
            "Use only the retrieved context to precisely answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n{context}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answering_chain)


    def _initialize_components(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Initialize with empty vectorstore
        self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory="db")
        self.retriever = self.vectorstore.as_retriever()
        
        model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", convert_system_message_to_human=True)
        system_prompt = (
            "You are an assistant. "
            "Use only the retrieved context to precisely answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n{context}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answering_chain)
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and update the vectorstore"""
        self.documents = []
        
        for uploaded_file in uploaded_files:
            # Create a temporary file path
            temp_file_path = f"temp_{uploaded_file.name}"
            
            # Save the uploaded file temporarily
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                    self.documents.extend(loader.load())
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(temp_file_path)
                    self.documents.extend(loader.load())
                else:
                    with open(temp_file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        self.documents.append(text)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # Split and update vectorstore
        self.chunks = self.splitter.split_documents(self.documents)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory="db"
        )
        self.retriever = self.vectorstore.as_retriever()
        self._recreate_rag_chain()
        
        # Recreate the chain with updated retriever
        model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", convert_system_message_to_human=True)
        system_prompt = (
            "You are an assistant. "
            "Use only the retrieved context to precisely answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n{context}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answering_chain)

if __name__ == "__main__":
    chatbot = Chatbot()