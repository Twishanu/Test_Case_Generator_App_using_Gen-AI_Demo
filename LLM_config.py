import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from typing import List, Dict, Any
import uuid

load_dotenv()

class LLMConfig:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize Chroma DB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"}  # Better for semantic search
        )
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def store_embeddings(self, text: str, chat_id: str, metadata: Dict[str, Any] = None):
        """Store text chunks as embeddings with chat_id metadata"""
        try:
            if not text.strip():
                return 0
                
            chunks = self.chunk_text(text)
            if not chunks:
                return 0
            
            # Generate embeddings for each chunk
            embeddings_list = self.embeddings.embed_documents(chunks)
            
            # Generate unique IDs for each chunk
            ids = [f"{chat_id}_{uuid.uuid4()}" for _ in range(len(chunks))]
            
            # Prepare metadata for each chunk - ensure chat_id is properly formatted
            metadatas = [{"chat_id": str(chat_id), "chunk_index": i, **(metadata or {})} 
                        for i in range(len(chunks))]
            
            # Store in Chroma DB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            
            return len(chunks)
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return 0
    
    def retrieve_relevant_context(self, query: str, chat_id: str, top_k: int = 3) -> str:
        """Retrieve relevant context from embeddings for specific chat"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Chroma DB with PROPER chat_id filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"chat_id": {"$eq": str(chat_id)}}  # FIXED: Proper filter syntax
            )
            
            if results and results['documents'] and len(results['documents']) > 0:
                # Combine retrieved chunks into context
                context_chunks = []
                for doc in results['documents'][0]:
                    if doc:  # Skip empty documents
                        context_chunks.append(doc)
                
                if context_chunks:
                    context = "\n\n".join(context_chunks)
                    return context
            return ""
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def delete_chat_embeddings(self, chat_id: str) -> int:
        """Delete all embeddings for a specific chat"""
        try:
            # Use proper filter syntax to delete by chat_id
            results = self.collection.get(where={"chat_id": {"$eq": str(chat_id)}})
            
            if results and results['ids']:
                # Delete all documents for this chat
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            return 0
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return 0
    
    def get_chat_embedding_count(self, chat_id: str) -> int:
        """Get count of embeddings for a specific chat (for debugging)"""
        try:
            results = self.collection.get(where={"chat_id": {"$eq": str(chat_id)}})
            return len(results['ids']) if results and results['ids'] else 0
        except Exception as e:
            print(f"Error getting embedding count: {e}")
            return 0
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    def generate_response(self, question: str, context: str = "") -> str:
        prompt = f"""You are an assistant.
Use only the retrieved context to precisely answer the question.
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"