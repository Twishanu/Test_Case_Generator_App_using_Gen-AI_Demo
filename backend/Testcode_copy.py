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
from langchain_core.prompts import MessagesPlaceholder

import warnings
warnings.filterwarnings('ignore')


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print(GOOGLE_API_KEY[:6])

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("hello, world! I am Twishanu yes")
# print(vector[:5])

file_path = "D:/AI enabled Test case gen/Documents/fictional_rag_story.pdf"  
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    raise ValueError("Unsupported file type")

documents = loader.load()

# print(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# print(chunks)

vectorstore=Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
print("Embedding complete and saved!")
retriever = vectorstore.as_retriever()
print(retriever)

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite",convert_system_message_to_human=True)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answering_chain=create_stuff_documents_chain(model, chat_prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)
response = rag_chain.invoke({"input":"Who is the librarian?"})
print(response["answer"])