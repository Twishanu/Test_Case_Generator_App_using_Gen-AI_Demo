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

file_path = "D:/AI enabled Test case gen/Documents/website_ui_changes.pdf"  
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    raise ValueError("Unsupported file type")

documents = loader.load()

# print(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# print(chunks)

vectorstore=Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
print("Embedding complete and saved!")

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",convert_system_message_to_human=True)
print(model.invoke("hi").content)

# Retrieval function
def retrieve(query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

def ask_rag(query):
    context = retrieve(query)
    prompt = f"""You are an assistant who is a functional software tester. Your role is to generate test cases based on the context provided which is a functional specification document. Generate precise test cases. Don't imagine or halucinate.
Context: {context}
Question: {query}
Answer:"""

    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",convert_system_message_to_human=True, temperature=0)
    response = model.invoke(prompt).content
    print(response)

# Example
ask_rag("Generate test cases for Navigation Menu Redesign")


# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# print(retriever)
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question "
#     "If you don't know the answer, say that you don't know."
#     "Use three sentences maximum and keep the answer concise."
#     "\n\n"
#     "{context}"
# )

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
# rag_chain = create_retrieval_chain(retriever, question_answering_chain)
# response = rag_chain.invoke({"input":"what do you summarize from the given doc?"})
# response["answer"]

