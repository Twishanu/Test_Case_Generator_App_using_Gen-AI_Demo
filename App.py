from db_config import DBOperations
from LLM_config import LLMConfig
import streamlit as st
import PyPDF2
from io import BytesIO

# Initialize components
db = DBOperations()
llm = LLMConfig()
st.set_page_config(layout="wide")
st.title("RAG embedded - Test Case Generator")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = db.getChats()
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = {}

# Function to extract text from uploaded files
def extract_text_from_uploaded_files(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            for page in pdf_reader.pages:
                all_text += page.extract_text() + "\n"
        elif file.type == "text/plain":
            all_text += str(file.read(), "utf-8") + "\n"
    return all_text

# Function to create new chat
def create_new_chat():
    chat_id = db.createChats("Untitled")
    if chat_id:
        st.session_state.chats = db.getChats()
        st.session_state.active_chat = chat_id
        st.session_state.messages = []  # Clear current messages
        st.session_state.uploaded_files_processed[chat_id] = set()
        # st.rerun()

# Function to delete chat
def delete_chat(chat_id):
    # Delete from database first
    success = db.delete_chat(chat_id)
    if success:
        # Delete corresponding embeddings
        deleted_count = llm.delete_chat_embeddings(chat_id)
        # st.success(f"Deleted chat and {deleted_count} embeddings")
    
    # Update session state
    st.session_state.chats = db.getChats()
    if st.session_state.active_chat == chat_id:
        st.session_state.active_chat = st.session_state.chats[0]["chat_id"] if st.session_state.chats else None
        st.session_state.messages = []
    st.rerun()

# Function to load messages for a chat
def load_chat_messages(chat_id):
    st.session_state.messages = db.getChatMessages(chat_id)

# Sidebar - CHAT MANAGEMENT ONLY
with st.sidebar:
    st.title("Chats")
    st.button("Start New Chat", on_click=create_new_chat, use_container_width=True)
    
    st.divider()
    
    # Display chat list
    for chat in st.session_state.chats:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                chat["title"],
                key=f"chat_{chat['chat_id']}",
                use_container_width=True,
                type="primary" if st.session_state.active_chat == chat["chat_id"] else "secondary"
            ):
                st.session_state.active_chat = chat["chat_id"]
                load_chat_messages(chat["chat_id"])
                st.rerun()
        with col2:
            if st.button("X", key=f"delete_{chat['chat_id']}"):
                delete_chat(chat["chat_id"])

# Main chat area
if st.session_state.active_chat:
    # Initialize processed files tracking for this chat
    if st.session_state.active_chat not in st.session_state.uploaded_files_processed:
        st.session_state.uploaded_files_processed[st.session_state.active_chat] = set()
    
    # --- MOVED FILE UPLOADER TO CHAT LEVEL ---
    st.subheader("Upload Documents for this Chat")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.active_chat}"  # Unique key per chat
    )
    
    # Process uploaded files if any
    context = ""
    if uploaded_files:
        with st.spinner("Processing and embedding documents..."):
            for file in uploaded_files:
                # Check if file was already processed for this chat
                if file.name not in st.session_state.uploaded_files_processed[st.session_state.active_chat]:
                    file_text = extract_text_from_uploaded_files([file])
                    if file_text.strip():
                        # Store embeddings with chat_id metadata
                        chunk_count = llm.store_embeddings(
                            file_text, 
                            st.session_state.active_chat,
                            metadata={"filename": file.name}
                        )
                        st.session_state.uploaded_files_processed[st.session_state.active_chat].add(file.name)
                        st.success(f"Processed {file.name} into {chunk_count} chunks")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# In app.py - replace the chat input section with this:

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # If this is the first message in the chat, use it as the title
        if len(st.session_state.messages) == 0:
            # Create a shortened version for the title (max 40 chars)
            short_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
            
            # Update the chat title in the database
            if db.update_chat_title(st.session_state.active_chat, short_title):
                # Refresh the chats list to show the new title
                st.session_state.chats = db.getChats()
        # Retrieve relevant context using RAG
        with st.spinner("Retrieving relevant context..."):
            retrieved_context = llm.retrieve_relevant_context(prompt, st.session_state.active_chat)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        db.addMessage(st.session_state.active_chat, "user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = llm.generate_response(prompt, retrieved_context)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        db.addMessage(st.session_state.active_chat, "assistant", response)
        
        # Reload messages to ensure sync with database
        load_chat_messages(st.session_state.active_chat)
        st.rerun()

else:
    st.code("Create a new chat or select an existing one from the sidebar to get started.")