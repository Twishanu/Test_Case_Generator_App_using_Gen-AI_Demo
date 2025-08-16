from Testcode_copy import Chatbot
import streamlit as st
import time

bot = Chatbot()

st.title("RAG embedded - Test Case Generator")
# Sidebar code fulll
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        bot.process_uploaded_files(uploaded_files) 
    if st.button("Delete Memory", help="Clears all stored documents"):
        if bot.clear_memory():
            st.success("Memory cleared successfully!")
        else:
            st.error("Failed to clear memory")
# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke({"input": input})
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm RAG assistant. How can I assist you with?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer by going through the document..."):
            response = generate_response(input) 
            result_text = response["answer"]
            st.write(result_text) 
    message = {"role": "assistant", "content": response["answer"]}
    st.session_state.messages.append(message)
