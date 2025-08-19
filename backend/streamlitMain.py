from Testcode_copy import Chatbot
import streamlit as st

st.markdown(
    """
<style>
    # div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
    #     position: sticky;
    #     top: 2.875rem;
    #     background-color: #0f0f0fff;
    #     z-index: 999;
    # }
</style>
    """,
    unsafe_allow_html=True
)
bot = Chatbot()

st.title("RAG embedded - Test Case Generator")


if "chats" not in st.session_state:
    st.session_state.chats = ["Chat 1"]
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 1
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"

# Function to add a new chat
def create_new_chat():
    st.session_state.chat_count += 1
    new_chat = f"Chat {st.session_state.chat_count}"
    st.session_state.chats.append(new_chat)
    st.session_state.active_chat = new_chat  # auto-select new chat
# Function to delete a chat
# Function to delete a chat
def delete_chat(chat_name):
    st.session_state.chats = [c for c in st.session_state.chats if c != chat_name]
    # if active chat is deleted, select the first available chat if exists
    if st.session_state.active_chat == chat_name:
        st.session_state.active_chat = st.session_state.chats[0] if st.session_state.chats else None
# Sidebar
with st.sidebar:
    if st.button("Delete Memory", help="Clears all stored documents", width="stretch"):
        if bot.clear_memory():
            st.success("Memory cleared successfully!")
        else:
            st.error("Failed to clear memory")
    # Later on we can fine-tune the model
    raw_files = st.file_uploader("Choose base docs", accept_multiple_files=True)
    if raw_files:
        bot.process_raw_docs(raw_files)
        st.success("Base docs processed")
        
    st.title("Chats")
    st.button("New Chat", width="stretch", on_click=create_new_chat)
    
    for chat_name in st.session_state.chats:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.button(
                chat_name,
                type="primary" if chat_name == st.session_state.active_chat else "secondary",
                use_container_width=True,
                key=f"btn-{chat_name}",
                on_click=lambda c=chat_name: st.session_state.update({"active_chat": c})
            )
        with col2:
            if st.button("X", key=f"del-{chat_name}"):
                delete_chat(chat_name)
                st.rerun()


# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm RAG assistant. You can attach docs and type your question in one go."}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# --- Single input with files ---
prompt = st.chat_input(
    "Type your message and let's get goin...",
    accept_file="multiple",
    file_type=["pdf", "docx", "txt"]
)

if prompt:
    # 1) Always index files first (if any)
    if getattr(prompt, "files", None):
        with st.spinner(f"Processing {len(prompt.files)} document(s)…"):
            # If your API supports batch processing, keep as one call.
            # Otherwise, process one-by-one for smoother UX (optional).
            bot.clear_memory()
            bot.process_uploaded_files(prompt.files)
        st.session_state.messages.append(
            {"role": "system", "content": f"Processed {len(prompt.files)} document(s)."}
        )
        st.success(f"Processed {len(prompt.files)} file(s) ✅")

    # 2) Then answer the query (if provided)
    if getattr(prompt, "text", None):
        user_text = prompt.text
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Searching the uploaded docs and composing an answer…"):
                resp = bot.rag_chain.invoke({"input": user_text})
                answer = (
                    resp.get("answer", None)
                    if isinstance(resp, dict) else resp
                )
                if answer is None and isinstance(resp, dict):
                    # Fallback for different chain output keys
                    answer = resp.get("output_text", str(resp))
                st.write(answer)

            # --- Feedback System per response ---
            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            selected = st.feedback("thumbs", key=f"feedback_{len(st.session_state.messages)}")
            if selected is not None:
                st.markdown(f"You selected: {sentiment_mapping[selected]}")

        # Store assistant message in session
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # If user only uploaded files and didn’t type text, just guide them
    if getattr(prompt, "files", None) and not getattr(prompt, "text", None):
        st.info("Documents indexed. How can I help you with the document?.")
