import streamlit as st
import time
import uuid
import json
from io import StringIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spectra",
    page_icon="",
    layout="wide",
    menu_items={
        "About": "AI powered Test case geneartor",
    },
)

# =========================
# GLOBAL STYLES (Dark + Animations)
# =========================
st.markdown(
    """
    <style>
      :root {
        --bg: #0b0f17;
        --panel: #111827;
        --ink: #e5e7eb;
        --muted: #9ca3af;
        --accent: #8b5cf6; /* violet */
        --accent-2: #06b6d4; /* cyan */
        --success: #10b981;
        --danger: #ef4444;
      }

      html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 600px at 80% -10%, rgba(139,92,246,0.18), transparent 60%),
                    radial-gradient(1000px 500px at -20% 10%, rgba(6,182,212,0.15), transparent 50%),
                    var(--bg) !important;
        color: var(--ink);
      }

      /* Floating orbs animation */
      .orb { position: fixed; width: 220px; height: 220px; border-radius: 50%; filter: blur(60px); opacity: .18; z-index: 0; }
      .orb.violet { background: #8b5cf6; top: 8%; left: 70%; animation: float1 12s infinite ease-in-out alternate; }
      .orb.cyan { background: #06b6d4; bottom: 8%; left: 8%; animation: float2 14s infinite ease-in-out alternate; }
      @keyframes float1 { from { transform: translateY(-10px) translateX(0); } to { transform: translateY(20px) translateX(10px); } }
      @keyframes float2 { from { transform: translateY(0) translateX(0); } to { transform: translateY(-20px) translateX(20px); } }

      /* Panels */
      .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
      }

      /* Buttons */
      .stButton>button {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: white; border: 0; border-radius: 12px; padding: .6rem 1rem;
        transition: transform .15s ease, box-shadow .2s ease; font-weight: 600;
      }
      .stButton>button:hover { transform: translateY(-1px) scale(1.02); box-shadow: 0 6px 18px rgba(139,92,246,.35); }

      /* Inputs */
      .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div>div, .stFileUploader>div>div {
        background: #0f1420 !important; color: var(--ink) !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.08) !important;
      }

      /* Chat bubbles */
      .bubble { padding: .8rem 1rem; border-radius: 14px; margin: .25rem 0; animation: fadeIn .25s ease-out; }
      .bubble.user { background: linear-gradient(90deg, #1d2534, #1a2030); border: 1px solid rgba(99,102,241,0.25); }
      .bubble.bot { background: linear-gradient(90deg, #141a25, #101520); border: 1px solid rgba(6,182,212,0.25); }
      .role { font-size: .75rem; color: var(--muted); margin-bottom: .2rem; }
      @keyframes fadeIn { from { opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }

      /* Header shimmer */
      .shimmer { background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.22), rgba(255,255,255,0.1));
                 background-size: 200% 100%; animation: shimmer 3s infinite; -webkit-background-clip: text; color: transparent; }
      @keyframes shimmer { from { background-position: 200% 0;} to { background-position: -200% 0;} }

      .small { color: var(--muted); font-size: .85rem; }
    </style>
    <div class="orb violet"></div>
    <div class="orb cyan"></div>
    """,
    unsafe_allow_html=True,
)

# =========================
# UTILITIES (Placeholders to wire your RAG)
# =========================

def run_rag(query: str, doc_bytes: bytes | None, filename: str | None) -> str:
    """Replace this with your real RAG call.
    query: user question
    doc_bytes: raw bytes of uploaded doc (current session)
    filename: filename for any file-type branching you need
    Return: answer string
    """
    # --- BEGIN MOCK ---
    time.sleep(0.5)
    return f"This is a placeholder answer for: '{query}'. Plug in your RAG pipeline here."
    # --- END MOCK ---


def typewriter(text: str, placeholder, speed: float = 0.015):
    """Typing animation: progressively reveal text inside a placeholder.
    Works for markdown, preserves simple ** formatting.
    """
    acc = ""
    for ch in text:
        acc += ch
        placeholder.markdown(acc)
        time.sleep(speed)


# =========================
# SESSION STATE (Multi-session chat)
# =========================
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}  # {session_id: {name, messages, file_name, file_bytes}}

if "current_session" not in st.session_state:
    sid = str(uuid.uuid4())[:8]
    st.session_state.current_session = sid
    st.session_state.all_sessions[sid] = {
        "name": f"Session {sid}",
        "messages": [],  # list of dicts: {role: "user"|"assistant", content: str}
        "file_name": None,
        "file_bytes": None,
    }

# =========================
# SIDEBAR — Session Manager
# =========================
with st.sidebar:
    st.markdown("## 🗂️ Sessions")

    # List sessions
    ids = list(st.session_state.all_sessions.keys())
    labels = [st.session_state.all_sessions[i]["name"] for i in ids]
    idx = ids.index(st.session_state.current_session)
    chosen = st.selectbox("Active session", options=ids, format_func=lambda x: st.session_state.all_sessions[x]["name"], index=idx)
    st.session_state.current_session = chosen

    # Rename session
    new_name = st.text_input("Rename session", value=st.session_state.all_sessions[chosen]["name"])
    st.session_state.all_sessions[chosen]["name"] = new_name or st.session_state.all_sessions[chosen]["name"]

    # Create + Delete buttons
    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ New", use_container_width=True):
            sid = str(uuid.uuid4())[:8]
            st.session_state.all_sessions[sid] = {"name": f"Session {sid}", "messages": [], "file_name": None, "file_bytes": None}
            st.session_state.current_session = sid
    with colB:
        if st.button("🗑️ Delete", type="secondary", use_container_width=True):
            if len(st.session_state.all_sessions) > 1:
                del st.session_state.all_sessions[chosen]
                st.session_state.current_session = list(st.session_state.all_sessions.keys())[0]
            else:
                st.warning("Cannot delete the last remaining session.")

    st.markdown("---")
    # Upload in sidebar so it sticks while chatting
    file = st.file_uploader("📄 Upload document (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], help="Each session keeps its own file")
    if file is not None:
        st.session_state.all_sessions[st.session_state.current_session]["file_name"] = file.name
        st.session_state.all_sessions[st.session_state.current_session]["file_bytes"] = file.read()
        st.success(f"Attached: {file.name}")

    # Export chat
    if st.button("⬇️ Export Chat JSON", use_container_width=True):
        ss = st.session_state.all_sessions[st.session_state.current_session]
        payload = {
            "session": ss["name"],
            "file_name": ss["file_name"],
            "messages": ss["messages"],
        }
        s = StringIO()
        json.dump(payload, s, indent=2)
        st.download_button("Download JSON", s.getvalue(), file_name=f"{ss['name'].replace(' ','_')}.json", mime="application/json")

# =========================
# HEADER
# =========================
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown("# <span class='shimmer'>RAG Chat</span>", unsafe_allow_html=True)
    st.caption("Ask questions about your uploaded document — with sleek dark mode and typing animation.")
with right:
    st.markdown("""
    <div class='glass' style='padding:12px; text-align:center;'>
      <div class='small'>Active Session</div>
      <div style='font-weight:700; margin-top:4px;'>""" + st.session_state.all_sessions[st.session_state.current_session]["name"] + """</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# =========================
# CHAT HISTORY RENDER
# =========================
ss = st.session_state.all_sessions[st.session_state.current_session]

# Container panel for chat
chat_panel = st.container()
with chat_panel:
    st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
    if not ss["messages"]:
        st.markdown("<div class='small'>No messages yet. Upload a document and ask your first question!</div>", unsafe_allow_html=True)
    else:
        for m in ss["messages"]:
            role = m["role"]
            content = m["content"]
            icon = "🙋‍♂️" if role == "user" else "🤖"
            role_lbl = "You" if role == "user" else "Assistant"
            st.markdown(f"<div class='role'>{icon} {role_lbl}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bubble {'user' if role=='user' else 'bot'}'>{content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# =========================
# INPUT AREA (sticky)
# =========================
with st.container():
    st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
    q = st.text_input("Type your question and press Enter", key="q_input", placeholder="e.g., Summarize section 2…")
    cols = st.columns([1,1,1,2,2])
    with cols[0]:
        stream_speed = st.slider("Typing speed", 0.0, 0.06, 0.02, 0.005, help="Lower is faster")
    with cols[1]:
        show_thinking = st.toggle("Show thinking spinner", value=True)
    with cols[2]:
        clear_btn = st.button("🧹 Clear Chat", type="secondary")
    with cols[3]:
        ask_btn = st.button("Ask ✨", use_container_width=True)
    with cols[4]:
        retry_btn = st.button("↻ Retry last", type="secondary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Handle Clear
if clear_btn:
    ss["messages"] = []
    st.rerun()

# Helper: perform inference + streaming render

def answer_and_stream(prompt: str):
    # Persist user msg
    ss["messages"].append({"role": "user", "content": prompt})

    # Thinking spinner
    spinner = st.empty()
    if show_thinking:
        spinner = st.spinner("Thinking with RAG…")
    else:
        class Dummy:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                return False
        spinner = Dummy()

    with spinner:
        # Call your RAG (replace run_rag)
        answer = run_rag(prompt, ss["file_bytes"], ss["file_name"]) if ss["file_bytes"] else "Please upload a document first."
        time.sleep(0.2)

    # Render as typing animation
    st.markdown("<div class='role'>🤖 Assistant</div>", unsafe_allow_html=True)
    ph = st.empty()
    typewriter(answer, ph, speed=stream_speed)

    # Persist assistant msg
    ss["messages"].append({"role": "assistant", "content": answer})


# Handle Ask
if ask_btn and q.strip():
    if ss["file_bytes"] is None:
        st.warning("📎 Please upload a document for this session before asking.")
    else:
        # Render the user's message immediately for snappy UX
        st.markdown("<div class='role'>🙋‍♂️ You</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
        answer_and_stream(q)
        st.stop()

# Handle Retry
if retry_btn:
    # Find last user message
    last_user = None
    for m in reversed(ss["messages"]):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if last_user:
        answer_and_stream(last_user)
        st.stop()

# Enter-to-send support: if user hit Enter in the input (and not ask button), still answer
if q and not ask_btn and ss["file_bytes"] is not None and st.session_state.get("_submitted_once") != q:
    # naive debounce
    st.session_state["_submitted_once"] = q
    st.markdown("<div class='role'>🙋‍♂️ You</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bubble user'>{q}</div>", unsafe_allow_html=True)
    answer_and_stream(q)
