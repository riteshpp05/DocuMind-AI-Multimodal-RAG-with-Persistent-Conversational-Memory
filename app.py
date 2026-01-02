import streamlit as st
import uuid
from backend import chatbot, retrieve_all_threads, check_ollama
from langchain_core.messages import HumanMessage, AIMessage

# ==============================
# Utility functions
# ==============================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    if new_thread not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(new_thread)
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    """
    Load conversation safely from LangGraph SQLite memory
    """
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    if hasattr(state, "values") and "messages" in state.values:
        messages = state.values["messages"]
        parsed = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parsed.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                parsed.append({"role": "assistant", "content": msg.content})
        return parsed
    return []

def get_conversation_title(thread_id):
    """
    Title = first user message (cached)
    """
    if thread_id in st.session_state["thread_titles"]:
        return st.session_state["thread_titles"][thread_id]

    messages = load_conversation(thread_id)
    if messages:
        first = messages[0]["content"]
        title = first[:25] + "..." if len(first) > 25 else first
    else:
        title = "Untitled Chat"

    st.session_state["thread_titles"][thread_id] = title
    return title


# ==============================
# Streamlit setup
# ==============================
st.set_page_config(
    page_title="Multimodal RAG Chat",
    layout="wide"
)

if not check_ollama():
    st.error("❌ Ollama is not running. Start it first.")
    st.stop()

# ==============================
# Session State Init
# ==============================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Ensure current thread exists
if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(st.session_state["thread_id"])


# ==============================
# Sidebar UI (Threads)
# ==============================
st.sidebar.title("GuideAI 🤖")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.markdown("### 📂 My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    title = get_conversation_title(tid)
    is_active = (tid == st.session_state["thread_id"])

    if st.sidebar.button(
        f"💬 {title}",
        key=f"thread-{tid}",
        use_container_width=True
    ):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = load_conversation(tid)


# ==============================
# Main Chat UI
# ==============================
st.title("📄 Multimodal RAG Chat")

# Display existing messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask from your PDFs (text + images)...")

if user_input:
    # Show user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Run LangGraph chatbot
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                full_response += chunk.content
                placeholder.markdown(full_response)

    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )
