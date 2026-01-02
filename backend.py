import os
import requests
import fitz
import pickle
import pytesseract
from PIL import Image
from io import BytesIO
import sqlite3

from typing import TypedDict, Annotated
from sentence_transformers import SentenceTransformer

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS


# ===============================
# WINDOWS OCR SETUP
# ===============================
TESSERACT_DIR = r"C:\Users\Ritesh\AppData\Local\Programs\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = os.path.join(
    TESSERACT_DIR, "tesseract.exe"
)
os.environ["PATH"] = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")

# ===============================
# PATHS
# ===============================
DB_DIR = "faiss_db"
INDEX_PATH = os.path.join(DB_DIR, "index.faiss")
TEXTS_PATH = os.path.join(DB_DIR, "texts.pkl")
SQLITE_PATH = "chatbot.db"

# ===============================
# OLLAMA CHECK
# ===============================
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

# ===============================
# EMBEDDINGS (LangChain compatible)
# ===============================
class SBERTEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# ===============================
# BUILD / LOAD MULTIMODAL INDEX
# ===============================
def build_or_load_vectorstore(pdf_dir):
    embeddings = SBERTEmbeddings()

    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        with open(TEXTS_PATH, "rb") as f:
            texts = pickle.load(f)
        return FAISS.from_texts(texts, embeddings)

    texts = []
    os.makedirs(DB_DIR, exist_ok=True)

    for pdf in os.listdir(pdf_dir):
        if not pdf.lower().endswith(".pdf"):
            continue

        doc = fitz.open(os.path.join(pdf_dir, pdf))

        for page in doc:
            text = page.get_text()
            if text.strip():
                texts.append(text)

            for img in page.get_images(full=True):
                base = doc.extract_image(img[0])
                image = Image.open(BytesIO(base["image"]))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    texts.append(ocr_text)

    if not texts:
        raise ValueError("No multimodal content extracted")

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    return FAISS.from_texts(texts, embeddings)

# ===============================
# LANGGRAPH STATE
# ===============================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===============================
# RAG + MEMORY NODE
# ===============================
def chat_node(state: ChatState):
    messages = state["messages"]

    # Latest user query
    user_query = messages[-1].content

    # ---- CHAT MEMORY (LAST N TURNS) ----
    chat_history = []
    for msg in messages[:-1][-6:]:  # last 3 turns
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        chat_history.append(f"{role}: {msg.content}")

    chat_history_text = "\n".join(chat_history)

    # ---- DOCUMENT RETRIEVAL ----
    docs = retriever.invoke(user_query)
    context = "\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
                """
        You are a helpful assistant.

        Use CHAT MEMORY first.
        Use DOCUMENT CONTEXT if relevant.
        If the answer is not found in either, say:
        "Not available."

        CHAT MEMORY:
        {chat_history}

        DOCUMENT CONTEXT:
        {context}

        Question:
        {question}
        """
            )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "chat_history": chat_history_text,
        "context": context,
        "question": user_query
    })

    return {"messages": [AIMessage(content=response)]}


# ===============================
# INITIALIZATION
# ===============================
vectorstore = build_or_load_vectorstore("data/pdfs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(
    model="llama2:7b",
    base_url="http://localhost:11434",
    temperature=0,
    num_ctx=1024,
    num_predict=256
)

# ===============================
# SQLITE CHECKPOINTER (THREAD MEMORY)
# ===============================
conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ===============================
# LANGGRAPH BUILD
# ===============================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ===============================
# THREAD RETRIEVAL (FOR SIDEBAR)
# ===============================
def retrieve_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )
    return list(threads)
