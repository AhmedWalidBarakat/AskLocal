"""
AskLocal - a fully local RAG chat UI.
Retrieves relevant chunks from a local Chroma DB and asks a local Ollama
model to answer using only that context. Nothing leaves the machine.
"""

import os
from collections import Counter

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

DB_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

PROMPT = ChatPromptTemplate.from_template(
    """Use the following context to answer the question.
If the answer isn't in the context, say you don't know — do not make something up.

Context:
{context}

Question: {question}

Answer:"""
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --al-accent: #4f7cff;
    --al-accent-soft: rgba(79, 124, 255, 0.12);
}

h1, h2, h3, [data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.al-hero {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.1rem;
}
.al-hero .al-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--al-accent), #7c3aed);
    font-size: 1.3rem;
}
.al-hero h1 {
    margin: 0 !important;
    font-size: 1.9rem !important;
    letter-spacing: -0.02em;
}
.al-tagline {
    color: var(--al-accent);
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0.2rem 0 1.1rem 0;
}

.al-stat {
    background: var(--al-accent-soft);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.55rem;
}
.al-stat .al-stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
}
.al-stat .al-stat-value {
    font-size: 0.95rem;
    font-weight: 600;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}
</style>
"""


@st.cache_resource
def load_chain():
    if not os.path.isdir(DB_FOLDER):
        st.error("No vector database found. Run `python ingest.py` first.")
        st.stop()

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    def answer(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = llm.invoke(PROMPT.format(context=context, question=question))
        return response.content, docs

    return answer, vectordb


def library_stats(vectordb):
    """Chunk count and per-file breakdown, for the sidebar."""
    data = vectordb.get()
    sources = [os.path.basename(m.get("source", "unknown")) for m in data["metadatas"]]
    return len(sources), Counter(sources)


st.set_page_config(page_title="AskLocal", page_icon="🔒", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

answer_question, vectordb = load_chain()
chunk_count, per_file = library_stats(vectordb)

with st.sidebar:
    st.markdown("## 🔒 AskLocal")
    st.caption("Private document Q&A — 100% on-device")

    st.markdown(
        f"""<div class="al-stat">
        <div class="al-stat-label">Documents indexed</div>
        <div class="al-stat-value">{len(per_file)}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="al-stat">
        <div class="al-stat-label">Chunks in vector store</div>
        <div class="al-stat-value">{chunk_count}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="al-stat">
        <div class="al-stat-label">Chat model</div>
        <div class="al-stat-value">{CHAT_MODEL} (Ollama)</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="al-stat">
        <div class="al-stat-label">Embedding model</div>
        <div class="al-stat-value">{EMBED_MODEL} (Ollama)</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if per_file:
        with st.expander("Source files"):
            for name, count in sorted(per_file.items()):
                st.markdown(f"- {name} · {count} chunks")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """<div class="al-hero"><div class="al-badge">🔒</div><h1>AskLocal</h1></div>
    <div class="al-tagline">Ask your documents anything. Nothing leaves your machine.</div>""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🔒"):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔒"):
        with st.spinner("Searching documents and generating an answer..."):
            answer, sources = answer_question(question)
            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        src = os.path.basename(doc.metadata.get("source", "unknown"))
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"- **{src}**, page {page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
