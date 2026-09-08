import streamlit as st
from multimodel import (
    ask_multimodal_rag,
    vector_store,
    process_uploaded_files,
    create_uploaded_knowledge_base
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0F172A;
    color: #F8FAFC;
}

h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

p, label {
    color: #CBD5E1 !important;
}

textarea {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

textarea:focus {
    border: 1px solid #2DD4BF !important;
    box-shadow: 0 0 0 1px #2DD4BF !important;
}

.stButton > button {
    background-color: #2DD4BF;
    color: #0F172A;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
}

.stButton > button:hover {
    background-color: #5EEAD4;
    color: #0F172A;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] h2 {
    color: #A78BFA !important;
}

[data-testid="stSidebar"] p {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #E2E8F0 !important;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Data Research Assistant",
    page_icon="🔎",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🔎 AI Data Research Assistant")

st.write(
    "Ask questions about the documents in your RAG knowledge base."
)

uploaded_files = st.file_uploader(
    "📤 Upload documents or images",
    type=["txt", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    uploaded_vector_store = create_uploaded_knowledge_base(uploaded_files)

    st.success(
        f"Successfully processed {len(uploaded_files)} file(s)."
    )


with st.sidebar:
    st.header("🧠 RAG Pipeline")
    st.write("1. 📄 Text + Image Documents")
    st.write("2. 🔎 Multimodal Retrieval")
    st.write("3. 🔀 RRF Fusion")
    st.write("4. 🎯 LLM Reranking")
    st.write("5. 🤖 Final LLM Answer")
    st.write("6. 📚 Source Attribution")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


example_question = st.selectbox(
    "💡 Try an example question:",
    [
        "Select a question",
        "What is this document about?",
        "What are the key points discussed?",
        "What information is shown in the image?",
        "What are the main concepts discussed?",
        "How are the text and image related?"
    ],
    key="example_question"
)

if "question" not in st.session_state:
    st.session_state.question = ""

if example_question != "Select a question":
    st.session_state.question = example_question

if st.session_state.chat_history:
    st.subheader("💬 Conversation")

    for item in st.session_state.chat_history:
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**AI:** {item['answer']}")
        
question = st.text_area(
    "Enter your question:",
    placeholder="Type your own question here...",
    height=120,
    key="question"
)



if st.button("Ask AI", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            active_vector_store = uploaded_vector_store if uploaded_files else vector_store

            answer, sources = ask_multimodal_rag(
                question,
                active_vector_store,
                st.session_state.chat_history
            )

        st.write("### Answer")

        st.write(answer)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        st.write("### 📚 Sources")

        for source in sources:
            st.write(f"📄 {source}")