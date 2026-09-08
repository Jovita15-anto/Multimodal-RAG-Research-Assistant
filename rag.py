from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama


folder = Path("Data_RAG")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def ask_rag(question):

    document_stores = {}

    for file_path in folder.glob("*.txt"):

        document = Document(
            page_content=file_path.read_text(encoding="utf-8"),
            metadata={"source": file_path.name}
        )

        chunks = splitter.split_documents([document])

        vector_store = InMemoryVectorStore(
            embedding=embeddings
        )

        vector_store.add_documents(chunks)

        document_stores[file_path.name] = vector_store


    all_retrieved_docs = []

    for filename, vector_store in document_stores.items():

        docs_with_scores = vector_store.similarity_search_with_score(
            question,
            k=2
        )

        for doc, score in docs_with_scores:

            all_retrieved_docs.append(
                (doc, score)
            )


    all_retrieved_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )


    top_k_docs = all_retrieved_docs[:4]


    context = "\n\n".join(
        doc.page_content
        for doc, score in top_k_docs
    )


    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )


    prompt = f"""
You are a research assistant.

Answer the question using ONLY the supplied context.

Question:
{question}

Context:
{context}

Instructions:
- Use information from all relevant documents.
- Do not invent information.
- Mention the source document for important claims.
- Combine information across documents into one answer.
"""


    response = llm.invoke(prompt)

    sources = list(dict.fromkeys(
        doc.metadata["source"]
        for doc, score in top_k_docs
    ))

    return response.content, sources