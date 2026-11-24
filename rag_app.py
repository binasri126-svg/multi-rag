import streamlit as st
import tempfile
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.title("Multi-PDF RAG App")

files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if files and st.button("Index PDFs"):
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    for f in files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp:
            temp.write(f.getvalue())
            temp_path = temp.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = f.name
            page.metadata["page"] += 1

        chunks.extend(splitter.split_documents(pages))

        os.remove(temp_path)

    if chunks:
        st.session_state.store = FAISS.from_documents(
            chunks, OpenAIEmbeddings()
        )

        st.session_state.chunk_count = len(chunks)
        st.session_state.file_names = [f.name for f in files]
    else:
        st.warning("No readable text found in the PDFs.")

if "store" in st.session_state:
    st.write(
        "Uploaded:",
        ", ".join(st.session_state.file_names)
    )

    st.success(
        f"Indexed {st.session_state.chunk_count} chunks"
    )

    question = st.text_input("Ask a question")

    if question and st.button("Get Answer"):
        docs = st.session_state.store.as_retriever(
            search_kwargs={"k": 4}
        ).invoke(question)

        context = ""

        for i, d in enumerate(docs, start=1):
            context += (
                f"\nSource ID: {i}\n"
                f"File: {d.metadata['source']}\n"
                f"Page: {d.metadata['page']}\n"
                f"{d.page_content}\n"
            )

        prompt = f"""
Answer the question using only the context below.

If the documents disagree, clearly explain the
disagreement and identify both documents.

If the answer is not available, say so.

Select only the source IDs that directly support
your answer. Do not select unrelated sources.

Do not include a separate source list in the answer.

Return a JSON object with two fields:
answer: the answer to the question
source_ids: a list of supporting source IDs

If no answer is available, return an empty source_ids list.

Context:
{context}

Question:
{question}
"""

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ).bind(response_format={"type": "json_object"})

        response = llm.invoke(prompt)
        result = json.loads(response.content)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")

        sources = set()

        for source_id in result["source_ids"]:
            if isinstance(source_id, int) and 1 <= source_id <= len(docs):
                d = docs[source_id - 1]

                source = (
                    f"{d.metadata['source']}"
                    f" - page {d.metadata['page']}"
                )

                if source not in sources:
                    st.caption(source)
                    sources.add(source)

        if not sources:
            st.caption("No supporting sources found.")
            