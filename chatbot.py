import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Assistant")

st.write(
    "Upload a PDF document and ask questions about its contents. "
    "The application uses semantic search and an OpenAI language model "
    "to generate answers based on the uploaded document."
)


# ---------------------------------------------------------
# OpenAI configuration
# ---------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning(
        "OPENAI_API_KEY is not configured. "
        "Set the environment variable before using the application."
    )


# ---------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------

def extract_pages_from_pdf(pdf_file):
    """Extract text and page numbers from all readable PDF pages."""

    pdf_reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages.append(
                {
                    "text": page_text,
                    "page": page_number
                }
            )

    return pages


# ---------------------------------------------------------
# Split document into page-aware chunks
# ---------------------------------------------------------

def create_text_chunks(pages):
    """Split PDF pages into chunks while preserving page metadata."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    documents = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page["page"]
                    }
                )
            )

    return documents


# ---------------------------------------------------------
# Create FAISS vector store
# ---------------------------------------------------------

def create_vector_store(documents):
    """Create embeddings and store document chunks in FAISS."""

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


# ---------------------------------------------------------
# Generate an answer
# ---------------------------------------------------------

def generate_answer(vector_store, question):
    """Retrieve relevant chunks and generate a cited answer."""

    relevant_documents = vector_store.similarity_search(
        question,
        k=4
    )

    context_parts = []

    for document in relevant_documents:

        page_number = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"[Page {page_number}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that answers questions about
                an uploaded PDF document.

                Answer the question using only the supplied document context.

                Do not use outside knowledge.

                If the answer cannot be found in the document, say:
                "I could not find that information in the uploaded document."

                Keep the answer clear and concise.
                """
            ),
            (
                "human",
                """
                Document context:

                {context}

                Question:

                {question}
                """
            )
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    source_pages = sorted(
        {
            document.metadata.get("page")
            for document in relevant_documents
            if document.metadata.get("page") is not None
        }
    )

    return response.content, source_pages


# ---------------------------------------------------------
# Document upload
# ---------------------------------------------------------

with st.sidebar:

    st.header("Your Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.caption(
        "Your PDF is processed so you can ask questions about its contents."
    )


# ---------------------------------------------------------
# Process uploaded document
# ---------------------------------------------------------

if uploaded_file is not None:

    if not OPENAI_API_KEY:

        st.error(
            "An OpenAI API key must be configured before "
            "the document can be processed."
        )

        st.stop()

    try:

        with st.spinner("Reading document..."):

            pages = extract_pages_from_pdf(
                uploaded_file
            )

        if not pages:

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()

        documents = create_text_chunks(pages)

        with st.spinner(
            "Creating document search index..."
        ):

            vector_store = create_vector_store(
                documents
            )

        st.success(
            f"Document ready — "
            f"{len(documents)} text chunks indexed "
            f"from {len(pages)} readable pages."
        )


        # -------------------------------------------------
        # User question
        # -------------------------------------------------

        question = st.text_input(
            "Ask a question about your document",
            placeholder="What is this document about?"
        )


        if question:

            with st.spinner(
                "Searching the document..."
            ):

                answer, source_pages = generate_answer(
                    vector_store,
                    question
                )

            st.subheader("Answer")

            st.write(answer)

            if source_pages:

                pages_display = ", ".join(
                    str(page)
                    for page in source_pages
                )

                label = (
                    "Page"
                    if len(source_pages) == 1
                    else "Pages"
                )

                st.caption(
                    f"📚 Sources: {label} {pages_display}"
                )


    except Exception as error:

        st.error(
            "The document could not be processed."
        )

        with st.expander("Technical details"):
            st.code(str(error))


else:

    st.info(
        "Upload a PDF from the sidebar to get started."
    )