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
    "Upload a PDF and have a conversation about its contents. "
    "The assistant uses Retrieval-Augmented Generation (RAG) "
    "to retrieve relevant information and generate "
    "document-grounded answers with source pages."
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
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ---------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------

def extract_pages_from_pdf(pdf_file):
    """Extract text and page numbers from readable PDF pages."""

    pdf_reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(
        pdf_reader.pages,
        start=1
    ):
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
    """Split pages into chunks while preserving page metadata."""

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
    """Create embeddings and store chunks in FAISS."""

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


# ---------------------------------------------------------
# Build conversation history
# ---------------------------------------------------------

def build_conversation_history(messages):
    """Create concise conversation history for follow-up questions."""

    history = []

    # Limit history so prompts do not grow indefinitely.
    for message in messages[-6:]:

        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        history.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(history)


# ---------------------------------------------------------
# Generate answer
# ---------------------------------------------------------

def generate_answer(
    vector_store,
    question,
    messages
):
    """
    Retrieve relevant chunks and generate a contextual,
    document-grounded answer with source pages.
    """

    conversation_history = build_conversation_history(
        messages
    )

    # First rewrite contextual follow-up questions into
    # standalone questions for better vector retrieval.
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Rewrite the user's latest question as a standalone
                search question using the conversation history.

                Do not answer the question.

                If the question is already standalone, return it
                without changing its meaning.
                """
            ),
            (
                "human",
                """
                Conversation history:

                {history}

                Latest question:

                {question}

                Standalone search question:
                """
            )
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    rewrite_chain = rewrite_prompt | llm

    rewritten_response = rewrite_chain.invoke(
        {
            "history": conversation_history,
            "question": question
        }
    )

    search_question = rewritten_response.content.strip()

    # Retrieve document chunks using the standalone question.
    relevant_documents = vector_store.similarity_search(
        search_question,
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

    # Generate final answer using retrieved context and
    # conversation history.
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI assistant that answers questions
                about an uploaded PDF document.

                Use only the supplied document context to provide
                factual information about the document.

                Conversation history may be used to understand
                references and follow-up questions, but it must
                not be treated as an additional factual source.

                Do not use outside knowledge.

                If the answer cannot be found in the supplied
                document context, say:

                "I could not find that information in the
                uploaded document."

                Keep answers clear and concise.
                """
            ),
            (
                "human",
                """
                Conversation history:

                {history}

                Document context:

                {context}

                Current question:

                {question}
                """
            )
        ]
    )

    answer_chain = answer_prompt | llm

    response = answer_chain.invoke(
        {
            "history": conversation_history,
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
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Your Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.caption(
        "Your PDF is processed so you can ask questions "
        "about its contents."
    )

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


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

    # Only rebuild embeddings when a different PDF is uploaded.
    if (
        st.session_state.vector_store is None
        or st.session_state.document_name
        != uploaded_file.name
    ):

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

            documents = create_text_chunks(
                pages
            )

            with st.spinner(
                "Creating document search index..."
            ):

                st.session_state.vector_store = (
                    create_vector_store(documents)
                )

            st.session_state.document_name = (
                uploaded_file.name
            )

            # A new document starts a new conversation.
            st.session_state.messages = []

            st.success(
                f"Document ready — "
                f"{len(documents)} text chunks indexed "
                f"from {len(pages)} readable pages."
            )

        except Exception as error:

            st.error(
                "The document could not be processed."
            )

            with st.expander(
                "Technical details"
            ):
                st.code(str(error))

            st.stop()


    # -----------------------------------------------------
    # Display conversation
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                source_pages = message["sources"]

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
                    f"📚 Sources: "
                    f"{label} {pages_display}"
                )


    # -----------------------------------------------------
    # Chat input
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask a question about your document"
    )

    if question:

        # Display and save user message.
        with st.chat_message("user"):
            st.markdown(question)

        # Capture history before adding the current
        # question so it is not duplicated.
        previous_messages = (
            st.session_state.messages.copy()
        )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate assistant response.
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    answer, source_pages = (
                        generate_answer(
                            st.session_state.vector_store,
                            question,
                            previous_messages
                        )
                    )

                    st.markdown(answer)

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
                            f"📚 Sources: "
                            f"{label} {pages_display}"
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_pages
                        }
                    )

                except Exception as error:

                    st.error(
                        "The question could not be processed."
                    )

                    with st.expander(
                        "Technical details"
                    ):
                        st.code(str(error))


else:

    st.info(
        "Upload a PDF from the sidebar to start chatting."
    )