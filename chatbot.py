import os
import html

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(20, 184, 166, 0.08),
            transparent 30%
        ),
        #07111f;
    color: #f8fafc;
}

.main .block-container {
    max-width: 1280px;
    padding-top: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 6rem;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: #0b1728;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] p {
    color: #cbd5e1;
}


/* Hero */

.app-badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(45,212,191,0.35);
    background: rgba(20,184,166,0.08);
    color: #5eead4;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}

.hero-title {
    font-size: clamp(2.5rem, 5vw, 4.4rem);
    line-height: 1.03;
    font-weight: 800;
    letter-spacing: -0.045em;
    margin: 0;
    color: #f8fafc;
}

.hero-title span {
    color: #5eead4;
}

.hero-description {
    max-width: 900px;
    margin-top: 1.2rem;
    margin-bottom: 1.5rem;
    color: #b7c3d4;
    font-size: 1.02rem;
    line-height: 1.7;
}


/* Feature chips */

.feature-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 2.2rem;
}

.feature-chip {
    padding: 0.43rem 0.75rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    color: #dbe5f0;
    font-size: 0.82rem;
}


/* Document cards */

.document-card {
    padding: 1rem 1.15rem;
    margin: 0.5rem 0 1.4rem 0;
    border-radius: 12px;
    border: 1px solid rgba(45,212,191,0.28);
    background: rgba(20,184,166,0.06);
}

.document-label {
    color: #5eead4;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.document-name {
    color: #ffffff;
    font-weight: 600;
    font-size: 0.95rem;
    word-break: break-word;
}


/* Initial state */

.start-state {
    padding: 2.5rem 2rem;
    margin: 1rem 0 1.5rem 0;
    border-radius: 16px;
    border: 1px dashed rgba(45,212,191,0.28);
    background: rgba(255,255,255,0.018);
}

.start-icon {
    color: #5eead4;
    font-size: 1.7rem;
    margin-bottom: 0.7rem;
}

.start-title {
    color: #f8fafc;
    font-size: 1.12rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.start-text {
    max-width: 700px;
    color: #aab8ca;
    font-size: 0.94rem;
    line-height: 1.65;
}

.start-text strong {
    color: #f1f5f9;
}


/* Empty chat */

.empty-state {
    padding: 2.2rem 1.5rem;
    margin: 1rem 0 1.5rem 0;
    text-align: center;
    border-radius: 16px;
    border: 1px dashed rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.02);
}

.empty-state-title {
    color: #f8fafc;
    font-weight: 700;
    font-size: 1.08rem;
    margin-bottom: 0.45rem;
}

.empty-state-text {
    max-width: 650px;
    margin: 0 auto;
    color: #aab8ca;
    font-size: 0.92rem;
    line-height: 1.6;
}


/* Chat */

[data-testid="stChatMessage"] {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.035);
    border-radius: 14px;
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.8rem;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #f8fafc !important;
    opacity: 1 !important;
}

[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"] p {
    color: #f8fafc !important;
    opacity: 1 !important;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] strong {
    color: #ffffff !important;
}

[data-testid="stChatMessage"] a {
    color: #5eead4 !important;
}


/* Source citations */

[data-testid="stChatMessage"]
[data-testid="stCaptionContainer"] {
    margin-top: 0.45rem;
}

[data-testid="stChatMessage"]
[data-testid="stCaptionContainer"] p {
    color: #a5b4c7 !important;
    font-size: 0.82rem;
    opacity: 1 !important;
}


/* Chat input */

[data-testid="stChatInput"] {
    border-color: rgba(45,212,191,0.40);
}

[data-testid="stChatInput"] textarea {
    color: #0f172a !important;
    background: #f8fafc !important;
    caret-color: #0f172a !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: #f1f5f9;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #2dd4bf;
    color: #5eead4;
    background: rgba(20,184,166,0.08);
}


/* Metrics */

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] {
    color: #aab8ca !important;
}


/* Sidebar notes */

.sidebar-note {
    color: #9ba9bc;
    font-size: 0.78rem;
    line-height: 1.55;
}

.sidebar-note strong {
    color: #dbe5f0;
}


/* Misc */

hr {
    border-color: rgba(255,255,255,0.08);
}

[data-testid="stStatusWidget"] p {
    color: #f1f5f9 !important;
}

footer {
    visibility: hidden;
}


/* Responsive */

@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    .hero-title {
        font-size: 2.5rem;
    }

    .hero-description {
        font-size: 0.95rem;
    }

    .start-state {
        padding: 1.5rem;
    }
}
</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="app-badge">Conversational RAG</div>
<h1 class="hero-title">Ask your documents.<br><span>Get grounded answers.</span></h1>
<div class="hero-description">Upload a PDF and explore it conversationally. AI PDF Assistant uses semantic retrieval and Retrieval-Augmented Generation to answer questions from your document while preserving source-page context.</div>
<div class="feature-row"><span class="feature-chip">Semantic Search</span><span class="feature-chip">FAISS Vector Retrieval</span><span class="feature-chip">Conversational Context</span><span class="feature-chip">Page Citations</span><span class="feature-chip">Document Grounding</span></div>
""",
    unsafe_allow_html=True
)


# =========================================================
# OPENAI CONFIGURATION
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        OPENAI_API_KEY = None

st.write("Secret names detected:", list(st.secrets.keys()))

if not OPENAI_API_KEY:
    st.warning(
        "OPENAI_API_KEY is not configured. "
        "Set the environment variable before using the application."
    )


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = None


# =========================================================
# PDF PROCESSING
# =========================================================

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


def create_vector_store(documents):
    """Create embeddings and store chunks in FAISS."""

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


# =========================================================
# CONVERSATION
# =========================================================

def build_conversation_history(messages):
    """Create concise conversation history."""

    history = []

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


def generate_answer(
    vector_store,
    question,
    messages
):
    """
    Retrieve relevant PDF chunks and generate a contextual,
    document-grounded answer with source pages.
    """

    conversation_history = build_conversation_history(
        messages
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    # Rewrite contextual follow-up questions into
    # standalone retrieval queries.
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

    rewrite_chain = rewrite_prompt | llm

    rewritten_response = rewrite_chain.invoke(
        {
            "history": conversation_history,
            "question": question
        }
    )

    search_question = (
        rewritten_response.content.strip()
    )

    # Semantic retrieval.
    relevant_documents = (
        vector_store.similarity_search(
            search_question,
            k=4
        )
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

    # Generate grounded response.
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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📄 AI PDF Assistant")

    st.caption(
        "Document-grounded conversational AI"
    )

    st.divider()

    st.markdown("### Upload document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-note">Drag a PDF into the box above or select <strong>Browse files</strong>. After the document is processed, the chat interface will be ready.</div>',
        unsafe_allow_html=True
    )

    # Only show document controls once a document
    # has actually been processed.
    if st.session_state.document_name:

        st.divider()

        safe_document_name = html.escape(
            st.session_state.document_name
        )

        st.markdown(
            f'<div class="document-card"><div class="document-label">Active document</div><div class="document-name">{safe_document_name}</div></div>',
            unsafe_allow_html=True
        )

        if st.session_state.document_stats:

            stats = st.session_state.document_stats

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Pages",
                    stats["pages"]
                )

            with col2:
                st.metric(
                    "Chunks",
                    stats["chunks"]
                )

        if st.button(
            "Clear conversation",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-note"><strong>Powered by</strong><br>OpenAI · LangChain · FAISS · Streamlit</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    if not OPENAI_API_KEY:

        st.error(
            "An OpenAI API key must be configured before "
            "the document can be processed."
        )

        st.stop()

    if (
        st.session_state.vector_store is None
        or st.session_state.document_name
        != uploaded_file.name
    ):

        try:

            with st.status(
                "Preparing your document...",
                expanded=True
            ) as status:

                st.write(
                    "Extracting PDF text..."
                )

                pages = extract_pages_from_pdf(
                    uploaded_file
                )

                if not pages:

                    status.update(
                        label="Document could not be read.",
                        state="error"
                    )

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.stop()

                st.write(
                    f"Found {len(pages)} readable pages."
                )

                st.write(
                    "Creating page-aware text chunks..."
                )

                documents = create_text_chunks(
                    pages
                )

                st.write(
                    f"Created {len(documents)} searchable chunks."
                )

                st.write(
                    "Generating embeddings and building "
                    "the FAISS index..."
                )

                st.session_state.vector_store = (
                    create_vector_store(
                        documents
                    )
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.document_stats = {
                    "pages": len(pages),
                    "chunks": len(documents)
                }

                # New document = new conversation.
                st.session_state.messages = []

                status.update(
                    label="Document ready",
                    state="complete",
                    expanded=False
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


    # =====================================================
    # DOCUMENT READY
    # =====================================================

    safe_uploaded_name = html.escape(
        uploaded_file.name
    )

    st.markdown(
        f'<div class="document-card"><div class="document-label">Ready for questions</div><div class="document-name">📄 {safe_uploaded_name}</div></div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DISPLAY CONVERSATION
    # =====================================================

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

                source_pages = (
                    message["sources"]
                )

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
                    f"📚 Source context · "
                    f"{label} {pages_display}"
                )


    # =====================================================
    # DOCUMENT READY / EMPTY CONVERSATION
    # =====================================================

    if not st.session_state.messages:

        st.markdown(
            '<div class="empty-state"><div class="empty-state-title">Your document is ready</div><div class="empty-state-text">Ask a question below. You can request a summary, explore specific details, or ask contextual follow-up questions.</div></div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask a question about this document..."
    )

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        previous_messages = (
            st.session_state.messages.copy()
        )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document context..."
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
                            f"📚 Source context · "
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


# =========================================================
# INITIAL STATE — NO DOCUMENT
# =========================================================

else:

    st.markdown(
        '<div class="start-state"><div class="start-icon">← 📄</div><div class="start-title">Start with a document</div><div class="start-text">Use the <strong>Upload document</strong> panel in the sidebar to drag in a PDF or select <strong>Browse files</strong>. Once processing is complete, this area becomes your conversational document workspace.</div></div>',
        unsafe_allow_html=True
    )