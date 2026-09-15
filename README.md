# AI PDF Assistant

A deployed conversational AI application that uses **Retrieval-Augmented Generation (RAG)** to let users upload PDF documents and ask natural-language questions about their content.

The application extracts text while preserving page information, divides the document into overlapping chunks, generates vector embeddings, stores them in a FAISS vector index, and retrieves relevant document context for each question. An OpenAI language model then generates a response grounded in the retrieved content, with source-page citations.

## Live Demo

**Try the deployed application:**  
https://denistita-ai-pdf-assistant.streamlit.app/

## Features

- Upload and process PDF documents
- Page-aware PDF text extraction
- Overlapping text chunking with page metadata
- OpenAI vector embeddings
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- Natural-language document question answering
- Conversational follow-up questions
- Context-aware query rewriting
- Document-grounded responses
- Source-page citations
- Conversation history within the active session
- Responsive Streamlit web interface
- Secure API-key handling using environment variables and Streamlit Secrets
- Public deployment with Streamlit Community Cloud

## Architecture

```text
PDF Upload
    ↓
Page-Aware Text Extraction
    ↓
Overlapping Text Chunks + Page Metadata
    ↓
OpenAI Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Conversation-Aware Query Rewriting
    ↓
Semantic Similarity Search
    ↓
Relevant Document Context
    ↓
OpenAI Language Model
    ↓
Grounded Answer + Source Page Citations
```

## Technology Stack

- **Python**
- **Streamlit**
- **OpenAI API**
- **LangChain**
- **FAISS**
- **PyPDF2**
- **Retrieval-Augmented Generation (RAG)**
- **Vector Embeddings**
- **Semantic Search**

## How It Works

1. The user uploads a PDF through the Streamlit interface.

2. PyPDF2 extracts readable text from each page while preserving page numbers.

3. LangChain's `RecursiveCharacterTextSplitter` divides page content into overlapping chunks.

4. Each chunk is stored with page metadata so retrieved information can be traced back to its source page.

5. OpenAI embeddings convert the document chunks into vector representations.

6. FAISS creates an in-memory vector index for semantic similarity search.

7. When the user asks a question, recent conversation history is used to rewrite contextual follow-up questions into standalone retrieval queries.

8. FAISS retrieves the document chunks most semantically relevant to the question.

9. The retrieved document context, conversation context, and current question are passed to an OpenAI language model.

10. The model is instructed to answer using only the supplied document context.

11. The application displays the generated response together with the source pages associated with the retrieved context.

## Document Grounding

The application is designed to reduce unsupported answers by constraining generation to retrieved PDF content.

Conversation history is used to understand follow-up questions and references, but it is not treated as an additional factual source.

If the requested information cannot be found in the retrieved document context, the assistant is instructed to state that the information could not be found in the uploaded document.

## Local Installation

Clone the repository:

```bash
git clone https://github.com/denistita/ai-pdf-assistant.git
cd ai-pdf-assistant
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Install the required dependencies.

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS/Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## OpenAI API Configuration

The application requires an OpenAI API key.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### macOS/Linux

```bash
export OPENAI_API_KEY="your-api-key"
```

Never commit API keys or other credentials to the repository.

For the hosted version, the API key is configured securely using **Streamlit Secrets** rather than being stored in source code.

## Run Locally

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m streamlit run chatbot.py
```

### macOS/Linux

```bash
streamlit run chatbot.py
```

Streamlit will provide a local address that can be opened in a web browser.

## Project Structure

```text
ai-pdf-assistant/
├── chatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local development files such as `.venv/`, `.idea/`, `.vscode/`, environment files, API keys, and uploaded documents are excluded from version control.

## Security

API credentials are never stored directly in the application source code.

For local development, credentials can be supplied through environment variables. The deployed application supports Streamlit's encrypted Secrets configuration.

Uploaded documents are processed by the running application and are not intended to be committed to this repository.

Sensitive, confidential, or proprietary documents should not be added to the public repository.

## Deployment

The application is deployed using **Streamlit Community Cloud** from the `main` branch of this repository.

Live application:

https://denistita-ai-pdf-assistant.streamlit.app/

Updates pushed to the deployed GitHub branch can be picked up by the Streamlit deployment.

## Future Improvements

Potential enhancements include:

- Multiple-document conversations
- Persistent or cached vector indexes
- Streaming AI responses
- More detailed citation snippets
- Hybrid search and reranking
- Additional document formats
- Improved document metadata filtering
- Automated RAG evaluation and groundedness testing
- Docker containerization
- CI/CD validation and automated testing
- Usage and cost controls for public deployments

## Author

**Denis Tita**

Software Engineer & DevSecOps Engineer

- Portfolio: https://denistita.com
- LinkedIn: https://www.linkedin.com/in/denistita/
- GitHub: https://github.com/denistita