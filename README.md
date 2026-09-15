# AI PDF Assistant

An AI-powered PDF question-answering application that uses Retrieval-Augmented Generation (RAG) to answer natural-language questions based on the contents of an uploaded PDF document.

The application extracts text from a PDF, divides the document into manageable chunks, generates vector embeddings, stores them in a FAISS vector index, retrieves the most relevant document sections for a user's question, and uses an OpenAI language model to generate a document-grounded response.

## Features

- Upload and process PDF documents
- Extract text from PDF pages
- Split documents into overlapping text chunks
- Generate semantic embeddings with OpenAI
- Store and search document embeddings with FAISS
- Retrieve relevant document context using similarity search
- Ask natural-language questions about uploaded documents
- Generate answers grounded in retrieved document content
- Interactive web interface built with Streamlit
- Secure API-key handling through environment variables

## Architecture

```text
PDF Upload
    ↓
Text Extraction (PyPDF2)
    ↓
Text Chunking (LangChain)
    ↓
OpenAI Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Relevant Document Context
    ↓
OpenAI Language Model
    ↓
Document-Grounded Answer
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
2. PyPDF2 extracts readable text from the document.
3. LangChain's `RecursiveCharacterTextSplitter` divides the text into overlapping chunks.
4. OpenAI embeddings convert the chunks into vector representations.
5. FAISS stores the vectors and provides similarity search.
6. When the user submits a question, FAISS retrieves the most relevant portions of the document.
7. The retrieved context and question are passed to an OpenAI language model.
8. The model generates an answer based on the retrieved document context.

## Installation

Clone the repository:

```bash
git clone https://github.com/denistita/chatbot.git
cd chatbot
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```powershell
.\.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## OpenAI API Configuration

This application requires an OpenAI API key.

Set the API key as an environment variable.

### PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### macOS/Linux

```bash
export OPENAI_API_KEY="your-api-key"
```

Never commit API keys or other credentials to the repository.

## Run the Application

Start the Streamlit application:

```bash
streamlit run chatbot.py
```

Then open the local Streamlit address displayed in the terminal.

## Project Structure

```text
chatbot/
├── chatbot.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local development files such as `.venv/`, `.idea/`, environment files, API keys, and uploaded documents are excluded from version control.

## Security

API credentials are read from environment variables rather than stored in source code.

Uploaded documents and sensitive or proprietary data should not be committed to this repository.

## Future Improvements

Potential enhancements include:

- Support for multiple PDF documents
- Persistent vector indexes
- Conversational question history
- Source citations and page references
- Improved document metadata handling
- Additional document formats
- Deployment to a hosted environment

## Author

**Denis Tita**

Software Engineer & DevSecOps Engineer

- Portfolio: denistita.com
- LinkedIn: linkedin.com/in/denistita
- GitHub: github.com/denistita