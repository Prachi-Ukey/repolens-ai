# RepoLens AI — GitHub Repository Intelligence Platform

RepoLens AI is an AI-powered platform that helps developers understand public GitHub repositories by asking questions in natural language. It combines Retrieval-Augmented Generation (RAG), semantic search, code parsing, and a locally running language model to generate repository-grounded answers with file and line references.

Instead of manually searching through hundreds of files, developers can ask questions about a codebase and use the retrieved evidence to understand its purpose, implementation, and execution flow.

## 📸 Application Screenshots

### Dashboard
![RepoLens AI Dashboard](screenshots/dashboardpage.png)

### Repository Analysis
![Repository Analysis](screenshots/repository-analysis.png)

### AI Chat with Code References
![AI Chat Response](screenshots/ai-chat.png)

## Key Features

- **GitHub Repository Analysis** — Analyze public GitHub repositories to explore their source code and documentation.
- **AST-Aware Code Parsing** — Organize supported source code into meaningful chunks while preserving file paths, symbols, and line numbers.
- **RAG-Based Retrieval** — Retrieve relevant repository content using semantic and keyword-based search.
- **AI-Powered Codebase Chat** — Ask natural-language questions about repository functionality, technologies, and implementation details.
- **Evidence-Based Answers** — Display file references and line ranges to help developers verify responses against source code.
- **Code Execution-Flow Analysis** — Use Abstract Syntax Tree (AST) analysis to investigate function relationships and call chains where supported.
- **Repository Isolation** — Keep repository-specific retrieval data separate to reduce cross-repository information mixing.
- **User Authentication** — Support registration and login with password hashing and JWT-based authentication.
- **Local LLM Integration** — Run question answering through Ollama without requiring a hosted LLM API for the configured local model.

## Technology Stack

### Frontend
- React
- Vite
- Tailwind CSS
- React Router
- Axios
- Lucide React

### Backend
- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- SQLite
- JWT authentication
- bcrypt

### AI and Code Intelligence
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers — `all-MiniLM-L6-v2` embeddings
- ChromaDB vector store
- Ollama
- Qwen2.5-Coder 1.5B
- Python Abstract Syntax Trees (AST)
- Semantic and keyword-based retrieval

### Development Tools
- Git
- GitHub
- Docker
- Docker Compose

> Technology and deployment details should be retained here only if they are present in the current implementation.

## How It Works

1. **Repository input:** The user provides a public GitHub repository.
2. **Repository processing:** RepoLens retrieves and processes supported source files and documentation.
3. **Code parsing:** Files are divided into meaningful chunks, preserving relevant file paths, symbols, and line numbers.
4. **Embedding generation:** Text chunks are converted into vector embeddings.
5. **Indexing:** Embeddings and associated metadata are stored in ChromaDB.
6. **Question submission:** The user asks a question through the chat interface.
7. **Hybrid retrieval:** RepoLens searches for relevant chunks using semantic and keyword-based matching.
8. **Context construction:** Retrieved evidence is added to the prompt sent to the language model.
9. **Answer generation:** Ollama generates an answer based on the available repository context.
10. **Evidence display:** The response includes file paths and line ranges so users can inspect the supporting source.

## RAG Pipeline

```text
Public GitHub Repository
          |
          v
  Fetch & Process Files
          |
          v
  AST-Aware Code Parsing
          |
          v
  Chunking + Metadata
          |
          v
  Generate Embeddings
          |
          v
  ChromaDB Vector Store
          |
          v
     User Question
          |
          v
  Hybrid Evidence Retrieval
          |
          v
  Grounded Prompt Construction
          |
          v
  Ollama + Qwen2.5-Coder 1.5B
          |
          v
  Answer + File/Line References
```

## Getting Started

### Prerequisites

Install the following tools:

- Python 3.11 or a compatible version supported by the dependencies
- Node.js and npm
- Git
- Ollama
- Sufficient memory and disk space for the selected local language model

Docker is optional if you intend to use the Docker Compose setup.

### 1. Clone the Repository

```bash
git clone https://github.com/Prachi-Ukey/repolens-ai.git
cd repolens-ai
```

### 2. Configure Ollama

Make sure Ollama is installed and running. Pull the model used by the current backend configuration:

```bash
ollama pull qwen2.5-coder:1.5b
```

Verify that the model is available:

```bash
ollama list
```

### 3. Set Up the Backend

Open a terminal in the project root:

```bash
cd backend
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the environment variables in `backend/.env` using the project's expected configuration. For example, the current setup uses Ollama as its LLM provider:

```env
LLM_PROVIDER=ollama
```

Do not replace existing required environment settings with this minimal example; preserve the configuration needed by your installation.

Start the backend from the `backend` directory using the development command configured for the application:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend API: http://localhost:8000

API documentation: http://localhost:8000/docs

### 4. Set Up the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. The default development URL is usually:

http://localhost:5173

### 5. Use RepoLens AI

1. Open the frontend in your browser.
2. Register or log in if authentication is enabled.
3. Add or select a supported public GitHub repository.
4. Wait for repository processing and indexing to complete.
5. Ask a question about the codebase.
6. Review the answer and inspect its supporting file and line references.

## Running with Docker Compose

If the current Docker configuration supports the required services, build and start them from the project root:

```bash
docker compose up --build
```

Stop the services with:

```bash
docker compose down
```

Refer to `docker-compose.yml` for service definitions, ports, and environment configuration. Ollama connectivity and model availability must also be configured appropriately for the container environment.

## Project Structure

The following is a representative structure. Adjust it to match the actual repository before publishing.

```text
repo-lens-ai/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   ├── github/
│   │   ├── routes/
│   │   ├── models/
│   │   └── utils/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Security and Privacy

- Never commit `.env` files, passwords, tokens, or other secrets.
- Use a strong, unique secret key for local configuration.
- Treat retrieved repository content as untrusted input.
- Only analyze repositories you are authorized to access.
- Keep authentication secrets and database files out of public commits.

## Current Limitations

- Answer quality depends on the relevance of retrieved evidence and the capabilities of the configured local language model.
- AST-based analysis is limited by the language and code patterns supported by the parser.
- Dynamic behavior, reflection, and external dependencies may not always be resolved accurately.
- Citations should be checked against the referenced source code; an AI-generated answer is not a substitute for verification.
- Local inference speed depends on the model, hardware, and available memory.

## Future Improvements

Potential improvements include broader language support, more precise retrieval evaluation, enhanced citation validation, and additional code-navigation capabilities.


