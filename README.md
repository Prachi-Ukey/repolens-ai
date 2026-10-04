# RepoLens AI – GitHub Repository Intelligence Platform

RepoLens AI is an AI-powered platform that analyzes public GitHub repositories and allows developers to ask natural-language questions about their codebase. It uses Retrieval-Augmented Generation (RAG), local embeddings, ChromaDB, and an LLM to provide grounded answers with file and line-level code references.

---

## Key Features

- **GitHub Repository Analysis**: Connect a public GitHub repository and analyze its source code automatically.
- **AST-Aware Code Parsing**: Parses source files into meaningful code chunks while preserving file paths, symbols, and line numbers.
- **RAG-Based Code Search**: Retrieves relevant code using semantic and keyword-based search with ChromaDB.
- **AI-Powered Codebase Chat**: Ask natural-language questions about the repository and get answers grounded in the actual source code.
- **Line-Level Code References**: AI responses include file paths and line ranges to help developers locate the relevant implementation.
- **Repository-Isolated Data**: Each analyzed repository has its own vector collection to keep code search isolated.
- **User Authentication**: Supports user registration and login with password hashing and JWT-based authentication.
- **Developer-Friendly Interface**: Provides a workspace for repository analysis, AI chat, and code exploration.

---

## Tech Stack

### Frontend

- **Framework**: React + Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router
- **HTTP Client**: Axios
- **Icons & UI**: Lucide React

### Backend

- **Framework**: Python + FastAPI
- **Server**: Uvicorn
- **Database**: SQLite / SQLAlchemy
- **Authentication**: JWT + bcrypt

### AI & RAG

- **RAG**: Retrieval-Augmented Generation
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB
- **LLM**: Ollama + Qwen2.5-Coder 7B
- **Code Analysis**: AST-aware code parsing

### Development & Deployment

- **Version Control**: Git + GitHub
- **Containerization**: Docker + Docker Compose

---

## How RAG Pipeline Works

```text
User Question
      │
      ▼
Generate Question Embedding
      │
      ▼
Search ChromaDB Vector Store
      │
      ▼
Hybrid Retrieval
(Semantic + Keyword Matching)
      │
      ▼
Select Relevant Code Chunks
(File Path + Symbol + Line Numbers)
      │
      ▼
Build Grounded LLM Prompt
      │
      ▼
Ollama + Qwen2.5-Coder 7B
      │
      ▼
AI Answer with Source References
[file_path:start_line-end_line]
```

The pipeline retrieves relevant code from the analyzed repository and provides it as context to the LLM. The model is instructed to answer only from the retrieved repository evidence and include file and line-level references.

---

## Installation & Local Execution

### Prerequisites

Make sure the following are installed:

- Python 3.11+
- Node.js 18+
- Git
- Ollama
- GitHub account

### 1. Clone the Repository

```bash
git clone https://github.com/Prachi-Ukey/repolens-ai.git
cd repolens-ai
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m app.main
```

Backend API:

`http://localhost:8000`

API documentation:

`http://localhost:8000/docs`

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

`http://localhost:5173`

### 4. Start Ollama

Make sure Ollama is installed and running.

Pull the required model:

```bash
ollama pull qwen2.5-coder:7b
```

RepoLens AI uses Qwen2.5-Coder 7B through Ollama for repository question answering.

---

## Environment Variables

Create or edit `backend/.env` according to your local configuration.

Example:

```env
APP_ENV=development
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./repolens.db

LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=local
CHROMA_PERSIST_DIRECTORY=./chroma_data
```

> **Security:** Never commit API keys, passwords, database credentials, or production secrets to GitHub.

---

## Running with Docker Compose

```bash
docker-compose up --build
```

The application services are configured through `docker-compose.yml`.

---

## Project Structure

```text
repo-lens-ai/
│
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   ├── github/
│   │   ├── routes/
│   │   ├── models/
│   │   └── utils/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
│
├── docs/
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Current AI Workflow

1. User provides a public GitHub repository URL.
2. RepoLens AI validates and downloads the repository.
3. Source files are parsed and divided into meaningful code chunks.
4. Code chunks are converted into vector embeddings.
5. Embeddings are stored in a repository-specific ChromaDB collection.
6. The user asks a natural-language question.
7. Relevant code chunks are retrieved using hybrid search.
8. Retrieved code is provided to the LLM as evidence.
9. The LLM generates a grounded response with file and line-level references.

---

## License

MIT License