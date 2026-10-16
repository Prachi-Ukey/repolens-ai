# RepoLens AI — Local Setup Guide

Follow this step-by-step guide to run RepoLens AI locally on Windows, Linux, or macOS.

## Prerequisites

- Python 3.11+
- Node.js 18+ & npm 9+
- Git

---

## 1. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration uses SQLite database and local sentence-transformers (`all-MiniLM-L6-v2`, 384 dimensions) for completely free, quota-free vector search:
```env
EMBEDDING_PROVIDER=local
EMBEDDING_DIMENSION=384
```
To use cloud LLMs (e.g. Gemini Flash) for chat responses, set:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

5. Start backend development server:
```bash
uvicorn app.main:app --reload --port 8000
```
Open API Docs at: `http://localhost:8000/docs`

---

## 2. Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start Vite dev server:
```bash
npm run dev
```
Open web UI at: `http://localhost:5173`

---

## 3. Running via Docker Compose

To start the full application using Docker:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
