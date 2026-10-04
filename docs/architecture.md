# RepoLens AI — Architectural Specification & Design

RepoLens AI is a production-quality full-stack AI-powered codebase analysis tool built with FastAPI, React, ChromaDB, and Retrieval-Augmented Generation (RAG).

```
                      +-----------------------------+
                      |     User Web Browser        |
                      |   (React 18 + Vite + Tailwind)|
                      +--------------+--------------+
                                     |
                                     | REST API / JWT
                                     v
                      +--------------+--------------+
                      |       FastAPI Backend       |
                      |  (Uvicorn / Pydantic v2)    |
                      +-------+--------------+------+
                              |              |
           +------------------+              +------------------+
           |                                                    |
           v                                                    v
+----------+----------+                               +---------+----------+
|  SQLite / Postgres  |                               |   ChromaDB Vector  |
|  (ORM via SQLAlchemy|                               |   Vector Store     |
|   Users, Repos,     |                               |   (Repo Isolated   |
|   Files, Chunks,    |                               |    Collections)    |
|   Conversations)    |                               +---------+----------+
+---------------------+                                         |
                                                                |
                                                                v
                                                      +---------+----------+
                                                      |   LLM & Embeddings |
                                                      |   Provider Layer   |
                                                      | (OpenAI/Gemini/Mock|
                                                      +--------------------+
```

## System Workflow & RAG Ingestion Pipeline

1. **Ingestion & URL Validation**:
   - The user submits a GitHub repository URL (e.g. `https://github.com/username/repo`).
   - Background `AnalysisJob` executes ingestion through distinct stages: `validating`, `downloading`, `detecting_structure`, `parsing_files`, `chunking`, `embedding`, `completed`.

2. **AST & Scope-Aware Code Chunker**:
   - File filters ignore `.git`, `node_modules`, `pycache`, binary files, lockfiles, and `.env` secrets.
   - Code files (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, `.cs`, `.html`, `.css`, `.json`, `.yaml`, `.md`) are split into semantic chunks preserving line ranges, symbol names (`def`, `class`, `function`), file paths, and SHA-256 hashes.

3. **Vector Embeddings & Isolated Storage**:
   - Vector storage uses ChromaDB with isolated collections per repository (`repo_{repo_id}`).
   - Embeddings layer uses provider abstraction (`OpenAI`, `Gemini`, or `MockEmbedding`).

4. **Prompt Injection Defense & Grounded Query Engine**:
   - Question preprocessing retrieves top-k relevant code chunks.
   - User question and code context are wrapped in strict untrusted data tags `<repository_code_context>`.
   - AI system prompt enforces line citations `[file_path:start_line-end_line]` and explicit fallback when context is insufficient.
