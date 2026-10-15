# RepoLens AI — Step-by-Step Deployment Checklist (Windows & Cloud)

Follow this checklist to test locally on Windows and deploy to **Vercel** (Frontend) and **Render** (Backend).

---

## 1. Local Pre-Deployment Verification (Windows PowerShell)

Run these checks in PowerShell from the project root `c:\Users\Prachi\.gemini\antigravity\scratch\repo-lens-ai`:

### A. Run Backend Test Suite
```powershell
cd backend
.\venv\Scripts\python -m pytest tests/
```
- **Expected result**: All tests pass (103 tests passed).

### B. Verify Frontend Production Build
```powershell
cd ..\frontend
npm run build
```
- **Expected result**: `dist/` is generated with `✓ built in X.XXs` with 0 errors.

### C. Verify Vector Dimension Compatibility
```powershell
cd ..\backend
.\venv\Scripts\python reindex_repo.py --verify-dimensions
```
- **Expected result**: All indexed repository collections display `COMPATIBLE` (384 dimensions).

---

## 2. Local Docker Compose with Windows Ollama

If running locally with Docker Compose while Ollama runs on your Windows desktop:

1. **Ensure Ollama is running and model is pulled**:
   ```powershell
   ollama list
   # If qwen2.5-coder:1.5b is not listed:
   ollama pull qwen2.5-coder:1.5b
   ```

2. **Verify `docker-compose.yml` environment**:
   ```yaml
   environment:
     - APP_ENV=production
     - DATABASE_URL=sqlite:///./repolens.db
     - CHROMA_PERSIST_DIRECTORY=./chroma_data
     - LLM_PROVIDER=ollama
     - OLLAMA_BASE_URL=http://host.docker.internal:11434
     - OLLAMA_MODEL=qwen2.5-coder:1.5b
     - EMBEDDING_PROVIDER=mock
     - EMBEDDING_DIMENSION=384
   ```

3. **Start Docker Compose**:
   ```powershell
   cd ..
   docker compose up -d --build
   ```

4. **Verify container logs and health**:
   ```powershell
   docker compose logs -f backend
   ```
   Open in your browser:
   - Backend Health: `http://localhost:8000/health`
   - Frontend UI: `http://localhost:3000`

---

## 3. Cloud Backend Deployment (Render)

Render supports FastAPI with persistent disks to preserve your SQLite database and ChromaDB vector store across deployments and restarts.

> **Note on Cloud LLM**: Local Ollama runs on your desktop, so a cloud backend cannot reach `localhost:11434` or `host.docker.internal`. In cloud production, configure a hosted LLM (OpenAI or Gemini), or use `LLM_PROVIDER=mock` for zero-cost portfolio demonstrations.

### Step 1: Push Code to GitHub
```powershell
git add .
git commit -m "feat: finalize core features, real Ollama integration, and deployment config"
git push origin main
```

### Step 2: Create a Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and click **New +** → **Web Service**.
2. Connect your GitHub repository: `Prachi-Ukey/repolens-ai`.
3. Configure service settings:
   - **Name**: `repolens-ai-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Region**: Closest to your users (e.g., Oregon, Frankfurt, Singapore)
   - **Build Command**: `pip install --no-cache-dir -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Starter` (Persistent Disk requires a paid Starter plan; for free tier, use ephemeral storage `./repolens.db` and `./chroma_data`)

### Step 3: Attach Persistent Disk (Optional for data persistence across restarts)
1. In the service settings, go to **Disks** → **Add Disk**.
2. **Name**: `repolens-data`
3. **Mount Path**: `/var/data`
4. **Size**: `10 GB`

### Step 4: Configure Environment Variables on Render
Under **Environment Variables**, add the following (or use `render.yaml` Blueprint which configures them automatically):

| Key | Recommended Value | Security & Production Notes |
|---|---|---|
| `APP_ENV` | `production` | **Required**: Enforces strict fail-fast production security validations. |
| `SECRET_KEY` | *(Render auto-generates or 32+ char hex)* | **Critical Security Requirement**: Must be a cryptographically secure random string (at least 16 characters). The application **fails fast on startup** if `SECRET_KEY` is missing, short, or matches placeholders (e.g. `CHANGE_ME_...` or old defaults). If using `render.yaml`, Render auto-generates this with `generateValue: true`. On Windows, you can generate one with PowerShell: `[Convert]::ToHexString((1..32 \| ForEach-Object { [byte](Get-Random -Max 256) })).ToLower()` |
| `CORS_ORIGINS` | `https://<your-vercel-app>.vercel.app` | **Critical Security Requirement**: Must be the exact deployed frontend URL (no trailing slash). **Do NOT use `*` in production**; browsers block authenticated requests/credentials when wildcard CORS is used. |
| `LLM_PROVIDER` | `gemini` (or `openai`) | **Required for Production**: Cloud servers cannot access local desktop Ollama. Do **not** deploy with `mock` in production. |
| `GEMINI_API_KEY` | *(your secret Gemini API key)* | Enter securely in the Render Dashboard. In `render.yaml`, this is marked `sync: false` to guarantee secrets are never committed to Git. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Fast, cost-effective model for production RAG inference. |
| `OPENAI_API_KEY` | *(your secret OpenAI API key)* | Only needed if `LLM_PROVIDER=openai`. Marked `sync: false` in `render.yaml`. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cost-effective OpenAI alternative. |
| `DATABASE_URL` | `sqlite:////var/data/repolens.db` | Points to persistent disk mount path. |
| `CHROMA_PERSIST_DIRECTORY` | `/var/data/chroma_data` | Points to persistent disk mount path. |
| `EMBEDDING_PROVIDER` | `gemini` | In `render.yaml`, configured as `gemini` using `gemini-embedding-001` (or `mock` for demo). |
| `EMBEDDING_DIMENSION` | `768` | 768-dimensional embeddings for `gemini-embedding-001`. |

> **Security Rule**: Never commit API keys, tokens, or production JWT secrets to Git. Always enter them through the hosted platform's secure dashboard or environment variable manager.

### Step 5: Verify Backend Deployment
Once deployed, open:
`https://<your-render-backend-url>/health`
Expected response:
```json
{
  "status": "ok",
  "app": "RepoLens AI",
  "env": "production",
  "llm_provider": "gemini",
  "embedding_provider": "gemini",
  "model": "gemini-2.5-flash"
}
```

---

## 4. Cloud Frontend Deployment (Vercel)

Vercel provides edge hosting, automatic SSL, and continuous deployment for the Vite React frontend.

### Step 1: Import Project on Vercel
1. Go to [vercel.com](https://vercel.com) and click **Add New...** → **Project**.
2. Select your repository: `Prachi-Ukey/repolens-ai`.

### Step 2: Configure Project Settings
- **Framework Preset**: `Vite`
- **Root Directory**: Click **Edit** and select `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### Step 3: Add Environment Variables on Vercel
Under **Environment Variables**, add:

| Key | Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `https://<your-render-backend-url>` | URL of your deployed Render backend (no trailing slash) |

### Step 4: Deploy and Test
1. Click **Deploy**.
2. When the build finishes, open the live Vercel URL: `https://<your-project>.vercel.app`.
3. Test the flow:
   - Register a new account / Log in.
   - Navigate to the repository workspace.
   - Ask a question (e.g. *"What is this repository and what is its main purpose?"*).
   - Verify that citations, source links, and answers load smoothly.

---

## 5. Troubleshooting & FAQ

- **Q: Why does local chat return "Connection to local AI failed"?**
  - **A**: Ensure Ollama is running (`ollama serve`) and `qwen2.5-coder:1.5b` is downloaded (`ollama pull qwen2.5-coder:1.5b`). If using Docker, check that `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

- **Q: Why are citations rejected as ungrounded?**
  - **A**: RepoLens validates citations against actual code lines in the repository. If a citation refers to nonexistent or drifted lines, it is rejected to prevent hallucinations.

- **Q: Can I use Ollama in the cloud on Render/Vercel?**
  - **A**: Standard serverless and container hosts don't have local GPUs or Ollama daemons. For cloud deployments, use `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` with an API key, or `LLM_PROVIDER=mock` for testing.
