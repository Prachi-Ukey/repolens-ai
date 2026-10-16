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
- **Expected result**: All tests pass (150 tests passed).

### B. Verify Frontend Production Build
```powershell
cd ..\frontend
npm run build
```
- **Expected result**: `dist/` is generated with `✓ built in X.XXs` with 0 errors.

### C. Verify Vector Dimension Compatibility & Reindexing
```powershell
cd ..\backend
.\venv\Scripts\python reindex_repo.py --verify-dimensions
```
- **Expected result**: All indexed repository collections display `COMPATIBLE` (384 dimensions).
- If any legacy 768-dimension collections exist from previous Gemini experiments, migrate them safely:
  ```powershell
  .\venv\Scripts\python reindex_repo.py --rebuild-incompatible
  ```

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
     - EMBEDDING_PROVIDER=local
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

Render supports FastAPI web services for both completely free ephemeral hosting and paid persistent hosting.

> **Crucial Free-Tier Hosting Facts**:
> 1. **Ephemeral Filesystem (Free Tier)**: Render's free tier spins down after 15 minutes of inactivity. When it restarts, ephemeral storage is recreated fresh, meaning SQLite (`./repolens.db`) and ChromaDB (`./chroma_data`) reset. To retain persistent vector data across restarts without paying, run the backend locally or use an external managed database.
> 2. **Memory Limit (512MB RAM)**: Render's free plan provides 512MB RAM. Sentence-Transformers (`all-MiniLM-L6-v2`) uses ~90MB weights and ~250MB PyTorch runtime, nearing the 512MB cap. RepoLens AI uses lazy singleton loading and bounded batch sizes (32) to fit. For pure UI/portfolio demonstrations with zero memory pressure, `EMBEDDING_PROVIDER=mock` can also be used.
> 3. **Free Gemini Chat Generation (Optional)**: While Gemini Embedding free tier easily exhausts its 30,000 TPM limit on repository ingestion, Gemini Flash chat generation (`gemini-2.5-flash`) has a 15 RPM / 1M TPM free tier limit, which comfortably handles interactive chat questions.

### Step 1: Push Code to GitHub
```powershell
git add .
git commit -m "feat: complete free-tier fix, local embeddings, and safe reindexing"
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
   - **Plan**: `Free` (or `Starter` if you want persistent disk `/var/data`)

### Step 3: Attach Persistent Disk (Optional - Paid Plan Only)
> On Render's Free tier, persistent disks are not available. If on a paid Starter plan:
1. In the service settings, go to **Disks** → **Add Disk**.
2. **Name**: `repolens-data`
3. **Mount Path**: `/var/data`
4. **Size**: `10 GB`

### Step 4: Configure Environment Variables on Render
Under **Environment Variables**, add the following (or use `render.yaml` Blueprint which sets them automatically):

| Key | Recommended Value | Security & Production Notes |
|---|---|---|
| `APP_ENV` | `production` | **Required**: Enforces strict fail-fast production security validations. |
| `SECRET_KEY` | *(Render auto-generates or 32+ char hex)* | **Critical Security Requirement**: Must be a cryptographically secure random string (at least 16 characters). The application **fails fast on startup** if `SECRET_KEY` is missing, short, or matches placeholders. |
| `CORS_ORIGINS` | `https://<your-vercel-app>.vercel.app` | **Critical Security Requirement**: Must be the exact deployed frontend URL (no trailing slash). **Do NOT use `*` in production**. |
| `LLM_PROVIDER` | `gemini` (or `mock` for zero-API demo) | Cloud servers cannot access desktop Ollama. `gemini` uses free Gemini Flash chat. |
| `GEMINI_API_KEY` | *(your secret Gemini API key)* | Enter securely in Render Dashboard. Passed via `x-goog-api-key` header with exponential backoff & jitter. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Fast, cost-effective model for production RAG inference. |
| `DATABASE_URL` | `sqlite:///./repolens.db` (Free) or `sqlite:////var/data/repolens.db` (Paid Disk) | Database path. |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` (Free) or `/var/data/chroma_data` (Paid Disk) | Vector store directory. |
| `EMBEDDING_PROVIDER` | `local` (or `mock`) | **Free Local Embedding**: Uses `all-MiniLM-L6-v2` (zero API quota/cost). |
| `EMBEDDING_DIMENSION` | `384` | Standardized 384 dimensions matching `all-MiniLM-L6-v2`. |

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
  "embedding_provider": "local",
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

- **Q: Why was I getting HTTP 429 RESOURCE_EXHAUSTED during repository ingestion?**
  - **A**: Google AI Studio free tier for Gemini Embeddings enforces a strict 30,000 TPM limit. When ingesting a multi-file repository, hundreds of chunks are embedded, immediately exceeding this quota. RepoLens AI now defaults to `EMBEDDING_PROVIDER=local` using `all-MiniLM-L6-v2`, which runs completely locally on CPU with zero rate limits and zero token charges.

- **Q: How does Render's 512MB memory limit affect local embeddings?**
  - **A**: PyTorch and Sentence-Transformers use ~340MB RAM combined with the web server. To prevent out-of-memory crashes on Render's free tier, models are lazily loaded on demand as singletons, and embedding batches are capped at 32 chunks. For ultra-low RAM usage in demo environments, configure `EMBEDDING_PROVIDER=mock`.

- **Q: What happens to my database when Render's free container sleeps?**
  - **A**: Render free web services spin down after 15 minutes without requests and their filesystem is ephemeral. Re-opening the app will start a fresh container with an empty database. For persistent data, host the backend locally on Windows or connect an external cloud database (such as Supabase PostgreSQL).

- **Q: How do I migrate existing 768-d Gemini vectors to 384-d local embeddings?**
  - **A**: Run `python reindex_repo.py --rebuild-incompatible`. The script safely detects dimension mismatches, re-indexes chunks using the active model into a temporary Chroma collection, updates the database metadata, and performs an atomic swap without data loss.

- **Q: Why are citations rejected as ungrounded?**
  - **A**: RepoLens validates citations against actual code lines in the repository. If a citation refers to nonexistent or drifted lines, it is rejected to prevent hallucinations.

- **Q: Can I use Ollama in the cloud on Render/Vercel?**
  - **A**: Standard serverless and container hosts don't have local GPUs or Ollama daemons. For cloud deployments, use `LLM_PROVIDER=gemini` with a free Gemini Flash API key, or `LLM_PROVIDER=mock` for testing.
