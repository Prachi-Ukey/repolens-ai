# RepoLens AI — REST API Reference

All REST API endpoints are served under `/api` (or direct backend origin `http://localhost:8000`).

## Authentication Endpoints

### POST `/auth/register`
Creates a new developer user account.
- **Request**: `{ "email": "user@example.com", "username": "devuser", "password": "secretpassword" }`
- **Response 201**: User object `UserResponse`.

### POST `/auth/login`
Authenticates a user and issues a Bearer JWT token.
- **Request**: `{ "username_or_email": "devuser", "password": "secretpassword" }`
- **Response 200**: `{ "access_token": "<jwt>", "token_type": "bearer" }`

### GET `/auth/me`
Returns current authenticated user details.
- **Headers**: `Authorization: Bearer <token>`
- **Response 200**: `UserResponse`.

---

## Repository Management

### POST `/repositories`
Submits a public GitHub URL for analysis.
- **Headers**: `Authorization: Bearer <token>`
- **Request**: `{ "url": "https://github.com/expressjs/express" }`
- **Response 201**: `{ "repository": RepositoryResponse, "job_id": "<uuid>", "message": "..." }`

### GET `/repositories`
Lists all repositories analyzed by the current user.

### GET `/repositories/{id}/analysis-status`
Returns real-time background analysis progress, stage, and percentage.

### GET `/repositories/{id}/files`
Returns flat file list and nested file tree structure for repository file navigation.

### GET `/repositories/{id}/files/{file_id}`
Returns file details and source code text.

### DELETE `/repositories/{id}`
Deletes repository metadata and cleans up Chroma vector collections.

---

## AI Codebase Chat & Search

### POST `/repositories/{id}/chat`
Queries the RAG engine for grounded codebase answers.
- **Request**: `{ "question": "Where is authentication handled?", "top_k": 5 }`
- **Response 200**:
```json
{
  "conversation_id": "conv-123",
  "question": "Where is authentication handled?",
  "answer": "Authentication is handled in server/middleware/auth.js...",
  "sources": [
    {
      "file_path": "server/middleware/auth.js",
      "start_line": 12,
      "end_line": 38,
      "symbol_name": "verifyToken",
      "snippet": "..."
    }
  ],
  "grounded": true
}
```

### GET `/repositories/{id}/conversations`
Retrieves past chat history for a repository.

---

## Code Quality & Insights

### GET `/repositories/{id}/insights`
Runs static analysis rule engine and returns list of severity-badged potential security/complexity items.

### GET `/repositories/{id}/dependencies`
Parses package manifests (`package.json`, `requirements.txt`) and lists ecosystems & versions.

### POST `/repositories/{id}/generate-documentation`
Generates README, API reference, Architecture overview, or Onboarding guide.

---

## Demo Mode

### POST `/demo/load-sample`
Instantly loads pre-indexed sample repository into user account without requiring GitHub credentials.
