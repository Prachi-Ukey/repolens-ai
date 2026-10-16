from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.routes.auth import router as auth_router
from app.routes.repository import router as repo_router
from app.routes.chat import router as chat_router
from app.routes.insights import router as insights_router
from app.routes.docgen import router as docgen_router
from app.routes.demo import router as demo_router
from app.utils.logger import logger

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-quality full-stack AI repository analysis and codebase assistant.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
cors_origins = (
    [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    if settings.CORS_ORIGINS and settings.CORS_ORIGINS != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
    if str(settings.APP_ENV).strip().lower() == "production":
        if not settings.CORS_ORIGINS or settings.CORS_ORIGINS.strip() == "*":
            logger.warning(
                "Security Warning: CORS_ORIGINS is set to '*' in production while allow_credentials=True. "
                "Browsers will reject credentialed cross-origin requests. "
                "Specify your exact deployed frontend origin (e.g. CORS_ORIGINS=https://repolens-ai.vercel.app)."
            )
    init_db()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."}
    )

# Register API Routers
app.include_router(auth_router)
app.include_router(repo_router)
app.include_router(chat_router)
app.include_router(insights_router)
app.include_router(docgen_router)
app.include_router(demo_router)

@app.get("/", tags=["Status"])
def root():
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "env": settings.APP_ENV,
        "docs": "/docs"
    }

@app.get("/health", tags=["Status"])
def health():
    active_model = settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else getattr(settings, f"{settings.LLM_PROVIDER.upper()}_MODEL", "default")
    from app.ai.embeddings import get_active_embedding_model_name
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_dimension": getattr(settings, "EMBEDDING_DIMENSION", 384),
        "embedding_model": get_active_embedding_model_name(),
        "model": active_model
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
