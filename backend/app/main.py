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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
