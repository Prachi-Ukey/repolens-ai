from app.routes.auth import router as auth_router
from app.routes.repository import router as repo_router
from app.routes.chat import router as chat_router
from app.routes.insights import router as insights_router
from app.routes.docgen import router as docgen_router
from app.routes.demo import router as demo_router

__all__ = [
    "auth_router",
    "repo_router",
    "chat_router",
    "insights_router",
    "docgen_router",
    "demo_router"
]
