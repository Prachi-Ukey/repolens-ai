# Lazy router exports to avoid importing heavy AI/Chroma modules at package load time
def __getattr__(name: str):
    if name == "auth_router":
        from app.routes.auth import router
        return router
    elif name == "repo_router":
        from app.routes.repository import router
        return router
    elif name == "chat_router":
        from app.routes.chat import router
        return router
    elif name == "insights_router":
        from app.routes.insights import router
        return router
    elif name == "docgen_router":
        from app.routes.docgen import router
        return router
    elif name == "demo_router":
        from app.routes.demo import router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "auth_router",
    "repo_router",
    "chat_router",
    "insights_router",
    "docgen_router",
    "demo_router"
]
