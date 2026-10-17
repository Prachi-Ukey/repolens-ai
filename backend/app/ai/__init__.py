# Lazy AI exports to avoid loading heavy ML frameworks until requested
def __getattr__(name: str):
    if name == "get_embedding_provider":
        from app.ai.embeddings import get_embedding_provider
        return get_embedding_provider
    elif name == "get_llm_provider":
        from app.ai.llm import get_llm_provider
        return get_llm_provider
    elif name == "get_vector_store":
        from app.ai.vector_store import get_vector_store
        return get_vector_store
    elif name == "RAGEngine":
        from app.ai.rag import RAGEngine
        return RAGEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_embedding_provider",
    "get_llm_provider",
    "get_vector_store",
    "RAGEngine"
]
