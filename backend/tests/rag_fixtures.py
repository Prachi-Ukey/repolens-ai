import atexit
from contextlib import contextmanager
import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import Dict, Any, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database
import app.ai.vector_store
from app.database import Base
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.models.user import User
from app.ai.vector_store import ChromaManager

logger = logging.getLogger("repolens.tests.rag_fixtures")

# Deterministic IDs for RAG fixtures
VEHICLE_REPO_ID = "077bbc54-2495-405c-8037-44bd82550df7"
FASTAPI_REPO_ID = "9602c9a4-5c96-45a6-ad05-ba48684490b0"

# Application modules that directly import SessionLocal or get_vector_store at the module level
PATCHED_MODULE_TARGETS = [
    ("app.services.ingestion_service", "SessionLocal"),
    ("app.services.ingestion_service", "get_vector_store"),
    ("app.routes.chat", "get_vector_store"),
    ("app.routes.demo", "get_vector_store"),
    ("app.routes.repository", "get_vector_store"),
]

_isolated_env: Optional[Dict[str, Any]] = None
_original_state: Dict[str, Any] = {}
_original_module_aliases: Dict[Tuple[str, str], Any] = {}
_original_test_module_aliases: Dict[Tuple[str, str], Any] = {}


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.2) -> None:
    """
    Safely removes a directory tree on Windows, retrying if file locks are briefly held.
    """
    if not path or not os.path.exists(path):
        return
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except Exception:
            time.sleep(delay)
    shutil.rmtree(path, ignore_errors=True)


def _dispose_engine(engine: Any) -> None:
    """Safely disposes an engine to release all connection pool handles."""
    if engine is None:
        return
    try:
        engine.dispose()
    except Exception as e:
        logger.debug("Error disposing engine: %s", e)


def _close_chroma_client(client: Any) -> None:
    """
    Safely closes a Chroma PersistentClient to release SQLite database file locks.
    If client.close() raises an exception, attempts the supported _system.stop() fallback.
    Logs cleanup failures without masking the caller's exception.
    """
    if client is None:
        return

    closed = False
    if hasattr(client, "close") and callable(client.close):
        try:
            client.close()
            closed = True
        except Exception as e:
            logger.warning("client.close() failed during Chroma cleanup: %s", e)

    if not closed and hasattr(client, "_system") and hasattr(client._system, "stop") and callable(client._system.stop):
        try:
            client._system.stop()
        except Exception as e:
            logger.warning("Fallback _system.stop() failed during Chroma cleanup: %s", e)


def _capture_original_state() -> None:
    """
    Captures original global singletons and module-level aliases exactly once,
    preserving true pristine references across multiple activation cycles.
    """
    global _original_state, _original_module_aliases, _original_test_module_aliases

    if "engine" not in _original_state:
        _original_state["engine"] = app.database.engine
        _original_state["SessionLocal"] = app.database.SessionLocal
        _original_state["_vector_store_instance"] = getattr(app.ai.vector_store, "_vector_store_instance", None)
        _original_state["get_vector_store"] = app.ai.vector_store.get_vector_store

    for mod_path, attr_name in PATCHED_MODULE_TARGETS:
        if (mod_path, attr_name) not in _original_module_aliases:
            try:
                mod = sys.modules.get(mod_path)
                if mod is None:
                    mod = importlib.import_module(mod_path)
                if hasattr(mod, attr_name):
                    _original_module_aliases[(mod_path, attr_name)] = getattr(mod, attr_name)
            except Exception as e:
                logger.debug("Could not import %s to capture %s: %s", mod_path, attr_name, e)



def get_isolated_rag_env() -> Dict[str, Any]:
    """
    Initializes and returns a dedicated, isolated test database and ChromaDB storage
    populated with deterministic seed data for vehicle-damage-detection and fastapi.
    Guarantees no access to normal development database or normal Chroma storage.
    Cleans up all temporary resources if setup fails midway.
    """
    global _isolated_env

    if _isolated_env is not None:
        _rebind_dependencies(_isolated_env)
        return _isolated_env

    # 1. Capture pristine original references once
    _capture_original_state()

    # 2. Create isolated temp directory for test DB and test vector store
    temp_dir = tempfile.mkdtemp(prefix="repolens_rag_isolated_")
    chroma_dir = os.path.join(temp_dir, "chroma")
    db_path = os.path.join(temp_dir, "isolated_rag.db")
    os.makedirs(chroma_dir, exist_ok=True)

    test_engine = None
    test_vector_store = None
    db = None

    try:
        # 3. Create dedicated test engine & sessionmaker (isolated from normal DATABASE_URL)
        test_engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        Base.metadata.create_all(bind=test_engine)

        # 4. Create dedicated isolated ChromaManager (isolated from normal CHROMA_PERSIST_DIRECTORY)
        test_vector_store = ChromaManager(persist_dir=chroma_dir)

        # 5. Populate isolated test database and test ChromaDB from deterministic seed data
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "rag_seed_data.json")
        if not os.path.exists(fixture_path):
            raise FileNotFoundError(f"RAG seed fixture file not found at: {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as fp:
            seed_data = json.load(fp)

        db = TestSessionLocal()

        # Test User
        test_user = User(
            id=str(uuid.uuid4()),
            username="rag_test_fixture_user",
            email="rag_fixture@repolens.ai",
            hashed_password="hashed_fixture_password",
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # Repositories
        for r_data in seed_data.get("repositories", []):
            r = Repository(
                id=r_data["id"],
                user_id=test_user.id,
                url=r_data["url"],
                owner=r_data["owner"],
                name=r_data["name"],
                default_branch=r_data.get("default_branch", "main"),
                primary_language=r_data.get("primary_language", "Python"),
                file_count=r_data.get("file_count", 0),
                chunk_count=r_data.get("chunk_count", 0),
                status=r_data.get("status", "completed"),
                embedding_model=test_vector_store.embedding_model_name,
                embedding_dimension=test_vector_store.embedding_dimension,
                summary_json=r_data.get("summary_json", "{}"),
            )
            db.add(r)
        db.commit()

        # Files
        file_id_map = {}
        for f_data in seed_data.get("files", []):
            f_id = f_data.get("id") or str(uuid.uuid4())
            f = RepositoryFile(
                id=f_id,
                repository_id=f_data["repository_id"],
                file_path=f_data["file_path"],
                file_name=f_data.get("file_name") or os.path.basename(f_data["file_path"]),
                language=f_data.get("language", "text"),
                size_bytes=f_data.get("size_bytes", len(f_data.get("content", "").encode("utf-8"))),
                line_count=f_data.get("line_count", len(f_data.get("content", "").splitlines())),
                sha_hash=f_data.get("sha_hash", "sha_fixture"),
                content=f_data.get("content", ""),
            )
            db.add(f)
            file_id_map[(f.repository_id, f.file_path)] = f_id
        db.commit()

        # Chunks
        chunks_by_repo: Dict[str, list] = {}
        for c_data in seed_data.get("chunks", []):
            repo_id = c_data["repository_id"]
            file_id = file_id_map.get((repo_id, c_data["file_path"]))
            if not file_id:
                file_id = str(uuid.uuid4())
                fallback_f = RepositoryFile(
                    id=file_id,
                    repository_id=repo_id,
                    file_path=c_data["file_path"],
                    file_name=os.path.basename(c_data["file_path"]),
                    language=c_data.get("language", "python"),
                    size_bytes=len(c_data.get("content", "").encode("utf-8")),
                    line_count=c_data.get("end_line", 1) - c_data.get("start_line", 1) + 1,
                    sha_hash="sha_fallback",
                    content=c_data.get("content", ""),
                )
                db.add(fallback_f)
                db.commit()
                file_id_map[(repo_id, c_data["file_path"])] = file_id

            c = CodeChunk(
                id=c_data.get("id") or str(uuid.uuid4()),
                chunk_id=c_data["chunk_id"],
                repository_id=repo_id,
                file_id=file_id,
                file_path=c_data["file_path"],
                language=c_data.get("language", "python"),
                start_line=c_data.get("start_line", 1),
                end_line=c_data.get("end_line", 1),
                symbol_name=c_data.get("symbol_name") or "",
                content=c_data["content"],
                sha_hash=c_data.get("sha_hash", "sha_chunk"),
            )
            db.add(c)
            chunks_by_repo.setdefault(repo_id, []).append({
                "chunk_id": c_data["chunk_id"],
                "content": c_data["content"],
                "file_path": c_data["file_path"],
                "language": c_data.get("language", "python"),
                "start_line": c_data.get("start_line", 1),
                "end_line": c_data.get("end_line", 1),
                "symbol_name": c_data.get("symbol_name") or "",
            })
        db.commit()
        db.close()
        db = None

        # Seed isolated ChromaDB collections
        for repo_id, c_list in chunks_by_repo.items():
            test_vector_store.safe_replace_collection(repo_id, c_list)

        _isolated_env = {
            "temp_dir": temp_dir,
            "chroma_dir": chroma_dir,
            "db_path": db_path,
            "engine": test_engine,
            "SessionLocal": TestSessionLocal,
            "vector_store": test_vector_store,
            "vehicle_repo_id": VEHICLE_REPO_ID,
            "fastapi_repo_id": FASTAPI_REPO_ID,
        }

        _rebind_dependencies(_isolated_env)
        return _isolated_env

    except Exception:
        # Partial setup failure cleanup: do not leave orphaned files, engines, or clients
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
        if test_engine is not None:
            _dispose_engine(test_engine)
        if test_vector_store is not None and hasattr(test_vector_store, "client"):
            _close_chroma_client(test_vector_store.client)
        _safe_rmtree(temp_dir)
        _isolated_env = None
        raise


def _rebind_test_module(mod_name: str, mod: Any, env: Dict[str, Any]) -> None:
    """
    Patches SessionLocal and get_vector_store on an affected test module.
    Captures original values before patching, handling modules imported after
    isolation started without overwriting previously captured originals.
    """
    global _original_test_module_aliases
    if not mod:
        return

    test_session_local = env["SessionLocal"]
    test_vector_store = env["vector_store"]

    if hasattr(mod, "SessionLocal"):
        if (mod_name, "SessionLocal") not in _original_test_module_aliases:
            curr = getattr(mod, "SessionLocal")
            if curr is test_session_local:
                # Module was imported after isolation started; its fallback original is core SessionLocal
                _original_test_module_aliases[(mod_name, "SessionLocal")] = _original_state.get(
                    "SessionLocal", app.database.SessionLocal
                )
            else:
                _original_test_module_aliases[(mod_name, "SessionLocal")] = curr
        setattr(mod, "SessionLocal", test_session_local)

    if hasattr(mod, "get_vector_store"):
        if (mod_name, "get_vector_store") not in _original_test_module_aliases:
            curr = getattr(mod, "get_vector_store")
            is_isolated_fn = False
            try:
                if callable(curr) and curr() is test_vector_store:
                    is_isolated_fn = True
            except Exception:
                pass

            if is_isolated_fn:
                # Module was imported after isolation started; its fallback original is core get_vector_store
                _original_test_module_aliases[(mod_name, "get_vector_store")] = _original_state.get(
                    "get_vector_store", app.ai.vector_store.get_vector_store
                )
            else:
                _original_test_module_aliases[(mod_name, "get_vector_store")] = curr
        setattr(mod, "get_vector_store", lambda: test_vector_store)


def _rebind_dependencies(env: Dict[str, Any]) -> None:
    """
    Rebinds and patches app.database, app.ai.vector_store, and direct module-level
    imports in application routes and services to use the isolated test instances.
    """
    _capture_original_state()

    test_engine = env["engine"]
    TestSessionLocal = env["SessionLocal"]
    test_vector_store = env["vector_store"]

    # 1. Rebind core singletons
    app.database.engine = test_engine
    app.database.SessionLocal = TestSessionLocal

    app.ai.vector_store._vector_store_instance = test_vector_store
    app.ai.vector_store.get_vector_store = lambda: test_vector_store

    # 2. Patch direct application module aliases
    for mod_path, attr_name in PATCHED_MODULE_TARGETS:
        mod = sys.modules.get(mod_path)
        if mod is None:
            try:
                mod = importlib.import_module(mod_path)
            except Exception:
                mod = None
        if mod is not None:
            if attr_name == "SessionLocal":
                setattr(mod, attr_name, TestSessionLocal)
            elif attr_name == "get_vector_store":
                setattr(mod, attr_name, lambda: test_vector_store)

    # 3. Rebind in any loaded RAG test modules in sys.modules
    for name, mod in list(sys.modules.items()):
        if mod and (name.startswith("tests.test_rag") or name == "tests.test_rag_call_chain_edges"):
            _rebind_test_module(name, mod, env)


def activate_rag_isolation() -> Dict[str, Any]:
    """
    Activates isolated test DB and vector store, rebinding dependencies.
    """
    env = get_isolated_rag_env()
    _rebind_dependencies(env)
    return env


def deactivate_rag_isolation() -> None:
    """
    Restores original production/development DB and vector store bindings,
    as well as all patched application and test module references to their exact originals.
    """
    global _original_state, _original_module_aliases, _original_test_module_aliases

    # 1. Restore core singletons
    if "engine" in _original_state:
        app.database.engine = _original_state["engine"]
        app.database.SessionLocal = _original_state["SessionLocal"]
        app.ai.vector_store._vector_store_instance = _original_state["_vector_store_instance"]
        app.ai.vector_store.get_vector_store = _original_state["get_vector_store"]

    # 2. Restore patched application modules
    for (mod_path, attr_name), orig_val in list(_original_module_aliases.items()):
        mod = sys.modules.get(mod_path)
        if mod is not None:
            setattr(mod, attr_name, orig_val)

    # 3. Restore test modules to their exact captured originals
    for (mod_name, attr), orig_val in list(_original_test_module_aliases.items()):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            setattr(mod, attr, orig_val)


@contextmanager
def isolated_rag_context():
    """
    Context manager that activates RAG isolation for the duration of a test block
    and guarantees exact restoration of original references on exit, even on error.
    """
    env = activate_rag_isolation()
    try:
        yield env
    finally:
        deactivate_rag_isolation()


def ensure_rag_fixtures() -> Dict[str, str]:
    """
    Public helper called by RAG test suites to guarantee isolated test DB and ChromaDB
    with deterministic repository IDs.
    """
    env = activate_rag_isolation()
    return {
        "vehicle_repo_id": env["vehicle_repo_id"],
        "fastapi_repo_id": env["fastapi_repo_id"],
    }


def get_rag_session():
    """Returns a new DB session from the isolated test database."""
    env = get_isolated_rag_env()
    return env["SessionLocal"]()


def get_rag_vector_store():
    """Returns the isolated test vector store instance."""
    env = get_isolated_rag_env()
    return env["vector_store"]


def teardown_isolated_rag_env() -> None:
    """
    Restores original database engine, SessionLocal, and vector store singletons,
    restores patched module aliases, disposes of test engine and Chroma client,
    and cleans up temporary isolated directory.
    """
    global _isolated_env

    deactivate_rag_isolation()

    if _isolated_env is not None:
        engine = _isolated_env.get("engine")
        if engine is not None:
            _dispose_engine(engine)

        vstore = _isolated_env.get("vector_store")
        if vstore is not None and hasattr(vstore, "client"):
            _close_chroma_client(vstore.client)

        temp_dir = _isolated_env.get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            _safe_rmtree(temp_dir)
        _isolated_env = None


atexit.register(teardown_isolated_rag_env)
