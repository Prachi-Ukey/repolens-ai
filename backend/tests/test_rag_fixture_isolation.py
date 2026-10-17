import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from app.config import settings
import app.database
import app.ai.vector_store
from app.ai.vector_store import get_vector_store
from app.models.repository import Repository
from app.models.file import RepositoryFile
from app.models.chunk import CodeChunk
from app.models.job import AnalysisJob
from app.models.user import User

import app.services.ingestion_service as ingestion_service
import app.routes.chat as chat_route
import app.routes.demo as demo_route
import app.routes.repository as repo_route

from tests.rag_fixtures import (
    ensure_rag_fixtures,
    get_isolated_rag_env,
    get_rag_session,
    get_rag_vector_store,
    activate_rag_isolation,
    deactivate_rag_isolation,
    teardown_isolated_rag_env,
    isolated_rag_context,
    _original_state,
    _original_module_aliases,
)


class TestRAGFixtureIsolation(unittest.TestCase):
    """
    Regression tests verifying that RAG fixtures use dedicated temporary test database
    and ChromaDB storage, with strict isolation from the development database and vector store,
    reliable cleanup on partial failure, module alias patching, and safe deactivation.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixtures = ensure_rag_fixtures()
        cls.env = get_isolated_rag_env()

    @classmethod
    def tearDownClass(cls):
        deactivate_rag_isolation()

    def test_database_is_isolated_from_normal_settings_url(self):
        """Verify the database engine and SessionLocal do not connect to settings.DATABASE_URL."""
        # 1. Engine URL must not be the production/development database URL
        active_db_url = str(app.database.engine.url)
        self.assertNotEqual(
            active_db_url,
            settings.DATABASE_URL,
            f"Active DB URL '{active_db_url}' unexpectedly matches settings.DATABASE_URL",
        )

        # 2. Must not contain 'repolens.db' in the path
        self.assertNotIn("repolens.db", active_db_url.lower())

        # 3. Active database file must reside in a dedicated temporary directory
        self.assertIn("isolated_rag.db", active_db_url)
        self.assertTrue(os.path.exists(self.env["db_path"]))

        # 4. SessionLocal produces sessions bound to the isolated engine
        session = app.database.SessionLocal()
        try:
            self.assertEqual(str(session.bind.url), str(app.database.engine.url))
        finally:
            session.close()

    def test_vector_store_is_isolated_from_normal_chroma_directory(self):
        """Verify ChromaManager does not point to settings.CHROMA_PERSIST_DIRECTORY."""
        vs = get_vector_store()
        active_persist_dir = os.path.abspath(vs.persist_dir)
        normal_persist_dir = os.path.abspath(settings.CHROMA_PERSIST_DIRECTORY)

        # 1. Active persist dir must not be the production/development directory
        self.assertNotEqual(
            active_persist_dir,
            normal_persist_dir,
            f"Active Chroma persist dir '{active_persist_dir}' unexpectedly matches normal '{normal_persist_dir}'",
        )

        # 2. Must reside in dedicated temporary directory
        self.assertTrue(os.path.exists(active_persist_dir))
        self.assertEqual(os.path.abspath(self.env["chroma_dir"]), active_persist_dir)

        # 3. Collections must exist in isolated store
        v_col_name = vs._get_collection_name(self.fixtures["vehicle_repo_id"])
        f_col_name = vs._get_collection_name(self.fixtures["fastapi_repo_id"])

        self.assertGreater(vs.client.get_collection(v_col_name).count(), 0)
        self.assertGreater(vs.client.get_collection(f_col_name).count(), 0)

    def test_deterministic_repository_records_in_isolated_database(self):
        """Verify consistent repository IDs, files, and chunks are present in isolated storage."""
        db = get_rag_session()
        try:
            v_repo = db.query(Repository).filter(Repository.id == self.fixtures["vehicle_repo_id"]).first()
            f_repo = db.query(Repository).filter(Repository.id == self.fixtures["fastapi_repo_id"]).first()

            self.assertIsNotNone(v_repo, "Vehicle damage detection repo missing in isolated DB")
            self.assertIsNotNone(f_repo, "FastAPI repo missing in isolated DB")

            self.assertEqual(v_repo.name, "vehicle-damage-detection")
            self.assertEqual(f_repo.name, "fastapi")

            # Verify files exist in isolated database
            v_files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == v_repo.id).all()
            f_files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == f_repo.id).all()

            self.assertGreaterEqual(len(v_files), 4)
            self.assertGreaterEqual(len(f_files), 3)

            # Verify chunks exist in isolated database
            v_chunks = db.query(CodeChunk).filter(CodeChunk.repository_id == v_repo.id).all()
            f_chunks = db.query(CodeChunk).filter(CodeChunk.repository_id == f_repo.id).all()

            self.assertGreaterEqual(len(v_chunks), 10)
            self.assertGreaterEqual(len(f_chunks), 3)
        finally:
            db.close()

    def test_direct_module_imports_patched_to_isolated_dependencies(self):
        """
        Verify that modules importing SessionLocal or get_vector_store directly
        are patched to use isolated dependencies while isolation is active.
        """
        # 1. app.services.ingestion_service.SessionLocal
        self.assertIs(
            ingestion_service.SessionLocal,
            self.env["SessionLocal"],
            "ingestion_service.SessionLocal not patched to isolated TestSessionLocal",
        )
        test_session = ingestion_service.SessionLocal()
        try:
            self.assertEqual(str(test_session.bind.url), str(self.env["engine"].url))
        finally:
            test_session.close()

        # 2. app.services.ingestion_service.get_vector_store
        self.assertIs(
            ingestion_service.get_vector_store(),
            self.env["vector_store"],
            "ingestion_service.get_vector_store() not returning isolated vector store",
        )

        # 3. app.routes.chat.get_vector_store
        self.assertIs(
            chat_route.get_vector_store(),
            self.env["vector_store"],
            "chat_route.get_vector_store() not returning isolated vector store",
        )

        # 4. app.routes.demo.get_vector_store
        self.assertIs(
            demo_route.get_vector_store(),
            self.env["vector_store"],
            "demo_route.get_vector_store() not returning isolated vector store",
        )

        # 5. app.routes.repository.get_vector_store
        self.assertIs(
            repo_route.get_vector_store(),
            self.env["vector_store"],
            "repo_route.get_vector_store() not returning isolated vector store",
        )

    def test_restoration_of_original_references_after_deactivation(self):
        """
        Verify that deactivating RAG isolation restores all module-level aliases
        and global database/vector store references to their exact original values.
        """
        # Verify active first
        self.assertIs(ingestion_service.SessionLocal, self.env["SessionLocal"])
        self.assertIs(ingestion_service.get_vector_store(), self.env["vector_store"])

        # Deactivate
        deactivate_rag_isolation()
        try:
            # Check module aliases restored
            orig_ingestion_session = _original_module_aliases.get(("app.services.ingestion_service", "SessionLocal"))
            orig_ingestion_vs = _original_module_aliases.get(("app.services.ingestion_service", "get_vector_store"))
            orig_chat_vs = _original_module_aliases.get(("app.routes.chat", "get_vector_store"))
            orig_demo_vs = _original_module_aliases.get(("app.routes.demo", "get_vector_store"))
            orig_repo_vs = _original_module_aliases.get(("app.routes.repository", "get_vector_store"))

            self.assertIs(ingestion_service.SessionLocal, orig_ingestion_session)
            self.assertIs(ingestion_service.get_vector_store, orig_ingestion_vs)
            self.assertIs(chat_route.get_vector_store, orig_chat_vs)
            self.assertIs(demo_route.get_vector_store, orig_demo_vs)
            self.assertIs(repo_route.get_vector_store, orig_repo_vs)

            # Check core singletons restored
            self.assertIs(app.database.SessionLocal, _original_state["SessionLocal"])
            self.assertIs(app.database.engine, _original_state["engine"])
            self.assertIs(app.ai.vector_store.get_vector_store, _original_state["get_vector_store"])
        finally:
            # Reactivate for subsequent tests
            activate_rag_isolation()

        # Confirm reactivated
        self.assertIs(ingestion_service.SessionLocal, self.env["SessionLocal"])
        self.assertIs(ingestion_service.get_vector_store(), self.env["vector_store"])

    def test_cleanup_after_failed_initialization(self):
        """
        Verify that if setup fails midway (e.g. error loading seed data or creating store),
        the temporary directory is cleaned up and not orphaned on Windows.
        """
        # Temporarily teardown active env so get_isolated_rag_env will initialize afresh
        teardown_isolated_rag_env()

        created_temp_dirs = []
        orig_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created_temp_dirs.append(d)
            return d

        try:
            with patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp):
                with patch("tests.rag_fixtures.ChromaManager", side_effect=RuntimeError("Simulated Chroma error")):
                    with self.assertRaises(RuntimeError):
                        get_isolated_rag_env()

            self.assertEqual(len(created_temp_dirs), 1)
            failed_dir = created_temp_dirs[0]
            self.assertFalse(
                os.path.exists(failed_dir),
                f"Failed setup leaked temporary directory: {failed_dir}",
            )
        finally:
            # Recreate isolated environment for class
            self.__class__.env = get_isolated_rag_env()
            self.__class__.fixtures = ensure_rag_fixtures()

    def test_isolation_not_modifying_normal_database_or_chroma_data(self):
        """
        Verify that fixture operations do not mutate or write to normal development
        database or normal ChromaDB persistence directory.
        """
        normal_db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        normal_db_full = os.path.abspath(normal_db_path)
        normal_chroma_full = os.path.abspath(settings.CHROMA_PERSIST_DIRECTORY)

        initial_db_mtime = None
        initial_db_size = None
        if os.path.exists(normal_db_full):
            initial_db_mtime = os.path.getmtime(normal_db_full)
            initial_db_size = os.path.getsize(normal_db_full)

        # Perform mutations in isolated environment
        db = get_rag_session()
        test_repo_id = "test-isolation-probe-repo"
        try:
            repo = Repository(
                id=test_repo_id,
                user_id="probe_user",
                url="https://github.com/repolens/probe",
                owner="repolens",
                name="probe",
            )
            db.add(repo)
            db.commit()
        finally:
            db.close()

        # Add collection to isolated Chroma
        vs = get_rag_vector_store()
        vs.safe_replace_collection(test_repo_id, [{"chunk_id": "c1", "content": "print('probe')", "file_path": "probe.py", "start_line": 1, "end_line": 1}])

        # Verify probe exists in isolated DB and isolated Chroma
        probe_db = get_rag_session()
        try:
            self.assertIsNotNone(probe_db.query(Repository).filter(Repository.id == test_repo_id).first())
        finally:
            probe_db.close()

        probe_col_name = vs._get_collection_name(test_repo_id)
        self.assertGreater(vs.client.get_collection(probe_col_name).count(), 0)

        # Verify normal DB was NOT touched
        if initial_db_mtime is not None:
            self.assertEqual(os.path.getmtime(normal_db_full), initial_db_mtime)
            self.assertEqual(os.path.getsize(normal_db_full), initial_db_size)

        # Verify normal Chroma directory was NOT populated with probe collection
        if os.path.exists(normal_chroma_full):
            # Normal chroma collections dir must not contain the probe collection
            self.assertFalse(
                os.path.exists(os.path.join(normal_chroma_full, probe_col_name)),
                "Normal Chroma persistence dir contains isolated test collection!",
            )

    def test_stalled_job_timestamp_timezone_aware_and_naive(self):
        """
        Verify get_analysis_status handles both timezone-aware and timezone-naive
        updated_at datetimes without TypeError, while preserving the >300s stalled threshold.
        """
        mock_user = MagicMock(spec=User)
        mock_user.id = "user-123"

        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = "repo-123"
        mock_repo.status = "processing"

        def create_mock_job(updated_at_val, status="processing"):
            job = MagicMock(spec=AnalysisJob)
            job.id = "job-123"
            job.repository_id = mock_repo.id
            job.status = status
            job.current_stage = "embedding"
            job.progress_percent = 40
            job.message = "Generating embeddings"
            job.error_details = None
            job.updated_at = updated_at_val
            return job

        # 1. Timezone-naive datetime (>300 seconds ago) -> stalled is True
        naive_stalled_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=350)
        job_naive_stalled = create_mock_job(naive_stalled_time)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = job_naive_stalled
        mock_db.query.return_value.filter.return_value.first.return_value = job_naive_stalled
        with patch("app.routes.repository.get_repository_by_id", return_value=mock_repo):
            resp = repo_route.get_analysis_status(mock_repo.id, db=mock_db, current_user=mock_user)
            self.assertTrue(resp.is_stalled, "Expected naive >300s job to be stalled")
            self.assertIn("Warning: No progress reported for >5m", resp.message)

        # 2. Timezone-aware UTC datetime (>300 seconds ago) -> stalled is True, no TypeError
        aware_utc_stalled_time = datetime.now(timezone.utc) - timedelta(seconds=350)
        job_aware_stalled = create_mock_job(aware_utc_stalled_time)

        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = job_aware_stalled
        mock_db.query.return_value.filter.return_value.first.return_value = job_aware_stalled
        with patch("app.routes.repository.get_repository_by_id", return_value=mock_repo):
            resp = repo_route.get_analysis_status(mock_repo.id, db=mock_db, current_user=mock_user)
            self.assertTrue(resp.is_stalled, "Expected aware UTC >300s job to be stalled")
            self.assertIn("Warning: No progress reported for >5m", resp.message)

        # 3. Timezone-aware non-UTC datetime (e.g. +05:30) (>300 seconds ago) -> stalled is True, no TypeError
        custom_tz = timezone(timedelta(hours=5, minutes=30))
        aware_custom_tz_time = datetime.now(custom_tz) - timedelta(seconds=350)
        job_custom_tz = create_mock_job(aware_custom_tz_time)

        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = job_custom_tz
        mock_db.query.return_value.filter.return_value.first.return_value = job_custom_tz
        with patch("app.routes.repository.get_repository_by_id", return_value=mock_repo):
            resp = repo_route.get_analysis_status(mock_repo.id, db=mock_db, current_user=mock_user)
            self.assertTrue(resp.is_stalled, "Expected non-UTC aware >300s job to be stalled")
            self.assertIn("Warning: No progress reported for >5m", resp.message)

        # 4. Fresh timezone-aware datetime (10 seconds ago) -> stalled is False
        aware_fresh_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        job_aware_fresh = create_mock_job(aware_fresh_time)

        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = job_aware_fresh
        mock_db.query.return_value.filter.return_value.first.return_value = job_aware_fresh
        with patch("app.routes.repository.get_repository_by_id", return_value=mock_repo):
            resp = repo_route.get_analysis_status(mock_repo.id, db=mock_db, current_user=mock_user)
            self.assertFalse(resp.is_stalled, "Expected aware fresh job to not be stalled")
            self.assertEqual(resp.message, "Generating embeddings")

        # 5. Fresh timezone-naive datetime (10 seconds ago) -> stalled is False
        naive_fresh_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=10)
        job_naive_fresh = create_mock_job(naive_fresh_time)

        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = job_naive_fresh
        mock_db.query.return_value.filter.return_value.first.return_value = job_naive_fresh
        with patch("app.routes.repository.get_repository_by_id", return_value=mock_repo):
            resp = repo_route.get_analysis_status(mock_repo.id, db=mock_db, current_user=mock_user)
            self.assertFalse(resp.is_stalled, "Expected naive fresh job to not be stalled")
            self.assertEqual(resp.message, "Generating embeddings")

    def test_test_module_alias_restoration_with_distinct_original(self):
        """
        Verify that test modules with custom/distinct original SessionLocal or get_vector_store
        attributes (different from core app.database/app.ai.vector_store) are captured and exactly
        restored, and that modules imported after isolation starts do not overwrite previously
        captured originals.
        """
        import types
        import sys
        from tests.rag_fixtures import (
            _rebind_dependencies,
            _original_test_module_aliases,
        )

        mod_name_distinct = "tests.test_rag_distinct_alias_dummy"
        mod_name_late = "tests.test_rag_late_imported_dummy"

        custom_session_sentinel = object()
        custom_vs_sentinel = lambda: "custom_vector_store"

        dummy_distinct = types.ModuleType(mod_name_distinct)
        dummy_distinct.SessionLocal = custom_session_sentinel
        dummy_distinct.get_vector_store = custom_vs_sentinel
        sys.modules[mod_name_distinct] = dummy_distinct

        try:
            # 1. Activate isolation and verify distinct module is patched
            activate_rag_isolation()
            self.assertIs(dummy_distinct.SessionLocal, self.env["SessionLocal"])
            self.assertIs(dummy_distinct.get_vector_store(), self.env["vector_store"])

            # 2. Simulate late-imported module arriving AFTER isolation started
            dummy_late = types.ModuleType(mod_name_late)
            # Its module-level import gets the already-isolated SessionLocal & vector_store
            dummy_late.SessionLocal = self.env["SessionLocal"]
            dummy_late.get_vector_store = lambda: self.env["vector_store"]
            sys.modules[mod_name_late] = dummy_late

            # Re-run rebind dependencies (e.g. repeated fixture call)
            _rebind_dependencies(self.env)

            # Assert previously captured distinct original was NOT overwritten
            self.assertIs(
                _original_test_module_aliases.get((mod_name_distinct, "SessionLocal")),
                custom_session_sentinel,
            )
            self.assertIs(
                _original_test_module_aliases.get((mod_name_distinct, "get_vector_store")),
                custom_vs_sentinel,
            )

            # 3. Deactivate isolation
            deactivate_rag_isolation()

            # 4. Verify distinct module is restored to its unique original (NOT app.database.SessionLocal)
            self.assertIs(dummy_distinct.SessionLocal, custom_session_sentinel)
            self.assertIs(dummy_distinct.get_vector_store, custom_vs_sentinel)

            # 5. Verify late module is restored to default core original
            self.assertIs(dummy_late.SessionLocal, _original_state["SessionLocal"])
            self.assertIs(dummy_late.get_vector_store, _original_state["get_vector_store"])

        finally:
            # Clean up dummy modules
            sys.modules.pop(mod_name_distinct, None)
            sys.modules.pop(mod_name_late, None)
            _original_test_module_aliases.pop((mod_name_distinct, "SessionLocal"), None)
            _original_test_module_aliases.pop((mod_name_distinct, "get_vector_store"), None)
            _original_test_module_aliases.pop((mod_name_late, "SessionLocal"), None)
            _original_test_module_aliases.pop((mod_name_late, "get_vector_store"), None)
            # Ensure isolation reactivated for subsequent tests
            activate_rag_isolation()

    def test_close_chroma_client_fallback_to_system_stop(self):
        """
        Verify that _close_chroma_client attempts client._system.stop() if client.close()
        raises an exception, logging the failure without raising or masking errors.
        """
        from tests.rag_fixtures import _close_chroma_client

        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Simulated close() failure")
        mock_client._system = MagicMock()
        mock_client._system.stop = MagicMock()

        # Should not raise exception
        _close_chroma_client(mock_client)

        mock_client.close.assert_called_once()
        mock_client._system.stop.assert_called_once()

    def test_pytorch_thread_limiting_robustness_on_runtime_error(self):
        """
        Verify that RuntimeError when setting PyTorch intra-op or inter-op threads
        is logged as a warning without preventing the embedding model from loading,
        while genuine model loading errors still raise RuntimeError.
        """
        import sys
        from app.ai.embeddings import LocalEmbedding
        import app.ai.embeddings as emb_mod

        # 1. Simulate PyTorch set_num_threads raising RuntimeError
        mock_torch = MagicMock()
        mock_torch.set_num_threads.side_effect = RuntimeError("Cannot set intraop threads")
        mock_torch.set_num_interop_threads.side_effect = RuntimeError("Cannot set interop threads")

        mock_st = MagicMock()
        mock_model_instance = MagicMock()
        mock_st.return_value = mock_model_instance

        # Save previous cached model if any
        prev_cached = emb_mod._cached_local_model
        emb_mod._cached_local_model = None

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}):
                with patch("app.ai.embeddings._get_sentence_transformer_cls", return_value=mock_st):
                    embedding_provider = LocalEmbedding(model_name="all-MiniLM-L6-v2")
                    loaded_model = embedding_provider._load_model()
                    self.assertIs(loaded_model, mock_model_instance)
                    mock_torch.set_num_threads.assert_called_with(1)
                    mock_torch.set_num_interop_threads.assert_called_with(1)

            # 2. Verify that genuine model instantiation failure is NOT silently hidden
            emb_mod._cached_local_model = None
            mock_failing_st = MagicMock(side_effect=Exception("Failed to download weights"))
            with patch("app.ai.embeddings._get_sentence_transformer_cls", return_value=mock_failing_st):
                embedding_provider = LocalEmbedding(model_name="failing-model")
                with self.assertRaises(RuntimeError) as ctx:
                    embedding_provider._load_model()
                self.assertIn("Failed to load or download local embedding model", str(ctx.exception))
        finally:
            emb_mod._cached_local_model = prev_cached
