import os
import sys
import unittest
import re
from fastapi.testclient import TestClient

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.utils.security import create_access_token
from app.database import SessionLocal
from app.models.user import User
from app.models.repository import Repository
from app.ai.rag import RAGEngine
from app.ai.vector_store import get_vector_store
from app.ai.llm import MockLLM
from app.ai.call_chain import ASTCallChainTracer

class TestRepoLensAcceptanceSuite(unittest.TestCase):
    """
    Phase 3: Finite acceptance test suite covering the entire core workflow.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.user = cls.db.query(User).first()
        if not cls.user:
            cls.user = User(
                id="test-user-uuid",
                username="testuser",
                email="testuser@example.com",
                hashed_password="hashed_test_pass"
            )
            cls.db.add(cls.user)
            cls.db.commit()

        cls.token = create_access_token({"sub": cls.user.id, "username": cls.user.username})
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}

        # Repositories
        cls.vehicle_repo = cls.db.query(Repository).filter(Repository.name == "vehicle-damage-detection").first()
        cls.fastapi_repo = cls.db.query(Repository).filter(Repository.name == "fastapi").first()

        cls.vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.engine = RAGEngine(vector_store=cls.vector_store, llm_provider=cls.mock_llm)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # 1. Backend health and core API endpoints
    def test_01_backend_health_and_root(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("app"), "RepoLens AI")
        self.assertEqual(data.get("status"), "online")

    # 2. Existing authentication and ownership protections
    def test_02_authentication_and_ownership_protection(self):
        # Unauthenticated request fails
        resp_unauth = self.client.get("/repositories")
        self.assertEqual(resp_unauth.status_code, 401)

        # Authenticated user can list repositories
        resp_auth = self.client.get("/repositories", headers=self.auth_headers)
        self.assertEqual(resp_auth.status_code, 200)
        self.assertIsInstance(resp_auth.json(), list)

        # Accessing non-existent repo or other user's repo returns 404
        resp_fake = self.client.get("/repositories/00000000-0000-0000-0000-000000000000", headers=self.auth_headers)
        self.assertEqual(resp_fake.status_code, 404)

    # 3. Repository selection and repository isolation
    def test_03_repository_selection_and_file_isolation(self):
        if not self.vehicle_repo or not self.fastapi_repo:
            self.skipTest("Indexed repositories not found in database.")

        # Fetch vehicle repo
        v_resp = self.client.get(f"/repositories/{self.vehicle_repo.id}", headers=self.auth_headers)
        self.assertEqual(v_resp.status_code, 200)
        self.assertEqual(v_resp.json().get("name"), "vehicle-damage-detection")

        # Fetch fastapi repo
        f_resp = self.client.get(f"/repositories/{self.fastapi_repo.id}", headers=self.auth_headers)
        self.assertEqual(f_resp.status_code, 200)
        self.assertEqual(f_resp.json().get("name"), "fastapi")

        # Verify files are isolated
        v_files = self.client.get(f"/repositories/{self.vehicle_repo.id}/files", headers=self.auth_headers).json()
        f_files = self.client.get(f"/repositories/{self.fastapi_repo.id}/files", headers=self.auth_headers).json()
        v_paths = {f["file_path"] for f in v_files.get("flat_files", [])}
        f_paths = {f["file_path"] for f in f_files.get("flat_files", [])}

        self.assertIn("app.py", v_paths)
        self.assertNotIn("app.py", f_paths)
        self.assertTrue(any("fastapi" in p for p in f_paths))

    # 4. Repository indexing success and failure handling
    def test_04_indexing_validation_and_status(self):
        # Invalid repository URL fails validation
        resp_bad = self.client.post("/repositories", json={"url": "not-a-valid-github-url"}, headers=self.auth_headers)
        self.assertEqual(resp_bad.status_code, 400)

        # Non-github domain fails validation
        resp_domain = self.client.post("/repositories", json={"url": "https://gitlab.com/user/repo"}, headers=self.auth_headers)
        self.assertEqual(resp_domain.status_code, 400)

        # Checking status of existing repo returns status payload
        if self.vehicle_repo:
            status_resp = self.client.get(f"/repositories/{self.vehicle_repo.id}/analysis-status", headers=self.auth_headers)
            self.assertIn(status_resp.status_code, (200, 404))
            if status_resp.status_code == 200:
                s_data = status_resp.json()
                self.assertIn("status", s_data)
                self.assertIn("progress_percent", s_data)

    # 5. Correct repository-purpose answers
    def test_05_repository_purpose_grounding(self):
        if not self.vehicle_repo:
            self.skipTest("Vehicle repo not found in database.")

        res = self.engine.ask_question(
            repo_id=self.vehicle_repo.id,
            question="What is the purpose of this repository?",
            top_k=5
        )
        answer = res.get("answer", "")
        # Must be vehicle damage specific and grounded in README
        self.assertTrue(any(term in answer.lower() for term in ("vehicle", "damage", "yolo", "insurance")))
        self.assertIn("README.md", answer)
        self.assertFalse("fastapi" in answer.lower())

        # Exact regression test: FastAPI purpose query with negative dependency constraint
        if self.fastapi_repo:
            fastapi_purpose_q = "Explain the purpose of the FastAPI repository in 3 simple sentences. Use evidence from README.md and do not just list dependencies."
            self.assertEqual(self.engine.classify_question(fastapi_purpose_q), "purpose")

            f_res = self.engine.ask_question(
                repo_id=self.fastapi_repo.id,
                question=fastapi_purpose_q,
                top_k=5
            )
            f_ans = f_res.get("answer", "")
            f_sources = [f"{s.get('file_path')}:{s.get('start_line')}-{s.get('end_line')}" for s in f_res.get("sources", [])]

            # Must explain purpose grounded in README.md introduction
            self.assertIn("FastAPI", f_ans)
            self.assertIn("README.md", f_ans)
            self.assertTrue(any("README.md" in s for s in f_sources))

            # Must NOT cite dependency manifests (pyproject.toml) as purpose evidence
            self.assertFalse(any("pyproject.toml" in s for s in f_sources))
            self.assertFalse(any("README.md:481-530" in s for s in f_sources))
            self.assertNotIn("pyproject.toml:41-90", f_ans)

    # 6. Correct dependency answers
    def test_06_dependency_analysis_grounding(self):
        if not self.vehicle_repo:
            self.skipTest("Vehicle repo not found in database.")

        # Route test
        dep_resp = self.client.get(f"/repositories/{self.vehicle_repo.id}/dependencies", headers=self.auth_headers)
        self.assertEqual(dep_resp.status_code, 200)
        dep_data = dep_resp.json()
        dep_names = [d["name"].lower() for d in dep_data.get("dependencies", [])]
        self.assertTrue(any(pkg in dep_names for pkg in ("ultralytics", "flask", "mysql-connector-python", "numpy", "pandas")))

        # RAG question test
        rag_res = self.engine.ask_question(
            repo_id=self.vehicle_repo.id,
            question="What dependencies and technologies does this project use?",
            top_k=5
        )
        rag_answer = rag_res.get("answer", "").lower()
        self.assertTrue(any(tech in rag_answer for tech in ("flask", "yolo", "ultralytics", "mysql", "python")))

    # 7. Execution-flow answers grounded in actual source
    def test_07_execution_flow_grounded_in_actual_source(self):
        if not self.fastapi_repo:
            self.skipTest("FastAPI repo not found in database.")

        res = self.engine.ask_question(
            repo_id=self.fastapi_repo.id,
            question="Explain the execution flow of fastapi/applications.py and cite the relevant source lines",
            top_k=5
        )
        answer = res.get("answer", "")
        # Must cite real methods and classes
        self.assertIn("FastAPI", answer)
        self.assertIn("fastapi/applications.py", answer)
        # Must not fabricate invented functions
        self.assertNotIn("app_initialization", answer)
        self.assertNotIn("routing.init_routes(app)", answer)
        self.assertNotIn("middleware.init_middleware(app)", answer)

    # 8. Direct and downstream call relationships with separate call-site & definition citations
    def test_08_call_chain_edges_and_citations(self):
        if not self.vehicle_repo:
            self.skipTest("Vehicle repo not found in database.")

        res = self.engine.ask_question(
            repo_id=self.vehicle_repo.id,
            question="Explain the actual call chain after YOLO detects vehicle damage",
            top_k=5
        )
        answer = res.get("answer", "")

        # Must identify dashboard as orchestrator
        self.assertIn("dashboard", answer)
        # Must distinguish call sites
        self.assertIn("[app.py:201]", answer)
        self.assertIn("[app.py:212]", answer)
        # Must cite definition spans
        self.assertIn("[app.py:296-333]", answer)
        # Must cite downstream helper
        self.assertIn("get_part_name_from_id", answer)
        self.assertIn("[app.py:318]", answer)
        # Must reject linear chaining of callees
        self.assertIn("do not call each other sequentially", answer)

    # 9. Cross-repository contamination prevention
    def test_09_cross_repository_contamination_prevention(self):
        if not self.vehicle_repo or not self.fastapi_repo:
            self.skipTest("Repositories not found in database.")

        # Question on vehicle repo
        v_res = self.engine.ask_question(
            repo_id=self.vehicle_repo.id,
            question="What is this repository about and what are its main components?",
            top_k=5
        )
        v_answer = v_res.get("answer", "")
        self.assertNotIn("Starlette", v_answer)
        self.assertNotIn("fastapi/applications.py", v_answer)

        # Question on fastapi repo
        f_res = self.engine.ask_question(
            repo_id=self.fastapi_repo.id,
            question="What is this repository about and what are its main components?",
            top_k=5
        )
        f_answer = f_res.get("answer", "")
        self.assertNotIn("vehicle damage", f_answer.lower())
        self.assertNotIn("ultralytics", f_answer.lower())

    # 10. Vehicle damage detection model and damaged parts query
    def test_10_vehicle_damage_detection_model_and_classes(self):
        if not self.vehicle_repo:
            self.skipTest("Vehicle repo not found in database.")

        model_q = "Which model does this repository use to detect vehicle damage, and what damaged parts can it identify?"
        self.assertEqual(self.engine.classify_question(model_q), "model")

        res = self.engine.ask_question(
            repo_id=self.vehicle_repo.id,
            question=model_q,
            top_k=4
        )
        answer = res.get("answer", "")
        sources = [f"{s.get('file_path')}:{s.get('start_line')}-{s.get('end_line')}" for s in res.get("sources", [])]

        # Must identify YOLOv8 model
        self.assertIn("yolov8", answer.lower())

        # Must identify supported damaged-part classes
        for part in ["Bonnet", "Bumper", "Door", "Windshield"]:
            self.assertIn(part.lower(), answer.lower())

        # Must cite README or code evidence, NOT internal damage cost calculation
        self.assertFalse(any("internal_damage_map.json" in s for s in sources))
        self.assertNotIn("internal_damage_map.json", answer)
        self.assertNotIn("calculate_internal_damage", answer)
        self.assertTrue(any("README.md" in s or "train_yolo.py" in s or "app.py" in s for s in sources))

    # 11. Purpose query conciseness, deduplication, and absence of raw HTML fragments
    def test_11_purpose_query_conciseness_and_no_malformed_html(self):
        if not self.fastapi_repo:
            self.skipTest("FastAPI repo not found in database.")

        res = self.engine.ask_question(
            repo_id=self.fastapi_repo.id,
            question="What is the purpose of this repository?",
            top_k=5
        )
        answer = res.get("answer", "")
        sources = [f"{s.get('file_path')}:{s.get('start_line')}-{s.get('end_line')}" for s in res.get("sources", [])]

        # 1. Must identify FastAPI and explain purpose
        self.assertIn("fastapi", answer.lower())

        # 2. Must be concise (between 2 and 5 sentences)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
        self.assertGreaterEqual(len(sentences), 2)
        self.assertLessEqual(len(sentences), 5)

        # 3. Must NOT contain raw HTML fragments (e.g. <dfn title="...">Completion</dfn>)
        self.assertFalse(bool(re.search(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*?)?>", answer)))

        # 4. Must NOT contain broken anchor markdown links or raw anchors (e.g. (#performance))
        self.assertNotIn("(#performance)", answer)
        self.assertFalse(bool(re.search(r"\]\(#[^\)]*\)", answer)))

        # 5. Must NOT contain duplicate paragraphs
        paragraphs = [p.strip().lower() for p in re.split(r"\n{2,}", answer.strip()) if p.strip()]
        self.assertEqual(len(paragraphs), len(set(paragraphs)), "Found duplicate paragraphs in response")

        # 6. Must cite README.md
        self.assertTrue(any("README.md" in s for s in sources))
        self.assertIn("README.md", answer)

if __name__ == "__main__":
    unittest.main()

