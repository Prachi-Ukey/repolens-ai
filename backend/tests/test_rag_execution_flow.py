import os
import sys
import unittest

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import RAGEngine
from app.ai.llm import MockLLM
from app.ai.vector_store import get_vector_store
from tests.rag_fixtures import ensure_rag_fixtures, deactivate_rag_isolation


class TestRAGExecutionFlowRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ensure_rag_fixtures()
        cls.vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.engine = RAGEngine(vector_store=cls.vector_store, llm_provider=cls.mock_llm)
        cls.fastapi_repo_id = fixtures["fastapi_repo_id"]
        cls.vehicle_repo_id = fixtures["vehicle_repo_id"]

    @classmethod
    def tearDownClass(cls):
        deactivate_rag_isolation()

    def test_reject_invented_function_names_and_calls(self):
        """Task 5.1: Reject invented function names and call relationships in fastapi/applications.py."""
        q = "Explain the execution flow of fastapi/applications.py and cite the relevant source lines"
        res = self.engine.ask_question(self.fastapi_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Must NOT contain invented symbols or calls
        self.assertNotIn("app_initialization", answer)
        self.assertNotIn("init_routes", answer)
        self.assertNotIn("init_middleware", answer)
        self.assertNotIn("routing.init_routes", answer)
        self.assertNotIn("middleware.init_middleware", answer)

        # Must mention actual FastAPI class and methods
        self.assertIn("FastAPI", answer)
        self.assertTrue(
            "__init__" in answer or "setup" in answer or "build_middleware_stack" in answer or "__call__" in answer,
            f"Expected actual FastAPI methods in answer, got:\n{answer}"
        )

    def test_distinguish_imports_from_executable_calls(self):
        """Task 5.2: Correctly distinguish imports/module-level definitions from executable calls."""
        q = "Explain the execution flow of fastapi/applications.py and cite the relevant source lines"
        res = self.engine.ask_question(self.fastapi_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        # The answer must explicitly clarify that module imports contain declarations, not runtime calls
        ans_lower = answer.lower()
        self.assertTrue(
            "import" in ans_lower or "declaration" in ans_lower,
            f"Expected explicit distinction of imports in answer, got:\n{answer}"
        )
        self.assertFalse(
            "app_initialization" in answer,
            "Imports must not be labeled as an app_initialization function"
        )

    def test_citations_support_associated_claims(self):
        """Task 5.3: Ensure citations support the specific claims they accompany."""
        q = "Explain the execution flow of fastapi/applications.py and cite the relevant source lines"
        res = self.engine.ask_question(self.fastapi_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Lines 1-34 or 1-44 must NOT be repeatedly cited for route/middleware/asgi claims
        for line in answer.splitlines():
            line_low = line.lower()
            if any(k in line_low for k in ["route", "router", "middleware", "asgi", "__call__"]):
                self.assertNotIn(
                    "[fastapi/applications.py:1-34]",
                    line,
                    f"Imports line 1-34 cited for runtime claim: {line}"
                )

        # Citations must be distinct across claims rather than repeatedly citing lines 1-34
        source_ranges = {(s["start_line"], s["end_line"]) for s in sources}
        self.assertGreater(
            len(source_ranges), 1,
            f"Expected multiple distinct cited ranges, got: {source_ranges}"
        )

        for s in sources:
            self.assertLessEqual(s["start_line"], s["end_line"])
            self.assertEqual(s["file_path"], "fastapi/applications.py")

    def test_requested_import_range_explains_imports_only(self):
        """Task 5.3b: If requested line range only contains imports, describe what it actually shows."""
        q = "Explain what lines 1-34 of fastapi/applications.py do and cite the source lines"
        res = self.engine.ask_question(self.fastapi_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        self.assertNotIn("app_initialization", answer)
        self.assertNotIn("init_routes", answer)
        self.assertNotIn("init_middleware", answer)

    def test_preserves_fastapi_purpose_and_dependencies(self):
        """Task 5.4a: Preserve existing FastAPI purpose and dependency behavior."""
        q_purp = "What is the main purpose of this repository, and what problem does it solve?"
        res_purp = self.engine.ask_question(self.fastapi_repo_id, q_purp, top_k=5)
        ans_purp = res_purp.get("answer", "").lower()
        self.assertNotIn("vehicle", ans_purp)
        self.assertNotIn("damage", ans_purp)
        self.assertTrue("fastapi" in ans_purp or "web framework" in ans_purp)

        q_dep = "Which technologies and dependencies does it use?"
        res_dep = self.engine.ask_question(self.fastapi_repo_id, q_dep, top_k=5)
        ans_dep = res_dep.get("answer", "").lower()
        self.assertNotIn("yolo", ans_dep)
        self.assertNotIn("opencv", ans_dep)

    def test_preserves_vehicle_damage_execution_flow(self):
        """Task 5.4b: Preserve existing vehicle damage repository execution flow behavior."""
        q = "Explain the execution flow of app.py"
        res = self.engine.ask_question(self.vehicle_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Must include vehicle pipeline functions
        self.assertIn("dashboard", answer)
        self.assertIn("get_part_prices", answer)
        self.assertIn("calculate_damage_severity", answer)

        # Must not attribute template rendering or insurance to predictive_analytics
        ans_lower = answer.lower()
        self.assertNotRegex(ans_lower, r"predictive_analytics[^\.\n]*?(?:renders|rendering)[^\.\n]*?template")
        self.assertNotRegex(ans_lower, r"predictive_analytics[^\.\n]*?(?:calculates|predicts)[^\.\n]*?insurance")


if __name__ == "__main__":
    unittest.main()
