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


class TestRAGCrossRepoRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.engine = RAGEngine(vector_store=cls.vector_store, llm_provider=cls.mock_llm)
        cls.fastapi_repo_id = "31806550-364e-4715-8904-55414d25691d"
        cls.vehicle_repo_id = "3f4d8af2-f8d0-4293-ae4e-885a4cbf3221"

    def test_fastapi_purpose_answer_no_vehicle_damage(self):
        """Regression test: FastAPI purpose answer must not contain vehicle damage instructions or claims."""
        question = "What is the main purpose of this repository, and what problem does it solve?"
        res = self.engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Answer must not contain vehicle damage claims
        self.assertNotIn("vehicle", answer.lower())
        self.assertNotIn("damage", answer.lower())
        self.assertNotIn("yolo", answer.lower())
        self.assertNotIn("insurance", answer.lower())
        self.assertNotIn("repair cost", answer.lower())
        self.assertNotIn("mysql", answer.lower())

        # Answer should describe FastAPI framework
        self.assertTrue(
            "fastapi" in answer.lower() or "api" in answer.lower() or "web framework" in answer.lower(),
            f"Expected FastAPI description, got: {answer}"
        )

        # Sources must belong to FastAPI repository
        self.assertGreater(len(sources), 0)
        for s in sources:
            fp = s.get("file_path", "").lower()
            self.assertFalse("insurance_predictor" in fp, f"Unexpected vehicle file cited: {fp}")
            self.assertFalse(fp == "app.py", f"Unexpected vehicle app.py cited in FastAPI: {fp}")

    def test_fastapi_dependencies_answer_no_yolo_or_vehicle_libraries(self):
        """Regression test: FastAPI dependency answer must not introduce YOLOv8, OpenCV, or vehicle damage libraries."""
        question = "Which technologies and dependencies does it use?"
        res = self.engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Answer must not mention vehicle-damage libraries
        self.assertNotIn("yolov8", answer.lower())
        self.assertNotIn("ultralytics", answer.lower())
        self.assertNotIn("opencv", answer.lower())
        self.assertNotIn("mysql", answer.lower())
        self.assertNotIn("internal_damage", answer.lower())

        # Citations must belong to FastAPI
        self.assertGreater(len(sources), 0)
        for s in sources:
            fp = s.get("file_path", "").lower()
            self.assertFalse("insurance_predictor" in fp)

    def test_fastapi_execution_flow_no_vehicle_symbols(self):
        """Regression test: FastAPI execution-flow answer must not mention predict_insurance, calculate_internal_damage, or estimate.html."""
        question = "What is the execution flow of app.py?"
        res = self.engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Must not contain vehicle damage symbols or templates
        self.assertNotIn("predict_insurance", answer)
        self.assertNotIn("calculate_internal_damage", answer)
        self.assertNotIn("estimate.html", answer)
        self.assertNotIn("get_part_prices", answer)
        self.assertNotIn("predictive_analytics", answer)
        self.assertNotIn("generate_recommendation", answer)

        # Citations must not point to vehicle repo files
        for s in sources:
            fp = s.get("file_path", "").lower()
            self.assertFalse("insurance_predictor" in fp)

    def test_fastapi_call_chain_uses_active_repo_symbols(self):
        """Regression test: FastAPI call chain query must not mention YOLO, vehicle damage, or vehicle pipeline functions."""
        question = "What functions are called after request handling?"
        res = self.engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Must not inject vehicle damage pipeline functions
        self.assertNotIn("predict_insurance", answer)
        self.assertNotIn("calculate_internal_damage", answer)
        self.assertNotIn("get_part_prices", answer)
        self.assertNotIn("yolo", answer.lower())

        for s in sources:
            fp = s.get("file_path", "").lower()
            self.assertFalse("insurance_predictor" in fp)

    def test_vehicle_damage_repo_still_answers_correctly(self):
        """Regression test: The existing vehicle-damage repository can still answer its purpose and call chain correctly."""
        # 1. Purpose
        q_purpose = "What is the main purpose of this repository, and what problem does it solve?"
        res_p = self.engine.ask_question(self.vehicle_repo_id, q_purpose, top_k=5)
        ans_p = res_p.get("answer", "")
        self.assertIn("vehicle damage", ans_p.lower())
        self.assertIn("yolo", ans_p.lower())

        # 2. Call chain
        q_chain = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res_c = self.engine.ask_question(self.vehicle_repo_id, q_chain, top_k=5)
        ans_c = res_c.get("answer", "")
        sources_c = res_c.get("sources", [])

        expected_funcs = [
            "get_part_prices",
            "calculate_damage_severity",
            "predictive_analytics",
            "calculate_internal_damage",
            "predict_insurance",
            "generate_recommendation",
            "get_part_name_from_id"
        ]
        for fn in expected_funcs:
            self.assertIn(fn, ans_c, f"Expected vehicle function '{fn}' missing from answer")

        source_files = {s["file_path"] for s in sources_c}
        self.assertIn("app.py", source_files)
        self.assertIn("insurance_predictor.py", source_files)


if __name__ == "__main__":
    unittest.main()
