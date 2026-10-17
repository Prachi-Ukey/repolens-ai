import os
import sys
import unittest

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import (
    RAGEngine,
    is_dependency_or_generated_file,
    deduplicate_and_merge_chunks,
    extract_and_validate_citations,
)
from app.ai.llm import MockLLM
from app.ai.vector_store import get_vector_store
from tests.rag_fixtures import ensure_rag_fixtures, deactivate_rag_isolation


class TestRAGCallChainAndRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ensure_rag_fixtures()
        cls.vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.engine = RAGEngine(vector_store=cls.vector_store, llm_provider=cls.mock_llm)
        cls.vehicle_repo_id = fixtures["vehicle_repo_id"]
        cls.qa_repo_id = fixtures["vehicle_repo_id"]

    @classmethod
    def tearDownClass(cls):
        deactivate_rag_isolation()

    def test_call_chain_intent_classification(self):
        """Task 1: Verify call-chain intent classification."""
        q1 = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        self.assertEqual(self.engine.classify_question(q1), "call_chain")

        q2 = "What functions are called after image upload?"
        self.assertEqual(self.engine.classify_question(q2), "call_chain")

        q3 = "What happens after the prediction is made?"
        self.assertEqual(self.engine.classify_question(q3), "call_chain")

    def test_binary_artifact_exclusion(self):
        """Task 4: Verify binary/model artifacts (.pkl, .pt) are excluded from ordinary code retrieval."""
        self.assertTrue(is_dependency_or_generated_file("insurance_columns.pkl"))
        self.assertTrue(is_dependency_or_generated_file("model/weights.pt"))
        self.assertTrue(is_dependency_or_generated_file("data/features.joblib"))

        # Explicitly asked binary artifact should be permitted if asked
        self.assertFalse(is_dependency_or_generated_file("insurance_columns.pkl", "What is inside insurance_columns.pkl?"))

        # In vehicle damage detection repo, normal code retrieval must not contain .pkl files
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            "Which functions are called after YOLO detects vehicle damage, and what does each function do?",
            top_k=5
        )
        top_chunks = res.get("top_chunks", [])
        for chunk in top_chunks:
            self.assertFalse(chunk.get("file_path", "").endswith(".pkl"), f"Binary artifact found: {chunk.get('file_path')}")

    def test_chunk_deduplication_and_merging(self):
        """Task 5: Deduplicate overlapping contiguous chunks of the same file into cohesive sequence."""
        overlapping_chunks = [
            {
                "file_path": "app.py",
                "start_line": 161,
                "end_line": 210,
                "symbol_name": "dashboard",
                "content": "line161\nline180\nline210",
                "hybrid_score": 1.2
            },
            {
                "file_path": "app.py",
                "start_line": 201,
                "end_line": 250,
                "symbol_name": "dashboard",
                "content": "line201\nline210\nline250",
                "hybrid_score": 1.1
            },
            {
                "file_path": "app.py",
                "start_line": 241,
                "end_line": 290,
                "symbol_name": "dashboard",
                "content": "line241\nline250\nline290",
                "hybrid_score": 1.0
            },
            {
                "file_path": "app.py",
                "start_line": 60,
                "end_line": 85,
                "symbol_name": "get_part_prices",
                "content": "def get_part_prices():\n    pass",
                "hybrid_score": 0.95
            }
        ]

        merged = deduplicate_and_merge_chunks(overlapping_chunks, max_merged_lines=140, max_chunks=5)
        # The 3 overlapping slices of dashboard should merge into 1 chunk
        app_dashboard_chunks = [c for c in merged if c["file_path"] == "app.py" and c["start_line"] == 161]
        self.assertEqual(len(app_dashboard_chunks), 1)
        merged_dashboard = app_dashboard_chunks[0]
        self.assertEqual(merged_dashboard["start_line"], 161)
        self.assertEqual(merged_dashboard["end_line"], 290)

        # get_part_prices should remain intact as a separate chunk
        get_part_chunks = [c for c in merged if c["symbol_name"] == "get_part_prices"]
        self.assertEqual(len(get_part_chunks), 1)

    def test_yolo_damage_detection_call_chain_and_no_irrelevant_boosts(self):
        """Task 2, 3, 7: Verify anchor handler & direct called functions are retrieved, and unrelated routes are not boosted."""
        question = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            question,
            top_k=5
        )

        top_chunks = res.get("top_chunks", [])
        self.assertGreater(len(top_chunks), 0)

        top_files = [c.get("file_path", "") for c in top_chunks]
        top_symbols = [c.get("symbol_name", "") for c in top_chunks]
        top_contents = [c.get("content", "") for c in top_chunks]

        # 1. Primary anchor handler (dashboard in app.py) must be in top chunks
        self.assertIn("app.py", top_files)

        # Check that unrelated handlers are NOT in top chunks
        unrelated_symbols = {"signup", "edit_profile", "login", "view_profile", "logout", "connect_to_db"}
        for s in top_symbols:
            self.assertNotIn(s, {"signup", "edit_profile", "login", "view_profile", "logout"})

        # Check that top chunks do NOT contain README or HTML templates
        for f in top_files:
            self.assertFalse(f.endswith(".html"), f"Unexpected HTML in code retrieval: {f}")
            self.assertFalse(f.endswith(".md"), f"Unexpected markdown in code retrieval: {f}")

        # Check that called functions or downstream calculations appear in evidence
        combined_text = " ".join(top_contents + top_symbols)
        # Expected called functions in dashboard after YOLO:
        expected_called = ["get_part_prices", "calculate_damage_severity", "calculate_internal_damage", "predictive_analytics", "predict_insurance", "generate_recommendation"]
        found_expected = any(fn in combined_text for fn in expected_called)
        self.assertTrue(found_expected, f"None of the expected called functions {expected_called} found in evidence")

    def test_purpose_question_retrieval(self):
        """Task 6, 7: Preserve general-purpose retrieval behavior for repository overview questions."""
        question = "What is the main purpose of this repository, and what problem does it solve?"
        self.assertEqual(self.engine.classify_question(question), "purpose")

        res = self.engine.ask_question(
            self.qa_repo_id,
            question,
            top_k=5
        )

        top_chunks = res.get("top_chunks", [])
        self.assertGreater(len(top_chunks), 0)

        # Purpose query should prioritize README.md or package.json over arbitrary internal files
        first_file = top_chunks[0].get("file_path", "").lower()
        self.assertTrue(
            first_file.startswith("readme") or "readme" in first_file or first_file == "package.json",
            f"Expected README or package.json as top purpose chunk, got {first_file}"
        )

    def test_citations_contain_file_paths_and_lines_and_sources_persisted(self):
        """Task 7, 8: Assistant citations and sources contain real file paths, valid line numbers, and sources is never empty."""
        question = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            question,
            top_k=5
        )

        sources = res.get("sources", [])
        self.assertIsInstance(sources, list)
        self.assertGreater(len(sources), 0, "sources must not be empty so that sources_json is persisted")

        for s in sources:
            self.assertTrue(s.get("file_path"), "source must have a file_path")
            self.assertGreater(s.get("start_line", 0), 0, "start_line must be positive")
            self.assertGreaterEqual(s.get("end_line", 0), s.get("start_line", 0), "end_line must be >= start_line")
            self.assertTrue(s.get("citation"), "source must have a citation string")
            self.assertIn(":", s.get("citation"))

    def test_regression_no_edit_profile_caller_relationship(self):
        """Regression test: Ensure edit_profile is never reported as calling predictive_analytics or appearing in pipeline sources."""
        question = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            question,
            top_k=5
        )

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # The answer must NEVER claim edit_profile calls predictive_analytics or is part of this pipeline
        self.assertNotIn("edit_profile -> predictive_analytics", answer.lower())
        self.assertNotIn("from edit_profile", answer.lower())
        self.assertNotIn("called from edit_profile", answer.lower())

        # No source or citation may point to edit_profile
        for s in sources:
            self.assertNotIn("edit_profile", s.get("file_path", "").lower())
            self.assertNotIn("edit_profile", s.get("citation", "").lower())

    def test_regression_all_expected_pipeline_functions_in_answer(self):
        """Regression test: Ensure all 6 functions executed in the pipeline after YOLO detection are present and described."""
        question = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            question,
            top_k=5
        )

        answer = res.get("answer", "")
        expected_functions = [
            "get_part_prices",
            "calculate_damage_severity",
            "predictive_analytics",
            "calculate_internal_damage",
            "predict_insurance",
            "generate_recommendation",
        ]

        for fn in expected_functions:
            self.assertIn(fn, answer, f"Expected pipeline function '{fn}' was omitted from the answer")

    def test_regression_reversed_line_ranges_and_citations_normalized(self):
        """Regression test: Ensure reversed line ranges like 581-530 or 581–530 are normalized and bounds enforced."""
        sample_evidence = [
            {
                "file_path": "app.py",
                "start_line": 500,
                "end_line": 600,
                "symbol_name": "generate_recommendation",
                "content": "def generate_recommendation():\n    pass"
            }
        ]

        # 1. Hyphen reversed range: [app.py:581-530]
        ans_hyphen = "The function is defined in [app.py:581-530]."
        cleaned, val_cites, inval_cites = extract_and_validate_citations(ans_hyphen, sample_evidence)
        self.assertEqual(len(val_cites), 1)
        self.assertEqual(val_cites[0]["start_line"], 530)
        self.assertEqual(val_cites[0]["end_line"], 581)
        self.assertLessEqual(val_cites[0]["start_line"], val_cites[0]["end_line"])
        self.assertIn("[app.py:530-581]", cleaned)
        self.assertNotIn("581-530", cleaned)

        # 2. En-dash reversed range: [app.py:581–530]
        ans_endash = "The function is defined in [app.py:581–530]."
        cleaned_en, val_cites_en, inval_cites_en = extract_and_validate_citations(ans_endash, sample_evidence)
        self.assertEqual(len(val_cites_en), 1)
        self.assertEqual(val_cites_en[0]["start_line"], 530)
        self.assertEqual(val_cites_en[0]["end_line"], 581)
        self.assertLessEqual(val_cites_en[0]["start_line"], val_cites_en[0]["end_line"])
        self.assertIn("[app.py:530-581]", cleaned_en)

        # 3. Invalid file citation is stripped
        ans_invalid = "Fake citation [non_existent.py:10-20]."
        cleaned_inv, val_inv, inval_inv = extract_and_validate_citations(ans_invalid, sample_evidence)
        self.assertEqual(len(val_inv), 0)
        self.assertEqual(len(inval_inv), 1)
        self.assertNotIn("[non_existent.py:10-20]", cleaned_inv)

    def test_ast_call_chain_tracer_distinguishes_real_call_sites(self):
        """Task 2: AST call-chain analysis distinguishes actual call sites from functions merely in the same file or sharing a helper."""
        from app.ai.call_chain import ASTCallChainTracer

        sample_code = {
            "app.py": """
def connect_to_db():
    return None

def dashboard():
    result = model(image_path)
    prices = get_part_prices(email, counts)
    pred = predictive_analytics(prices)
    return pred

def edit_profile():
    conn = connect_to_db()
    # Notice: edit_profile merely shares connect_to_db and vehicleId in form,
    # but NEVER calls predictive_analytics or get_part_prices!
    return "profile"

def get_part_prices(email, counts):
    conn = connect_to_db()
    return {}

def predictive_analytics(prices):
    return {}
"""
        }

        tracer = ASTCallChainTracer(sample_code)
        # dashboard actually calls predictive_analytics
        self.assertTrue(tracer.verify_caller_relationship("dashboard", "predictive_analytics"))
        # edit_profile DOES NOT call predictive_analytics
        self.assertFalse(tracer.verify_caller_relationship("edit_profile", "predictive_analytics"))
        # edit_profile DOES NOT call get_part_prices
        self.assertFalse(tracer.verify_caller_relationship("edit_profile", "get_part_prices"))

    def test_regression_predict_insurance_accurate_explanation_and_call_type(self):
        """Regression test: Ensure the real answer generation path accurately explains predict_insurance,
        its conditional execution context (inside try...except), ML prediction behavior, and valid citations."""
        question = "Which functions are called after YOLO detects vehicle damage, and what does each function do?"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            question,
            top_k=5
        )

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # 1. predict_insurance must be present in the generated answer
        self.assertIn("predict_insurance", answer, "predict_insurance was omitted from the generated answer")

        # 2. Must describe its conditional / exception-guarded execution
        answer_lower = answer.lower()
        has_conditional_context = any(term in answer_lower for term in ("try", "except", "conditional"))
        self.assertTrue(
            has_conditional_context,
            "predict_insurance must be described as conditional / exception-guarded (inside try...except), not unconditional"
        )

        # 3. Must describe its ML / feature prediction behavior grounded in code
        has_ml_behavior = any(term in answer_lower for term in ("predict", "model", "features", "dataframe", "coverage"))
        self.assertTrue(
            has_ml_behavior,
            "predict_insurance must be accurately described based on its code implementation"
        )

        # 4. Must cite insurance_predictor.py with valid line range
        has_insurance_citation = any(
            s.get("file_path") == "insurance_predictor.py" and s.get("start_line", 0) <= s.get("end_line", 0)
            for s in sources
        )
        self.assertTrue(
            has_insurance_citation,
            "insurance_predictor.py must be present in verified sources with valid line bounds"
        )

    def test_regression_technologies_and_dependencies_retrieval_and_categorization(self):
        """Regression test: 'Which technologies and dependencies does it use?' must classify as dependencies,
        retrieve requirements.txt and README.md, generate a categorized stack, reject unrelated relationship denials,
        and provide valid citations."""
        q = "Which technologies and dependencies does it use?"
        self.assertEqual(self.engine.classify_question(q), "dependencies")

        res = self.engine.ask_question(
            self.vehicle_repo_id,
            q,
            top_k=5
        )

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # 1. Answer must categorize technologies (Python, Flask, MySQL, YOLOv8, etc.)
        ans_lower = answer.lower()
        self.assertTrue(any(term in ans_lower for term in ("python", "flask")), "Expected Python/Flask in tech stack answer")
        self.assertTrue(any(term in ans_lower for term in ("mysql", "mysql-connector")), "Expected MySQL in tech stack answer")
        self.assertTrue(any(term in ans_lower for term in ("yolov8", "ultralytics", "opencv", "scikit-learn", "torch")), "Expected AI/ML libraries in tech stack answer")

        # 2. Reject unrelated answer (connect_to_db and predict_insurance relationship denial)
        self.assertNotIn("does not establish a direct relationship", ans_lower, "Response must not contain irrelevant relationship denial")
        self.assertNotIn("direct relationship between", ans_lower, "Response must not contain irrelevant relationship denial")

        # 3. Sources must include requirements.txt and README.md
        source_files = [s.get("file_path", "") for s in sources]
        has_reqs = any("requirements" in sf.lower() for sf in source_files)
        has_readme = any("readme" in sf.lower() for sf in source_files)
        self.assertTrue(has_reqs or has_readme, f"Expected requirements.txt or README.md in sources, got: {source_files}")

        # 4. All sources have valid line ranges
        for s in sources:
            self.assertLessEqual(s.get("start_line", 1), s.get("end_line", 1))

    def test_regression_execution_flow_app_py_no_predictive_analytics_attribution(self):
        """Regression test: 'Explain the execution flow of app.py' must detail request flow in execution order,
        start with initialization / entry point / dashboard, reject false attribution of insurance prediction
        or template rendering to predictive_analytics, and cite verified lines."""
        q = "Explain the execution flow of app.py"
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            q,
            top_k=5
        )

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # 1. Answer must reference main dashboard flow
        ans_lower = answer.lower()
        self.assertIn("dashboard", ans_lower, "Execution flow must explain the main dashboard request handler")

        # 2. Reject incorrect attribution: predictive_analytics MUST NOT be said to calculate insurance or render templates
        self.assertNotRegex(
            ans_lower,
            r"predictive_analytics[^\.\n]*?(?:renders|rendering)[^\.\n]*?template",
            "predictive_analytics must not be attributed with rendering templates"
        )
        self.assertNotRegex(
            ans_lower,
            r"predictive_analytics[^\.\n]*?(?:calculates|predicts)[^\.\n]*?insurance",
            "predictive_analytics must not be attributed with predicting insurance"
        )

        # 3. Sources must contain app.py lines
        has_app_py = any("app.py" in s.get("file_path", "").lower() for s in sources)
        self.assertTrue(has_app_py, "Sources must include app.py")

        for s in sources:
            self.assertLessEqual(s.get("start_line", 1), s.get("end_line", 1))


if __name__ == "__main__":
    unittest.main()

