import os
import sys
import unittest

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.call_chain import ASTCallChainTracer, CallEdge
from app.ai.rag import RAGEngine, extract_and_validate_citations
from app.ai.llm import MockLLM
from app.ai.vector_store import get_vector_store
from app.database import SessionLocal
from app.models.file import RepositoryFile


class TestRAGCallChainEdgesRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.engine = RAGEngine(vector_store=cls.vector_store, llm_provider=cls.mock_llm)
        cls.vehicle_repo_id = "3f4d8af2-f8d0-4293-ae4e-885a4cbf3221"
        cls.fastapi_repo_id = "31806550-364e-4715-8904-55414d25691d"

    def test_valid_direct_call_edge(self):
        """Task 7.1: Verify a valid direct call edge from AST call site."""
        code = {
            "main.py": """
def func_b(x):
    return x * 2

def func_a():
    val = func_b(10)
    return val
"""
        }
        tracer = ASTCallChainTracer(code)
        pipe = tracer.trace_pipeline("func_a")
        edges = pipe.get("edges", [])
        direct_edges = [e for e in edges if e.caller_name == "func_a" and e.callee_name == "func_b"]
        self.assertEqual(len(direct_edges), 1)
        edge = direct_edges[0]
        self.assertEqual(edge.caller_file, "main.py")
        self.assertEqual(edge.callee_file, "main.py")
        self.assertEqual(edge.call_site_start, 6)
        self.assertEqual(edge.call_site_end, 6)
        self.assertEqual(edge.callee_def_start, 2)
        self.assertEqual(edge.callee_def_end, 3)
        self.assertEqual(edge.call_type, "direct")
        self.assertTrue(edge.is_resolved)
        self.assertEqual(edge.depth, 1)

    def test_valid_multi_step_call_chain(self):
        """Task 7.2: Verify a valid multi-step call chain (depth 1 caller -> callee, depth 2 callee -> helper)."""
        code = {
            "pipeline.py": """
def helper_format(data):
    return str(data)

def process_item(item):
    formatted = helper_format(item)
    return formatted

def handle_request():
    result = process_item({"id": 1})
    return result
"""
        }
        tracer = ASTCallChainTracer(code)
        pipe = tracer.trace_pipeline("handle_request")
        edges = pipe.get("edges", [])

        # Depth 1: handle_request -> process_item
        edge1 = next((e for e in edges if e.caller_name == "handle_request" and e.callee_name == "process_item"), None)
        self.assertIsNotNone(edge1)
        self.assertEqual(edge1.depth, 1)
        self.assertEqual(edge1.call_site_start, 10)
        self.assertEqual(edge1.callee_def_start, 5)

        # Depth 2: process_item -> helper_format
        edge2 = next((e for e in edges if e.caller_name == "process_item" and e.callee_name == "helper_format"), None)
        self.assertIsNotNone(edge2)
        self.assertEqual(edge2.depth, 2)
        self.assertEqual(edge2.call_site_start, 6)
        self.assertEqual(edge2.callee_def_start, 2)

        # Confirm handle_request does NOT directly call helper_format
        self.assertFalse(tracer.verify_caller_relationship("handle_request", "helper_format"))

    def test_function_exists_but_not_called_by_previous_function(self):
        """Task 7.3: Reject linear sequential chaining of uncalled functions. Definition range is not evidence of invocation."""
        code = {
            "app.py": """
def step_one():
    return "one"

def step_two():
    return "two"

def step_three():
    return "three"

def orchestrator():
    a = step_one()
    b = step_two()
    c = step_three()
    return a, b, c
"""
        }
        tracer = ASTCallChainTracer(code)
        # step_one does NOT call step_two
        self.assertFalse(tracer.verify_caller_relationship("step_one", "step_two"))
        # step_two does NOT call step_three
        self.assertFalse(tracer.verify_caller_relationship("step_two", "step_three"))
        # step_one does NOT call step_three
        self.assertFalse(tracer.verify_caller_relationship("step_one", "step_three"))

        pipe = tracer.trace_pipeline("orchestrator")
        edges = pipe.get("edges", [])
        for e in edges:
            self.assertNotEqual((e.caller_name, e.callee_name), ("step_one", "step_two"))
            self.assertNotEqual((e.caller_name, e.callee_name), ("step_two", "step_three"))

    def test_conditional_and_try_except_calls(self):
        """Task 7.4: Capture conditional and try-except context on call edges."""
        code = {
            "workflow.py": """
def fallback_handler():
    return "recovered"

def risky_action():
    return 42

def optional_action():
    return "flag_on"

def run_workflow(flag):
    try:
        val = risky_action()
    except Exception:
        fallback_handler()

    if flag:
        optional_action()
"""
        }
        tracer = ASTCallChainTracer(code)
        pipe = tracer.trace_pipeline("run_workflow")
        edges = pipe.get("edges", [])

        # risky_action inside try block
        edge_try = next((e for e in edges if e.callee_name == "risky_action"), None)
        self.assertIsNotNone(edge_try)
        self.assertEqual(edge_try.call_type, "try_except")
        self.assertIn("try", edge_try.condition_desc.lower())
        self.assertEqual(edge_try.call_site_start, 13)

        # optional_action inside if flag block
        edge_if = next((e for e in edges if e.callee_name == "optional_action"), None)
        self.assertIsNotNone(edge_if)
        self.assertEqual(edge_if.call_type, "conditional")
        self.assertIn("if", edge_if.condition_desc.lower())
        self.assertEqual(edge_if.call_site_start, 18)

    def test_cross_file_calls_and_unresolved_dynamic_calls(self):
        """Task 7.5: Resolve cross-file calls via imports and handle unresolved dynamic calls conservatively."""
        code = {
            "predictor.py": """
def predict_score(data):
    return len(data)
""",
            "main.py": """
from predictor import predict_score

def execute(model_instance, raw_data):
    # 1. Unresolved dynamic call to object method
    features = model_instance(raw_data)
    # 2. Resolved cross-file call via import
    score = predict_score(features)
    return score
"""
        }
        tracer = ASTCallChainTracer(code)
        pipe = tracer.trace_pipeline("execute")
        edges = pipe.get("edges", [])

        # Cross-file resolved call
        resolved_edge = next((e for e in edges if e.callee_name == "predict_score"), None)
        self.assertIsNotNone(resolved_edge)
        self.assertTrue(resolved_edge.is_resolved)
        self.assertEqual(resolved_edge.callee_file, "predictor.py")
        self.assertEqual(resolved_edge.callee_def_start, 2)
        self.assertEqual(resolved_edge.callee_def_end, 3)

        # Unresolved dynamic call
        unresolved_edge = next((e for e in edges if not e.is_resolved), None)
        self.assertIsNotNone(unresolved_edge)
        self.assertEqual(unresolved_edge.called_name, "model_instance")
        self.assertEqual(unresolved_edge.call_site_start, 6)
        self.assertFalse(unresolved_edge.is_resolved)
        self.assertIn("not defined", unresolved_edge.unresolved_reason.lower())

    def test_vehicle_damage_repository_actual_call_graph(self):
        """Task 7.6: Verify the vehicle-damage repository's actual call graph from AST evidence."""
        db = SessionLocal()
        files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == self.vehicle_repo_id).all()
        file_map = {f.file_path: f.content for f in files if f.content}
        db.close()

        tracer = ASTCallChainTracer(file_map)
        pipe = tracer.trace_pipeline("Explain the actual call chain after YOLO detects vehicle damage")

        anchor = pipe.get("anchor")
        self.assertEqual(anchor["name"], "dashboard")
        self.assertEqual(anchor["file_path"], "app.py")
        self.assertEqual(pipe.get("event_line"), 182)

        direct = {d["name"]: d for d in pipe.get("direct_calls", [])}
        self.assertIn("get_part_prices", direct)
        self.assertEqual(direct["get_part_prices"]["call_line"], 201)
        self.assertEqual(direct["get_part_prices"]["start_line"], 296)
        self.assertEqual(direct["get_part_prices"]["end_line"], 333)

        self.assertIn("calculate_damage_severity", direct)
        self.assertEqual(direct["calculate_damage_severity"]["call_line"], 212)
        self.assertEqual(direct["calculate_damage_severity"]["start_line"], 336)
        self.assertEqual(direct["calculate_damage_severity"]["end_line"], 380)

        self.assertIn("predictive_analytics", direct)
        self.assertEqual(direct["predictive_analytics"]["call_line"], 214)
        self.assertEqual(direct["predictive_analytics"]["call_site_end"], 216)
        self.assertEqual(direct["predictive_analytics"]["start_line"], 383)
        self.assertEqual(direct["predictive_analytics"]["end_line"], 419)

        self.assertIn("calculate_internal_damage", direct)
        self.assertEqual(direct["calculate_internal_damage"]["call_line"], 218)
        self.assertEqual(direct["calculate_internal_damage"]["start_line"], 422)
        self.assertEqual(direct["calculate_internal_damage"]["end_line"], 460)

        self.assertIn("predict_insurance", direct)
        self.assertEqual(direct["predict_insurance"]["call_line"], 241)
        self.assertEqual(direct["predict_insurance"]["file_path"], "insurance_predictor.py")
        self.assertEqual(direct["predict_insurance"]["start_line"], 7)
        self.assertEqual(direct["predict_insurance"]["end_line"], 17)
        self.assertEqual(direct["predict_insurance"]["call_type"], "try_except")

        self.assertIn("generate_recommendation", direct)
        self.assertEqual(direct["generate_recommendation"]["call_line"], 254)
        self.assertEqual(direct["generate_recommendation"]["call_site_end"], 258)
        self.assertEqual(direct["generate_recommendation"]["start_line"], 545)
        self.assertEqual(direct["generate_recommendation"]["end_line"], 566)

        # Downstream helper
        indirect = {ind["name"]: ind for ind in pipe.get("indirect_calls", [])}
        self.assertIn("get_part_name_from_id", indirect)
        self.assertEqual(indirect["get_part_name_from_id"]["caller"], "get_part_prices")
        self.assertEqual(indirect["get_part_name_from_id"]["call_line"], 318)
        self.assertEqual(indirect["get_part_name_from_id"]["start_line"], 465)
        self.assertEqual(indirect["get_part_name_from_id"]["end_line"], 469)

        # Verify callees do NOT call each other
        self.assertFalse(tracer.verify_caller_relationship("get_part_prices", "calculate_damage_severity"))
        self.assertFalse(tracer.verify_caller_relationship("calculate_damage_severity", "predictive_analytics"))
        self.assertFalse(tracer.verify_caller_relationship("predictive_analytics", "calculate_internal_damage"))
        self.assertFalse(tracer.verify_caller_relationship("calculate_internal_damage", "predict_insurance"))
        self.assertFalse(tracer.verify_caller_relationship("predict_insurance", "generate_recommendation"))

        # Unresolved dynamic call
        unresolved = pipe.get("unresolved_calls", [])
        model_call = next((u for u in unresolved if u["called_name"] == "model"), None)
        self.assertIsNotNone(model_call)
        self.assertEqual(model_call["call_site_start"], 182)
        self.assertFalse(model_call["is_resolved"])

    def test_end_to_end_answer_distinguishes_call_sites_from_definitions(self):
        """Task 7.7: End-to-end question answering distinguishes call sites from definitions and rejects contradictory denials."""
        res = self.engine.ask_question(
            self.vehicle_repo_id,
            "Explain the actual call chain after YOLO detects vehicle damage",
            top_k=5
        )
        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Call-site citations
        self.assertIn("[app.py:201]", answer)
        self.assertIn("[app.py:212]", answer)
        self.assertIn("[app.py:218]", answer)
        self.assertIn("[app.py:241]", answer)
        self.assertIn("[app.py:318]", answer)

        # Callee definition citations
        self.assertIn("[app.py:296-333]", answer)
        self.assertIn("[app.py:336-380]", answer)
        self.assertIn("[app.py:383-419]", answer)
        self.assertIn("[app.py:422-460]", answer)
        self.assertIn("[insurance_predictor.py:7-17]", answer)
        self.assertIn("[app.py:545-566]", answer)
        self.assertIn("[app.py:465-469]", answer)

        # No contradictory disclaimers
        self.assertNotIn("does not establish a direct relationship", answer.lower())
        self.assertNotIn("does not establish direct relationships", answer.lower())

        # No false callee-to-callee chains
        self.assertNotIn("get_part_prices -> calculate_damage_severity", answer.lower())
        self.assertNotIn("get_part_prices calls calculate_damage_severity", answer.lower())
        self.assertNotIn("calculate_damage_severity calls predictive_analytics", answer.lower())

    def test_fastapi_and_cross_repo_isolation_regression(self):
        """Task 7.8: Existing FastAPI execution-flow and cross-repo isolation remain intact."""
        q_fastapi = "Explain the execution flow of fastapi/applications.py and cite the relevant source lines"
        res = self.engine.ask_question(self.fastapi_repo_id, q_fastapi, top_k=5)
        answer = res.get("answer", "")
        self.assertNotIn("app_initialization", answer)
        self.assertNotIn("init_routes", answer)
        self.assertIn("FastAPI", answer)
        self.assertNotIn("vehicle", answer.lower())
        self.assertNotIn("damage", answer.lower())
