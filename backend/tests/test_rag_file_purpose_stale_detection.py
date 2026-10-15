import os
import sys
import unittest

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.llm import MockLLM
from app.ai.rag import (
    RAGEngine,
    extract_and_validate_citations,
    finalize_rag_response
)
from app.ai.vector_store import get_vector_store


class TestRAGFilePurposeAndStaleDetection(unittest.TestCase):
    """
    Regression test suite verifying:
    1. MockLLM does not mistake method docstring for class purpose when a class has no class docstring.
    2. MockLLM generates function-level citations with exact source line offsets rather than whole-chunk line ranges.
    3. extract_and_validate_citations rejects stale/outdated line ranges and finalize_rag_response acknowledges when accurate citations cannot be established.
    4. RAGEngine accurately responds to file-purpose queries without confusing docstrings or producing stale citations.
    """

    def setUp(self):
        self.mock_llm = MockLLM()
        self.vector_store = get_vector_store()
        self.engine = RAGEngine(vector_store=self.vector_store, llm_provider=self.mock_llm)
        self.repolens_repo_id = "6ed6107b-7311-499d-960a-1a21edb7398a"

    def test_class_without_docstring_does_not_confuse_method_docstring(self):
        """
        1. Use a class with no class docstring but with a method docstring.
        Verify the generated opening sentence does not confuse the two.
        """
        user_prompt = """Question:
What is the purpose of backend/app/services/sample_service.py? Explain its main responsibilities in 3–5 bullet points with accurate citations.

Repository evidence:
--- CHUNK 1 | File: backend/app/services/sample_service.py | Lines: 1-60 | Symbol: WorkerManager ---
class WorkerManager:

    def __init__(self, queue):
        self.queue = queue

    def process_incoming_job(self, job_id: str):
        \"\"\"Drain the job queue and dispatch payload to downstream executors.\"\"\"
        pass

    def get_worker_status(self) -> str:
        \"\"\"Return the health status of active workers.\"\"\"
        return "healthy"

Instructions:
1. Explain the purpose and main responsibilities of `backend/app/services/sample_service.py` based strictly on the retrieved code evidence.
2. Structure the answer with:
   - An opening sentence stating the overall purpose of the file and its primary class or role.
   - 3–5 concise, substantive bullet points explaining its main responsibilities.
"""
        response = self.mock_llm.generate_response("system", user_prompt)
        opening_sentence = response.strip().split("\n")[0]

        # The opening sentence MUST NOT claim the class purpose is "Drain the job queue and dispatch payload..."
        self.assertNotIn("Drain the job queue", opening_sentence)
        self.assertNotIn("dispatch payload to downstream executors", opening_sentence)

        # The opening sentence should state the primary class or core role cleanly
        self.assertIn("WorkerManager", opening_sentence)
        self.assertIn("backend/app/services/sample_service.py", opening_sentence)

    def test_function_citations_use_exact_offsets_not_whole_chunk(self):
        """
        Verify citations point to actual function definitions in the current source,
        calculating exact line offsets rather than repeating the chunk's start-end span.
        """
        user_prompt = """Question:
What is the purpose of backend/app/utils/helpers.py? Explain its main responsibilities in 3–5 bullet points with accurate citations.

Repository evidence:
--- CHUNK 1 | File: backend/app/utils/helpers.py | Lines: 100-200 | Symbol: helpers ---
# Utility functions for data transformation
import os

def normalize_text(text: str) -> str:
    \"\"\"Normalize whitespace and lowercase input string.\"\"\"
    return text.strip().lower()

def compute_hash(data: bytes) -> str:
    \"\"\"Compute SHA-256 digest of input bytes.\"\"\"
    import hashlib
    return hashlib.sha256(data).hexdigest()

Instructions:
1. Explain the purpose and main responsibilities of `backend/app/utils/helpers.py` based strictly on the retrieved code evidence.
2. Structure the answer with:
   - An opening sentence stating the overall purpose of the file and its primary class or role.
   - 3–5 concise, substantive bullet points explaining its main responsibilities.
3. Every bullet point and technical claim must include an inline citation with the exact supporting file and line range.
"""
        response = self.mock_llm.generate_response("system", user_prompt)

        # The chunk range is 100-200.
        # normalize_text starts at line offset ~4 (line 104) and compute_hash at offset ~8 (line 108).
        # Neither function should be cited as [backend/app/utils/helpers.py:100-200].
        self.assertNotIn("[backend/app/utils/helpers.py:100-200]", response)

        # Citations should be narrower offsets starting within 100-115
        lines = response.splitlines()
        bullet_lines = [l for l in lines if l.strip().startswith("-")]
        self.assertTrue(len(bullet_lines) >= 2)
        for bl in bullet_lines:
            self.assertIn("[backend/app/utils/helpers.py:", bl)
            self.assertNotIn(":100-200]", bl)

    def test_stale_line_numbers_rejected_or_acknowledged(self):
        """
        Reproduce stale line numbers and ensure they are rejected or acknowledged.
        """
        # Test 1: Citation line range exceeds live file length on disk
        # backend/reindex_repo.py exists on disk and has around 80 lines.
        # A stale citation of [backend/reindex_repo.py:9000-9050] must be rejected as invalid.
        raw_answer = "This script maintains collections [backend/reindex_repo.py:9000-9050]."
        evidence_chunks = [{
            "file_path": "backend/reindex_repo.py",
            "start_line": 9000,
            "end_line": 9050,
            "content": "def stale_function(): pass"
        }]

        cleaned, valid, invalid = extract_and_validate_citations(raw_answer, evidence_chunks)
        self.assertEqual(len(valid), 0, "Stale citation beyond live file length must not be valid")
        self.assertIn("[backend/reindex_repo.py:9000-9050]", invalid)
        self.assertNotIn("[backend/reindex_repo.py:9000-9050]", cleaned)

        # Test 2: When all citations are rejected, finalize_rag_response sets grounded=False
        # and acknowledges that accurate line citations cannot be established
        final_answer, sources, grounded = finalize_rag_response(
            raw_answer=raw_answer,
            llm_context_chunks=evidence_chunks,
            is_call_chain=False,
            question="What does backend/reindex_repo.py do?"
        )
        self.assertFalse(grounded)
        self.assertEqual(len(sources), 0)
        self.assertIn("accurate line citations could not be established", final_answer.lower())

    def test_repolens_rag_engine_file_purpose_end_to_end(self):
        """
        Verify the live end-to-end question for backend/app/ai/rag.py:
        - Opening sentence states overall purpose and RAGEngine class (not classify_question)
        - Excludes nested closures such as sort_key, find_evidence_candidates, etc.
        - Prioritizes core RAG pipeline methods (ask_question, build_retrieval_queries, classify_question,
          extract_and_validate_citations, finalize_rag_response).
        - Bullet points cite exact function ranges supporting each responsibility.
        - Result is grounded with valid sources.
        """
        q = "What is the purpose of backend/app/ai/rag.py? Explain its main responsibilities in 3–5 bullet points with accurate citations."
        res = self.engine.ask_question(self.repolens_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        grounded = res.get("grounded", False)
        sources = res.get("sources", [])

        # Grounded flag must be True
        self.assertTrue(grounded, f"Expected grounded=True, got False. Answer:\n{answer}")
        self.assertTrue(len(sources) >= 5, f"Expected at least 5 sources, got {len(sources)}")

        # Opening sentence
        opening_sentence = answer.strip().split("\n")[0]
        self.assertIn("backend/app/ai/rag.py", opening_sentence)
        self.assertIn("RAGEngine", opening_sentence)
        self.assertNotIn("classify_question", opening_sentence)
        self.assertNotIn("classify a user question", opening_sentence.lower())

        # Responsibilities bullet points
        bullet_lines = [l for l in answer.splitlines() if l.strip().startswith("-")]
        self.assertTrue(3 <= len(bullet_lines) <= 5, f"Expected 3-5 bullet points, got {len(bullet_lines)}")

        # Exclude nested closures
        answer_lower = answer.lower()
        self.assertNotIn("sort_key", answer_lower, "sort_key closure should not be listed as a main responsibility")
        self.assertNotIn("find_evidence_candidates", answer_lower, "find_evidence_candidates closure should not be listed")
        self.assertNotIn("missing_sym_generator", answer_lower, "missing_sym_generator closure should not be listed")
        self.assertNotIn("response_generator", answer_lower, "response_generator closure should not be listed")

        # Must prioritize core RAG pipeline responsibilities
        self.assertTrue(any("RAG Pipeline Orchestration" in b or "ask_question" in b for b in bullet_lines),
                        "Expected ask_question / RAG Pipeline Orchestration in responsibilities")
        self.assertTrue(any("Retrieval Query Construction" in b or "build_retrieval_queries" in b for b in bullet_lines),
                        "Expected build_retrieval_queries in responsibilities")
        self.assertTrue(any("Query Classification" in b or "classify_question" in b for b in bullet_lines),
                        "Expected classify_question in responsibilities")
        self.assertTrue(any("Citation Extraction & Validation" in b or "extract_and_validate_citations" in b for b in bullet_lines),
                        "Expected extract_and_validate_citations in responsibilities")
        self.assertTrue(any("Response Finalization" in b or "finalize_rag_response" in b for b in bullet_lines),
                        "Expected finalize_rag_response in responsibilities")

        # Citations point to backend/app/ai/rag.py with exact method definition ranges
        for b in bullet_lines:
            self.assertIn("[backend/app/ai/rag.py:", b)

        # Check source citation boundaries
        source_citations = [s["citation"] for s in sources]
        self.assertTrue(any(":231" in sc for sc in source_citations), f"Expected ask_question range starting around 2311-2320, got {source_citations}")
        self.assertTrue(any(":195" in sc for sc in source_citations), f"Expected build_retrieval_queries range starting around 1948-1955, got {source_citations}")
        self.assertTrue(any(":161" in sc for sc in source_citations), f"Expected classify_question range starting around 1608-1615, got {source_citations}")
        self.assertTrue(any(":42" in sc or ":43" in sc for sc in source_citations), f"Expected extract_and_validate_citations range starting around 428-430, got {source_citations}")
        self.assertTrue(any(":88" in sc for sc in source_citations), f"Expected finalize_rag_response range starting around 880-885, got {source_citations}")

    def test_nested_closures_excluded_and_core_methods_prioritized(self):
        """
        Verify that MockLLM responsibility selection directly filters out nested closures
        (such as sort_key, helper closures with indent >= 8) and prioritizes core pipeline methods.
        """
        chunk_code = '''
class SamplePipeline:
    """Core pipeline manager for repository queries."""

    def __init__(self, config):
        self.config = config

    def build_retrieval_queries(self, query: str):
        """Construct targeted queries for vector retrieval."""
        def sort_key(item):
            return item.score
        items = []
        return sorted(items, key=sort_key)

    def ask_question(self, query: str):
        """Orchestrate pipeline retrieval and LLM inference."""
        def response_generator():
            yield "result"
        return response_generator()
'''
        user_prompt = f"""Question:
What is the purpose of backend/app/ai/pipeline.py? Explain its main responsibilities in 3–5 bullet points with accurate citations.

Repository evidence:
--- CHUNK 1 | File: backend/app/ai/pipeline.py | Lines: 100-200 | Symbol: SamplePipeline ---
{chunk_code}

Instructions:
1. Explain the purpose and main responsibilities of `backend/app/ai/pipeline.py` based strictly on the retrieved code evidence.
2. Structure the answer with:
   - An opening sentence stating the overall purpose of the file and its primary class or role.
   - 3–5 concise, substantive bullet points explaining its main responsibilities.
3. Every bullet point and technical claim must include an inline citation with the exact supporting file and line range.
"""
        response = self.mock_llm.generate_response("system", user_prompt)

        # Nested closures sort_key and response_generator must NOT appear as responsibilities
        self.assertNotIn("sort_key", response.lower())
        self.assertNotIn("response_generator", response.lower())
        self.assertNotIn("Sort Key", response)
        self.assertNotIn("Response Generator", response)

        # ask_question and build_retrieval_queries MUST be present as top responsibilities
        self.assertIn("RAG Pipeline Orchestration", response)
        self.assertIn("Retrieval Query Construction", response)

        # Lines must be cited within the chunk
        self.assertIn("[backend/app/ai/pipeline.py:", response)

    def test_os_module_imported_and_target_resolution_uses_sqlite_not_stale_chroma(self):
        """
        Regression test:
        1. Ensures app.ai.rag imports os so candidate resolution does not abort with NameError.
        2. Ensures when Chroma has stale chunks and SQLite has drifted from live file,
           SQLite candidate resolution succeeds, triggers targeted refresh,
           and does NOT fall back to stale Chroma chunks.
        """
        import app.ai.rag as rag_module
        self.assertTrue(hasattr(rag_module, "os"), "app.ai.rag must have 'os' in module namespace")
        self.assertIsNotNone(getattr(rag_module, "os", None))

        # Test candidate resolution and stale Chroma fallback prevention
        # We simulate a vector store with stale chunks for backend/app/ai/rag.py
        # and verify ask_question uses SQLite/live resolution rather than stale Chroma chunks.
        stale_chunks = [
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 1502,
                "end_line": 1507,
                "symbol_name": "RAGEngine.ask_question",
                "content": "def ask_question(self, repo_id, question): pass",
                "chunk_id": "backend/app/ai/rag.py:1502-1507",
                "defined_symbols": "ask_question",
                "called_symbols": "",
                "imported_symbols": "",
            },
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 1026,
                "end_line": 1057,
                "symbol_name": "RAGEngine.classify_question",
                "content": "def classify_question(self, question): pass",
                "chunk_id": "backend/app/ai/rag.py:1026-1057",
                "defined_symbols": "classify_question",
                "called_symbols": "",
                "imported_symbols": "",
            }
        ]

        class StaleMockVectorStore:
            def __init__(self, stale_data):
                self.stale_data = stale_data
                self.upserted = []

            def get_or_create_collection(self, repo_id):
                class MockCollection:
                    def __init__(self, parent):
                        self.parent = parent
                    def get(self, include=None):
                        docs = [c["content"] for c in self.parent.stale_data]
                        metas = [{k: v for k, v in c.items() if k != "content"} for c in self.parent.stale_data]
                        return {"documents": docs, "metadatas": metas}
                    def delete(self, where=None):
                        pass
                return MockCollection(self)

            def upsert_chunks(self, repo_id, chunks):
                self.upserted.extend(chunks)

            def query_similar_chunks(self, repository_id, query_text, top_k=20):
                return []

        mock_vs = StaleMockVectorStore(stale_chunks)
        test_engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)

        # Ask question for backend/app/ai/rag.py
        q = "What is the purpose of backend/app/ai/rag.py? Explain its main responsibilities in 3–5 bullet points with accurate citations."
        res = test_engine.ask_question(self.repolens_repo_id, q, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Stale line numbers from Chroma (1502-1507, 1026-1057) must NOT be present in sources or answer
        self.assertNotIn(":1502-1507", answer)
        self.assertNotIn(":1026-1057", answer)
        for s in sources:
            self.assertNotEqual(s.get("citation"), "[backend/app/ai/rag.py:1502-1507]")
            self.assertNotEqual(s.get("citation"), "[backend/app/ai/rag.py:1026-1057]")

        # Current live ranges (e.g. ask_question starting at 2311+, classify_question starting at 1608+)
        # should be used because SQLite / live file resolution succeeded
        source_citations = [s.get("citation", "") for s in sources]
        self.assertTrue(
            any(":231" in sc for sc in source_citations),
            f"Expected live ask_question range starting around 2311-2320, got citations: {source_citations}"
        )


if __name__ == "__main__":
    unittest.main()
