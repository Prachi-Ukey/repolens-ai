import os
import sys
import unittest
from typing import List, Dict, Any
import logging

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import (
    RAGEngine,
    extract_target_references,
    extract_and_validate_citations,
    finalize_rag_response,
)
from app.ai.llm import MockLLM
from app.ai.vector_store import get_vector_store


class DummyVectorStore:
    """Mock vector store with controlled collections for testing repository isolation & ranking."""
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, repo_id: str):
        if repo_id not in self.collections:
            self.collections[repo_id] = DummyCollection(repo_id)
        return self.collections[repo_id]

    def query_similar_chunks(self, repository_id: str, query_text: str, top_k: int = 20):
        col = self.get_or_create_collection(repository_id)
        return col.query_similar(query_text, top_k)


class DummyCollection:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.chunks = []

    def count(self):
        return len(self.chunks)

    def add_chunk(self, file_path: str, start_line: int, end_line: int, content: str, symbol_name: str = "", defined_symbols: str = ""):
        chunk = {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name,
            "content": content,
            "chunk_id": f"{file_path}:{start_line}-{end_line}",
            "defined_symbols": defined_symbols,
            "called_symbols": "",
            "imported_symbols": "",
            "distance": 0.5,
        }
        self.chunks.append(chunk)

    def get(self, include=None):
        docs = [c["content"] for c in self.chunks]
        metas = [{
            "file_path": c["file_path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "symbol_name": c["symbol_name"],
            "chunk_id": c["chunk_id"],
            "defined_symbols": c["defined_symbols"],
            "called_symbols": c["called_symbols"],
            "imported_symbols": c["imported_symbols"],
        } for c in self.chunks]
        return {"documents": docs, "metadatas": metas}

    def query_similar(self, query_text: str, top_k: int = 20):
        # Return chunks with artificial distances
        results = []
        for c in self.chunks:
            # If query matches README terms, make README appear closer to test that target file overrides it
            is_readme = "readme" in c["file_path"].lower()
            dist = 0.15 if is_readme else 0.45
            copy_c = dict(c)
            copy_c["distance"] = dist
            results.append(copy_c)
        return results[:top_k]


from tests.rag_fixtures import ensure_rag_fixtures, deactivate_rag_isolation


class TestRAGSourceRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ensure_rag_fixtures()
        cls.real_vector_store = get_vector_store()
        cls.mock_llm = MockLLM()
        cls.real_engine = RAGEngine(vector_store=cls.real_vector_store, llm_provider=cls.mock_llm)
        cls.fastapi_repo_id = fixtures["fastapi_repo_id"]
        cls.vehicle_repo_id = fixtures["vehicle_repo_id"]

    @classmethod
    def tearDownClass(cls):
        deactivate_rag_isolation()

    # =========================================================================
    # Test 1: Exact file and method question (fastapi/applications.py and FastAPI.__call__)
    # =========================================================================
    def test_01_exact_file_and_method_question(self):
        """
        Case 1: User asks for FastAPI.__call__ in fastapi/applications.py.
        Must retrieve fastapi/applications.py implementation, NOT README.md.
        """
        question = (
            "In `fastapi/applications.py`, find the `FastAPI` class and explain the purpose of its "
            "`__call__` method. Use evidence from that file and include relevant source lines. "
            "If the file or method is not available in the retrieved evidence, say so explicitly."
        )

        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", False)
        top_chunks = res.get("top_chunks", [])

        # 1. Answer must be grounded and cite the requested file
        self.assertTrue(grounded, "Expected response to be grounded")
        self.assertGreater(len(sources), 0, "Expected at least one supporting source")

        source_files = [s["file_path"] for s in sources]
        self.assertTrue(
            any("applications.py" in sf for sf in source_files),
            f"Expected fastapi/applications.py in sources, got: {source_files}"
        )
        self.assertFalse(
            any("readme" in sf.lower() for sf in source_files),
            f"README should not be in sources for exact file query: {source_files}"
        )

        # 2. Answer must discuss FastAPI and __call__
        self.assertIn("FastAPI", answer)
        self.assertIn("__call__", answer)

        # 3. Top retrieved chunks must contain applications.py with __call__
        top_files = [c.get("file_path", "") for c in top_chunks]
        self.assertTrue(any("applications.py" in tf for tf in top_files))
        chunk_contents = " ".join(c.get("content", "") for c in top_chunks)
        self.assertIn("__call__", chunk_contents)

    # =========================================================================
    # Test 2: Class or method implementation spanning multiple chunks
    # =========================================================================
    def test_02_class_or_method_spanning_multiple_chunks(self):
        """
        Case 2: FastAPI.__call__ method implementation spans lines 1184-1227 across chunks.
        Verify contiguous chunk expansion captures the full method body.
        """
        question = "In `fastapi/applications.py`, what is the implementation of `__call__`?"
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        top_chunks = res.get("top_chunks", [])
        self.assertGreater(len(top_chunks), 0)

        # Find the chunk containing __call__
        call_chunk = next((c for c in top_chunks if "applications.py" in c.get("file_path", "") and "__call__" in c.get("content", "")), None)
        self.assertIsNotNone(call_chunk, "Expected a chunk containing __call__")

        # Verify chunk spans across the entrypoint and middleware delegation (lines 1184 to 1220+)
        content = call_chunk.get("content", "")
        self.assertIn("async def __call__", content)
        # Verify middleware_stack or scope delegation is included due to adjacent chunk merging
        self.assertTrue(
            "middleware_stack" in content or "scope" in content,
            "Expected __call__ method body including middleware delegation in merged chunk"
        )

    # =========================================================================
    # Test 3: File exists in repository but is not initially nearest semantic result
    # =========================================================================
    def test_03_file_exists_but_lower_initial_semantic_rank(self):
        """
        Case 3: Semantic search initially ranks README closer (dist=0.15 vs code dist=0.45).
        Targeted file resolution must boost the requested implementation file above README.
        """
        dummy_vs = DummyVectorStore()
        col = dummy_vs.get_or_create_collection("test_repo_3")
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=50,
            content="RepoLens documentation and overview of components."
        )
        col.add_chunk(
            file_path="src/worker.py",
            start_line=1,
            end_line=30,
            content="class Worker:\n    def execute_job(self):\n        return True",
            symbol_name="Worker",
            defined_symbols="Worker,execute_job"
        )

        engine = RAGEngine(vector_store=dummy_vs, llm_provider=self.mock_llm)
        question = "In `src/worker.py`, find the `Worker` class and explain its `execute_job` method."

        res = engine.ask_question("test_repo_3", question, top_k=5)
        top_chunks = res.get("top_chunks", [])

        self.assertGreater(len(top_chunks), 0)
        first_chunk = top_chunks[0]
        self.assertEqual(first_chunk.get("file_path"), "src/worker.py")
        self.assertIn("Worker", first_chunk.get("content", ""))

    # =========================================================================
    # Test 4: File that is genuinely missing from the index
    # =========================================================================
    def test_04_file_genuinely_missing_from_index(self):
        """
        Case 4: User asks for a file that is not in the indexed repository.
        Must explicitly report that the file was not found in the indexed repository.
        Must distinguish between 'not found in index' and 'not found in retrieved evidence'.
        """
        question = "In `fastapi/nonexistent_file.py`, find the `MissingClass` and explain its logic."
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        grounded = res.get("grounded", False)
        sources = res.get("sources", [])

        # Must report not found in the indexed repository
        self.assertIn("not found in the indexed repository", answer.lower())
        self.assertIn("fastapi/nonexistent_file.py", answer)
        self.assertFalse(grounded)
        self.assertEqual(len(sources), 0)

    # =========================================================================
    # Test 5: General repository overview question
    # =========================================================================
    def test_05_general_repository_overview_question(self):
        """
        Case 5: Overview / purpose question must retrieve README.md and answer with overview.
        """
        question = "What is the main purpose of this repository, and what problem does it solve?"
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", False)

        self.assertTrue(grounded)
        self.assertGreater(len(sources), 0)
        source_files = [s["file_path"].lower() for s in sources]
        self.assertTrue(any("readme" in sf for sf in source_files))
        self.assertTrue("fastapi" in answer.lower())

    # =========================================================================
    # Test 6: Question that should retrieve implementation code rather than README
    # =========================================================================
    def test_06_implementation_question_over_readme(self):
        """
        Case 6: Implementation question on vehicle repo must retrieve app.py code, not README.
        """
        question = "In `app.py`, how is `calculate_damage_severity` implemented?"
        res = self.real_engine.ask_question(self.vehicle_repo_id, question, top_k=5)

        sources = res.get("sources", [])
        top_chunks = res.get("top_chunks", [])

        self.assertGreater(len(top_chunks), 0)
        self.assertEqual(top_chunks[0].get("file_path"), "app.py")
        self.assertFalse(any("readme" in (c.get("file_path") or "").lower() for c in top_chunks[:2]))

        source_files = [s["file_path"].lower() for s in sources]
        self.assertFalse(any("readme" in sf for sf in source_files))

    # =========================================================================
    # Test 7: Citation ranges outside the supplied evidence
    # =========================================================================
    def test_07_citation_ranges_outside_supplied_evidence(self):
        """
        Case 7: Answer cites lines outside the supplied evidence range.
        Must mark citation invalid, remove it, and set grounded=False.
        """
        evidence_chunks = [{
            "file_path": "app.py",
            "start_line": 100,
            "end_line": 200,
            "content": "def handler():\n    return True",
            "code": "def handler():\n    return True"
        }]

        # LLM output cites unsupported range lines 800-900
        raw_answer = "The handler is defined at [app.py:800-900]."
        final_answer, valid_cites, invalid_cites = extract_and_validate_citations(
            raw_answer,
            evidence_chunks
        )

        self.assertEqual(len(invalid_cites), 1)
        self.assertEqual(len(valid_cites), 0)
        self.assertIn("[app.py:800-900]", invalid_cites[0])

        final_ans, sources, grounded = finalize_rag_response(
            raw_answer,
            evidence_chunks,
            is_call_chain=False,
            question="Where is handler defined?"
        )

        self.assertFalse(grounded, "Unsupported line range must result in grounded=False")
        self.assertEqual(len(sources), 0)

    # =========================================================================
    # Test 8: Question with no relevant evidence
    # =========================================================================
    def test_08_question_with_no_relevant_evidence(self):
        """
        Case 8: Question for non-existent concept returns fallback and grounded=False.
        """
        dummy_vs = DummyVectorStore()
        # Empty collection
        engine = RAGEngine(vector_store=dummy_vs, llm_provider=self.mock_llm)
        question = "How does the quantum_superconducting_flux_qubit work in this codebase?"

        res = engine.ask_question("empty_repo", question, top_k=5)
        self.assertFalse(res.get("grounded"))
        self.assertEqual(len(res.get("sources")), 0)
        self.assertIn("couldn't find enough evidence", res.get("answer").lower())

    # =========================================================================
    # Test 9: Repository isolation between different indexed repositories
    # =========================================================================
    def test_09_repository_isolation(self):
        """
        Case 9: Querying the FastAPI repository must never retrieve chunks from the Vehicle Damage repository.
        """
        question = "What is the primary web application entry point?"
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        sources = res.get("sources", [])
        top_chunks = res.get("top_chunks", [])

        # Ensure no vehicle damage files leaked
        for s in sources:
            fp = s["file_path"].lower()
            self.assertFalse("insurance_predictor" in fp)
            self.assertFalse(fp == "app.py")

        for c in top_chunks:
            fp = c.get("file_path", "").lower()
            self.assertFalse("insurance_predictor" in fp)
            self.assertFalse(fp == "app.py")

    # =========================================================================
    # Test 10: Synthetic test repository with __call__, caller, and README
    # =========================================================================
    def test_10_synthetic_repo_with_call_method_and_caller(self):
        """
        Section 7 Verification:
        Small synthetic test repository containing:
        - A known Python file (server.py)
        - A class (AppDispatcher) with a `__call__` method
        - Another function (handle_request) that calls that method
        - A README containing generic project information
        Confirm exact file/method question retrieves implementation rather than README.
        """
        dummy_vs = DummyVectorStore()
        col = dummy_vs.get_or_create_collection("synthetic_test_repo")

        # 1. README with generic information
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=80,
            content="SyntheticDispatcher is a high-performance HTTP service framework for Python."
        )

        # 2. Python file with class and __call__ method
        server_code = (
            "class AppDispatcher:\n"
            "    def __init__(self, app):\n"
            "        self.app = app\n"
            "\n"
            "    async def __call__(self, scope, receive, send):\n"
            "        # Runtime ASGI entrypoint\n"
            "        await self.app(scope, receive, send)\n"
            "\n"
            "def handle_request(dispatcher, scope, receive, send):\n"
            "    return dispatcher(scope, receive, send)\n"
        )
        col.add_chunk(
            file_path="server.py",
            start_line=1,
            end_line=15,
            content=server_code,
            symbol_name="AppDispatcher",
            defined_symbols="AppDispatcher,__call__,handle_request"
        )

        engine = RAGEngine(vector_store=dummy_vs, llm_provider=self.mock_llm)

        # Ask exact file and method question
        question = (
            "In `server.py`, find the `AppDispatcher` class and explain the purpose of its "
            "`__call__` method. Use evidence from that file."
        )

        res = engine.ask_question("synthetic_test_repo", question, top_k=5)

        top_chunks = res.get("top_chunks", [])
        self.assertGreater(len(top_chunks), 0)

        # Confirm that exact file/method retrieved the implementation rather than README
        selected_file = top_chunks[0].get("file_path")
        self.assertEqual(selected_file, "server.py")
        self.assertIn("async def __call__", top_chunks[0].get("content", ""))
        self.assertNotIn("README.md", [c.get("file_path") for c in top_chunks])

    # =========================================================================
    # Test 11: Symbol query without explicit filename (Live Bug Regression Test)
    # =========================================================================
    def test_11_symbol_only_query_without_filename_resolves_fastapi_call(self):
        """
        Live Bug Regression Test:
        User asks: 'Find the FastAPI class and explain the purpose of its __call__ method. Include relevant source citations.'
        Without mentioning fastapi/applications.py, the system must resolve FastAPI and __call__ to
        fastapi/applications.py, retrieve the real implementation, and produce a grounded response.
        """
        question = "Find the FastAPI class and explain the purpose of its __call__ method. Include relevant source citations."
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", False)
        top_chunks = res.get("top_chunks", [])

        # 1. Response must be grounded with supporting sources
        self.assertTrue(grounded, f"Expected response to be grounded, but got grounded={grounded}, answer={answer}")
        self.assertGreater(len(sources), 0, "Expected at least one supporting source")

        source_files = [s["file_path"] for s in sources]
        self.assertTrue(
            any("applications.py" in sf for sf in source_files),
            f"Expected fastapi/applications.py in sources, got: {source_files}"
        )
        self.assertFalse(
            any("readme" in sf.lower() for sf in source_files),
            f"README should not be in sources: {source_files}"
        )

        # 2. Answer must cite the real implementation file and mention FastAPI and __call__
        self.assertIn("FastAPI", answer)
        self.assertIn("__call__", answer)
        self.assertTrue(any("applications.py" in s["file_path"] for s in sources))

        # 3. Top retrieved chunks must contain the actual __call__ method implementation
        top_contents = " ".join(c.get("content", "") for c in top_chunks)
        self.assertIn("__call__", top_contents)

    # =========================================================================
    # Test 12: Non-existent symbol produces explicit uncertainty
    # =========================================================================
    def test_12_nonexistent_symbol_produces_explicit_uncertainty(self):
        """
        Verify that querying a non-existent class/method symbol produces an explicit
        uncertainty response, grounded=False, and empty sources.
        """
        question = "Find the GhostService class and explain its mysterious_action method. Include relevant source citations."
        res = self.real_engine.ask_question(self.fastapi_repo_id, question, top_k=5)

        answer = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", True)

        self.assertFalse(grounded, "Response for non-existent symbol must not be grounded")
        self.assertEqual(len(sources), 0, "Sources must be empty for non-existent symbol")
        self.assertTrue(
            any(k in answer.lower() for k in ("not available in the retrieved evidence", "not found")),
            f"Expected explicit uncertainty response, got: {answer}"
        )

    # =========================================================================
    # Test 13: Unsupported answer without citations cannot be grounded
    # =========================================================================
    def test_13_unsupported_answer_without_citations_cannot_be_grounded(self):
        """
        Verify that finalize_rag_response ensures answers without valid visible citations
        are strictly marked grounded=False and have sources=[].
        """
        raw_unsupported_answer = (
            "The __call__ method initializes the instance, configures OpenAPI routes, "
            "adds background tasks, and sets up startup/shutdown handlers."
        )
        evidence_chunks = [{
            "file_path": "fastapi/applications.py",
            "start_line": 1148,
            "end_line": 1247,
            "symbol_name": "FastAPI",
            "content": "async def __call__(self, scope, receive, send): pass"
        }]

        question = "Find the FastAPI class and explain the purpose of its __call__ method. Include relevant source citations."
        final_answer, sources, grounded = finalize_rag_response(
            raw_answer=raw_unsupported_answer,
            llm_context_chunks=evidence_chunks,
            is_call_chain=False,
            question=question
        )

        self.assertFalse(grounded, "Answer without citations must not be grounded")
        self.assertEqual(sources, [], "Sources must be empty when answer is ungrounded")
        self.assertTrue(
            "not available in the retrieved evidence" in final_answer.lower() or not grounded,
            f"Expected ungrounded status or uncertainty, got: {final_answer}"
        )

    # =========================================================================
    # Test 14: Prose and comma citation normalization
    # =========================================================================
    def test_14_prose_and_comma_citation_normalization(self):
        """
        Verify that prose citations and comma-delimited citations to evidence files
        are properly normalized and validated into bracketed citations.
        """
        evidence_chunks = [{
            "file_path": "fastapi/applications.py",
            "start_line": 1148,
            "end_line": 1247,
            "symbol_name": "FastAPI",
            "content": "async def __call__(self, scope, receive, send): pass"
        }]

        raw_answer = (
            "The method is defined in the fastapi/applications.py file, lines 1148-1197. "
            "It delegates to middleware [fastapi/applications.py, lines 1198-1247]."
        )

        cleaned, valid, invalid = extract_and_validate_citations(raw_answer, evidence_chunks)
        self.assertGreater(len(valid), 0, "Expected normalized citations to be validated")
        self.assertEqual(len(invalid), 0, f"Expected 0 invalid citations, got: {invalid}")
        self.assertTrue(
            "[fastapi/applications.py:1148-1197]" in cleaned or "[fastapi/applications.py:1148-1247]" in cleaned,
            f"Expected normalized citation, got: {cleaned}"
        )
        self.assertTrue(
            "[fastapi/applications.py:1198-1247]" in cleaned or "[fastapi/applications.py:1148-1247]" in cleaned,
            f"Expected normalized citation, got: {cleaned}"
        )

    # =========================================================================
    # Test 15: Workflow question prioritizes implementation over README documentation
    # =========================================================================
    def test_15_workflow_question_prioritizes_implementation_over_readme(self):
        """
        Verify that workflow questions (e.g. 'How does the application answer questions about source code?')
        are classified as runtime/workflow rather than general repository purpose,
        and that retrieved code candidates prioritize actual implementation files (routes, engine, llm)
        over README.md documentation.
        Verify that all 6 stages are described exactly once in chronological execution order with
        narrow valid citations and correct function attributions.
        """
        mock_vs = DummyVectorStore()
        col = mock_vs.get_or_create_collection("test-rag-workflow-repo")

        # Add README chunk
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=100,
            content="# RepoLens AI\nRepoLens AI is an AI-powered code analysis tool that indexes repositories and provides chat capabilities.",
            symbol_name=""
        )

        # Add route chunk
        col.add_chunk(
            file_path="backend/app/routes/chat.py",
            start_line=48,
            end_line=184,
            content="@router.post('/chat')\ndef ask_question(repo_id: str, request: QuestionRequest):\n    return rag_engine.ask_question(repo_id, request.question)",
            symbol_name="ask_question",
            defined_symbols="ask_question"
        )

        # Add RAG engine retrieval chunk
        col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=1258,
            end_line=1307,
            content="def build_retrieval_queries(self, question: str):\n    return [question]\ndef ask_question(self, repo_id: str, question: str):\n    return finalize_rag_response(answer, results)",
            symbol_name="RAGEngine",
            defined_symbols="build_retrieval_queries,ask_question"
        )

        # Add vector store chunk
        col.add_chunk(
            file_path="backend/app/ai/vector_store.py",
            start_line=14,
            end_line=410,
            content="class ChromaManager:\n    def query_similar_chunks(self, repository_id: str, query_text: str):\n        return []",
            symbol_name="ChromaManager",
            defined_symbols="ChromaManager,query_similar_chunks"
        )

        # Add LLM inference chunk
        col.add_chunk(
            file_path="backend/app/ai/llm.py",
            start_line=316,
            end_line=582,
            content="class OllamaLLM(LLMProvider):\n    def generate_response(self, system_prompt: str, user_prompt: str):\n        return ''",
            symbol_name="OllamaLLM",
            defined_symbols="OllamaLLM,generate_response"
        )

        # Add citation validation chunk
        col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=257,
            end_line=491,
            content="def extract_and_validate_citations(answer: str, evidence_chunks: list):\n    return cleaned, valid, invalid\ndef finalize_rag_response(answer, results):\n    return answer, sources, grounded",
            symbol_name="extract_and_validate_citations",
            defined_symbols="extract_and_validate_citations,finalize_rag_response"
        )

        test_engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)

        q = "How does the application answer questions about source code?"
        classification = test_engine.classify_question(q)
        self.assertEqual(classification, "runtime", f"Expected 'runtime' classification for workflow query, got '{classification}'")

        res = test_engine.ask_question("test-rag-workflow-repo", q, top_k=5)
        ans = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", False)

        self.assertTrue(grounded, "Workflow answer must be grounded")
        source_files = [s.get("file_path") for s in sources]
        self.assertTrue(
            any("chat.py" in f or "rag.py" in f for f in source_files),
            f"Expected implementation files in sources, got: {source_files}"
        )
        self.assertNotIn("README.md", source_files, "Workflow answer must prioritize implementation files over README.md")
        self.assertIn("RAG (Retrieval-Augmented Generation) pipeline", ans)

        # Verify stages in execution order
        stage1_idx = ans.find("Chat API Request Handling")
        stage2_idx = ans.find("Repository Code Retrieval & Ranking")
        stage3_idx = ans.find("Prompt Construction")
        stage4_idx = ans.find("Ollama Inference")
        stage5_idx = ans.find("Citation Extraction & Validation")
        stage6_idx = ans.find("Response Persistence & Return")

        self.assertGreaterEqual(stage1_idx, 0, "Stage 1 (Chat API Request Handling) must be present")
        self.assertGreater(stage2_idx, stage1_idx, "Stage 2 must follow Stage 1")
        self.assertGreater(stage3_idx, stage2_idx, "Stage 3 must follow Stage 2")
        self.assertGreater(stage4_idx, stage3_idx, "Stage 4 must follow Stage 3")
        self.assertGreater(stage5_idx, stage4_idx, "Stage 5 must follow Stage 4")
        self.assertGreater(stage6_idx, stage5_idx, "Stage 6 must follow Stage 5")

        # Verify no repeated stages
        self.assertEqual(ans.count("Chat API Request Handling"), 1)
        self.assertEqual(ans.count("Repository Code Retrieval & Ranking"), 1)
        self.assertEqual(ans.count("Prompt Construction"), 1)
        self.assertEqual(ans.count("Ollama Inference"), 1)
        self.assertEqual(ans.count("Citation Extraction & Validation"), 1)
        self.assertEqual(ans.count("Response Persistence & Return"), 1)

        # Verify stage attributions to actual functions
        self.assertIn("backend/app/routes/chat.py", ans)
        self.assertIn("ask_question", ans)
        self.assertIn("build_retrieval_queries", ans)
        self.assertIn("query_similar_chunks", ans)
        self.assertIn("OllamaLLM.generate_response", ans)
        self.assertIn("extract_and_validate_citations", ans)
        self.assertIn("finalize_rag_response", ans)

        # Verify prompt construction does NOT perform response validation
        self.assertTrue(
            "prompt construction" in ans.lower() and "does not perform response validation" in ans.lower(),
            "Prompt construction must explicitly not perform response validation"
        )
        self.assertNotIn("format_rag_user_prompt", ans)

        # Verify narrow citations
        self.assertIn("[backend/app/routes/chat.py:48-134]", ans)
        self.assertIn("[backend/app/ai/rag.py:1258-1307]", ans)
        self.assertIn("[backend/app/ai/vector_store.py:213-248]", ans)
        self.assertIn("[backend/app/ai/llm.py:417-465]", ans)
        self.assertIn("[backend/app/ai/rag.py:411-491]", ans)
        self.assertIn("[backend/app/routes/chat.py:152-184]", ans)

    # =========================================================================
    # Test 16: Workflow stage attribution, narrow citations, and prompt validation isolation
    # =========================================================================
    def test_16_workflow_stage_attribution_and_narrow_citations(self):
        """
        Verify that extract_and_validate_citations accepts narrow valid sub-ranges within chunk bounds,
        and does not widen them or attribute validation to prompt formatting.
        """
        evidence_chunks = [
            {
                "file_path": "backend/app/routes/chat.py",
                "start_line": 48,
                "end_line": 184,
                "symbol_name": "ask_question",
                "content": "@router.post('/chat')\ndef ask_question(...):\n    ...\n    return QuestionResponse(...)"
            },
            {
                "file_path": "backend/app/ai/vector_store.py",
                "start_line": 14,
                "end_line": 410,
                "symbol_name": "ChromaManager",
                "content": "class ChromaManager:\n    def query_similar_chunks(...):\n        pass"
            },
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 257,
                "end_line": 491,
                "symbol_name": "extract_and_validate_citations",
                "content": "def extract_and_validate_citations(...):\n    pass"
            }
        ]

        # Valid answer with narrow line citations within chunk boundaries
        valid_answer = (
            "1. Request received: [backend/app/routes/chat.py:48-134].\n"
            "2. Vector query: [backend/app/ai/vector_store.py:213-248].\n"
            "3. Citation validation: [backend/app/ai/rag.py:411-491].\n"
            "4. Persistence: [backend/app/routes/chat.py:152-184]."
        )

        cleaned, valid, invalid = extract_and_validate_citations(valid_answer, evidence_chunks)
        self.assertEqual(len(invalid), 0, f"Expected 0 invalid citations, got: {invalid}")
        self.assertEqual(len(valid), 4, f"Expected 4 valid citations, got: {len(valid)}")
        # Ensure narrow line ranges are preserved
        self.assertIn("[backend/app/routes/chat.py:48-134]", cleaned)
        self.assertIn("[backend/app/routes/chat.py:152-184]", cleaned)
        self.assertIn("[backend/app/ai/vector_store.py:213-248]", cleaned)
        self.assertIn("[backend/app/ai/rag.py:411-491]", cleaned)

        # Verify that an answer falsely attributing validation to format_rag_user_prompt
        # without supporting evidence chunk is marked invalid and ungrounded
        bogus_answer = (
            "Response validation is performed by format_rag_user_prompt in [backend/app/ai/prompts.py:51-100]."
        )
        b_clean, b_valid, b_invalid = extract_and_validate_citations(bogus_answer, evidence_chunks)
        self.assertIn("[backend/app/ai/prompts.py:51-100]", b_invalid)
        self.assertEqual(len(b_valid), 0)

        final_ans, sources, grounded = finalize_rag_response(
            bogus_answer,
            evidence_chunks,
            False,
            "How does the application validate citations?"
        )
        self.assertFalse(grounded)
        self.assertEqual(sources, [])

    # =========================================================================
    # Test 17: False-positive symbol extraction (multi-part architecture)
    # =========================================================================
    def test_17_false_positive_symbol_extraction_multi_part_architecture(self):
        """
        Verify that generic architectural and conceptual words such as multi, part,
        architecture, system, project, component, structure, and pattern are NOT
        treated as explicit code symbols, and that conceptual/architectural questions
        proceed to normal vector retrieval rather than aborting as missing symbols.
        """
        conceptual_questions = [
            "Explain the multi-part architecture of this application",
            "What is the system architecture?",
            "Explain the project structure and component patterns",
            "How does the multi-part architecture work?",
        ]
        for q in conceptual_questions:
            refs = extract_target_references(q)
            for bad_sym in ("multi", "part", "architecture", "system", "project", "component", "structure", "pattern"):
                self.assertNotIn(
                    bad_sym,
                    [s.lower() for s in refs["symbols"]],
                    f"Word '{bad_sym}' should not be extracted as symbol in '{q}'"
                )

        # End-to-end RAGEngine retrieval verification with DummyVectorStore
        mock_vs = DummyVectorStore()
        col = mock_vs.get_or_create_collection("test-multi-part-repo")
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=40,
            content="# RepoLens AI\nMulti-part architecture with frontend, FastAPI backend, and ChromaDB.",
            symbol_name=""
        )
        col.add_chunk(
            file_path="backend/app/main.py",
            start_line=1,
            end_line=50,
            content="app = FastAPI(title='RepoLens AI')\n# Core backend application entrypoint",
            symbol_name=""
        )

        engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)
        res = engine.ask_question("test-multi-part-repo", "Explain the multi-part architecture of this application")

        # Must not be aborted by missing symbol safeguard
        self.assertNotIn("The requested implementation is not available in the retrieved evidence", res.get("answer", ""))
        self.assertTrue(res.get("grounded", False), "Expected multi-part architecture query to be grounded")
        self.assertGreater(len(res.get("sources", [])), 0, "Expected supporting sources")

    # =========================================================================
    # Test 18: Genuinely missing explicit symbol safeguards preserved
    # =========================================================================
    def test_18_genuinely_missing_explicit_symbol_safeguards(self):
        """
        Verify that genuinely explicit function, class, and identifier lookups
        still trigger missing-symbol validation when the symbol is not in the codebase.
        """
        mock_vs = DummyVectorStore()
        mock_vs.get_or_create_collection("test-missing-repo")
        engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)

        # 1. Explicit class keyword
        res1 = engine.ask_question("test-missing-repo", "Where is class NonExistentClass defined?")
        self.assertFalse(res1.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res1.get("answer", ""))

        # 2. Explicit backticked symbol
        res2 = engine.ask_question("test-missing-repo", "Explain what `nonexistent_symbol_xyz` does.")
        self.assertFalse(res2.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res2.get("answer", ""))

        # 3. Explicit def keyword
        res3 = engine.ask_question("test-missing-repo", "Find def fake_helper_function")
        self.assertFalse(res3.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res3.get("answer", ""))

    # =========================================================================
    # Test 19: MockLLM fallback provides evidence-derived explanation
    # =========================================================================
    def test_19_general_file_question_mock_llm_explanation(self):
        """
        Verify that general questions about backend/app/ai/rag.py no longer receive
        only a citation sentence, but a concise explanation derived strictly from
        retrieved evidence, and that insufficient evidence is stated truthfully.
        """
        mock_llm = MockLLM()

        # Evidence chunk defining RAGEngine
        rag_prompt = (
            "USER QUESTION:\nWhat does backend/app/ai/rag.py do?\n\n"
            "REPOSITORY CODE EVIDENCE:\n"
            "FILE: backend/app/ai/rag.py\n"
            "LINES: 100-150\n"
            "SYMBOL: RAGEngine\n"
            "CODE:\n"
            "class RAGEngine:\n"
            "    \"\"\"Coordinate retrieval-augmented generation across repositories.\"\"\"\n"
            "    def ask_question(self, repo_id: str, question: str):\n"
            "        pass\n"
            "    def classify_question(self, question: str):\n"
            "        pass\n"
        )

        resp = mock_llm.generate_response("System prompt", rag_prompt)

        # 1. Must NOT be the old single citation sentence
        self.assertNotEqual(
            resp.strip(),
            "The relevant implementation is available in `backend/app/ai/rag.py` [backend/app/ai/rag.py:100-150]."
        )
        self.assertNotIn("The relevant implementation is available in `backend/app/ai/rag.py`", resp)

        # 2. Must contain evidence-derived details and citations
        self.assertIn("RAGEngine", resp)
        self.assertIn("ask_question", resp)
        self.assertIn("[backend/app/ai/rag.py:100-150]", resp)

        # 3. Insufficient evidence handling
        insufficient_prompt = (
            "USER QUESTION:\nExplain dummy.py\n\n"
            "REPOSITORY CODE EVIDENCE:\n"
            "FILE: dummy.py\n"
            "LINES: 1-2\n"
            "SYMBOL: \n"
            "CODE:\n"
            "# blank\n"
        )
        insufficient_resp = mock_llm.generate_response("System prompt", insufficient_prompt)
        self.assertIn("insufficient", insufficient_resp.lower())
        self.assertIn("[dummy.py:1-2]", insufficient_resp)


if __name__ == "__main__":
    unittest.main()
