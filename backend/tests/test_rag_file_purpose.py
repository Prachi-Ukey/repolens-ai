import re
import unittest

from app.ai.rag import RAGEngine
from app.ai.llm import MockLLM


class DummyCollection:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.chunks = []

    def count(self):
        return len(self.chunks)

    def add_chunk(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        symbol_name: str = "",
        defined_symbols: str = ""
    ):
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
        results = []
        for c in self.chunks:
            copy_c = dict(c)
            # Default distance
            copy_c["distance"] = 0.40
            results.append(copy_c)
        return results[:top_k]


class DummyVectorStore:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, repo_id: str):
        if repo_id not in self.collections:
            self.collections[repo_id] = DummyCollection(repo_id)
        return self.collections[repo_id]

    def query_similar_chunks(self, repository_id: str = "", query_text: str = "", top_k: int = 20, **kwargs):
        repo_id = repository_id or kwargs.get("repo_id", "")
        query = query_text or kwargs.get("query", "")
        col = self.get_or_create_collection(repo_id)
        return col.query_similar(query, top_k=top_k)


class TestRAGFilePurposeAndArchitecture(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MockLLM()
        self.mock_vs = DummyVectorStore()
        self.repo_id = "test-purpose-repo"
        self.col = self.mock_vs.get_or_create_collection(self.repo_id)

        # 1. Target file chunks for backend/app/ai/rag.py
        # Chunk 1: Top imports and helper function safe_print (lines 1-30)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=1,
            end_line=30,
            symbol_name="safe_print",
            defined_symbols="safe_print",
            content="""import logging
import sys
def safe_print(text: str) -> None:
    \"\"\"Print safely to stdout without crashing on Windows cp1252 consoles.\"\"\"
    try:
        print(text)
    except Exception:
        pass
LOCK_FILES = {"package-lock.json", "yarn.lock"}
IGNORED_RETRIEVAL_DIRS = {"node_modules", ".git"}
"""
        )

        # Chunk 2: Citation validation (lines 257-491)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=257,
            end_line=491,
            symbol_name="extract_and_validate_citations",
            defined_symbols="extract_and_validate_citations",
            content="""def extract_and_validate_citations(
    answer: str,
    evidence_chunks: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    \"\"\"Extract citations from the LLM answer and validate them against evidence.\"\"\"
    pass
"""
        )

        # Chunk 3: Response finalization (lines 494-543)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=494,
            end_line=543,
            symbol_name="finalize_rag_response",
            defined_symbols="finalize_rag_response",
            content="""def finalize_rag_response(
    raw_answer: str,
    llm_context_chunks: List[Dict[str, Any]],
    is_call_chain: bool,
    question: str
) -> Tuple[str, List[Dict[str, Any]], bool]:
    \"\"\"Finalize RAG response by validating citations and compiling verified sources.\"\"\"
    pass
"""
        )

        # Chunk 4: Core RAGEngine class and classify_question (lines 1008-1057)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=1008,
            end_line=1057,
            symbol_name="RAGEngine",
            defined_symbols="RAGEngine,classify_question",
            content="""class RAGEngine:
    \"\"\"Coordinate retrieval-augmented generation across repositories.\"\"\"

    def __init__(self, vector_store, llm_provider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    def classify_question(self, question: str) -> str:
        \"\"\"Classify a user question to guide retrieval strategy.\"\"\"
        pass
"""
        )

        # Chunk 5: build_retrieval_queries (lines 1258-1307)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=1258,
            end_line=1307,
            symbol_name="RAGEngine",
            defined_symbols="build_retrieval_queries",
            content="""    def build_retrieval_queries(self, question: str, question_type: str = "general") -> List[str]:
        \"\"\"Build retrieval queries for vector search.\"\"\"
        pass
"""
        )

        # Chunk 6: ask_question (lines 1458-1507)
        self.col.add_chunk(
            file_path="backend/app/ai/rag.py",
            start_line=1458,
            end_line=1507,
            symbol_name="RAGEngine",
            defined_symbols="ask_question",
            content="""    def ask_question(self, repo_id: str, question: str, top_k: int = 5, stream: bool = False) -> Dict[str, Any]:
        \"\"\"Execute end-to-end question answering pipeline.\"\"\"
        pass
"""
        )

        # Chunk 7: Unrelated file in repository
        self.col.add_chunk(
            file_path="backend/app/routes/auth.py",
            start_line=14,
            end_line=25,
            symbol_name="login",
            defined_symbols="login",
            content="""@router.post('/login')
def login(request: UserLogin):
    \"\"\"Handle user login and token issuance.\"\"\"
    pass
"""
        )

        self.engine = RAGEngine(vector_store=self.mock_vs, llm_provider=self.mock_llm)

    # =========================================================================
    # Test 1: File-level question about backend/app/ai/rag.py
    # =========================================================================
    def test_rag_py_purpose_and_responsibilities_retrieval_and_answer(self):
        """
        Verify that a question asking about backend/app/ai/rag.py's purpose and
        main responsibilities in 3-5 bullet points:
        1. Prioritizes RAGEngine, classify_question, ask_question over safe_print.
        2. Never claims safe_print is the purpose of the file.
        3. Returns an answer structured with an opening purpose sentence and 3-5 bullet points.
        4. Provides precise citations strictly supported by retrieved evidence.
        5. Does not cite or describe unrelated files (like auth.py).
        """
        question = (
            "What is the purpose of backend/app/ai/rag.py? Explain its main responsibilities "
            "in 3–5 bullet points, using relevant code evidence and precise line citations. "
            "Do not describe unrelated files."
        )

        res = self.engine.ask_question(self.repo_id, question, top_k=5)
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        grounded = res.get("grounded", False)
        top_chunks = res.get("top_chunks", [])

        # 1. Grounding and Sources verification
        self.assertTrue(grounded, "Response must be grounded")
        self.assertGreater(len(sources), 0, "Response must contain sources")
        for s in sources:
            self.assertEqual(
                s.get("file_path"),
                "backend/app/ai/rag.py",
                f"Source should be backend/app/ai/rag.py, got: {s.get('file_path')}"
            )

        # 2. Top chunks must include core RAGEngine and public methods, NOT safe_print
        top_symbols = [c.get("symbol_name") for c in top_chunks]
        self.assertIn("RAGEngine", top_symbols)
        self.assertNotIn("safe_print", [c.get("symbol_name") for c in top_chunks[:3]])

        # 3. Answer content verification
        self.assertIn("RAGEngine", answer)
        self.assertNotIn(
            "the implementation defines safe_print",
            answer.lower(),
            "safe_print must not be reported as the purpose of the file"
        )
        self.assertNotIn(
            "windows cp1252",
            answer.lower(),
            "windows cp1252 console printing must not be reported as the purpose of rag.py"
        )
        self.assertNotIn("auth.py", answer.lower(), "Unrelated files must not be described")

        # 4. 3-5 bullet points format verification
        bullet_items = [l.strip() for l in answer.splitlines() if l.strip().startswith("- ")]
        self.assertGreaterEqual(len(bullet_items), 3, "Must have at least 3 bullet points")
        self.assertLessEqual(len(bullet_items), 5, "Must have at most 5 bullet points")

        # 5. Core responsibilities explained with citations
        for b in bullet_items:
            # Each bullet point must have an inline citation [backend/app/ai/rag.py:...]
            self.assertTrue(
                bool(re.search(r"\[backend/app/ai/rag\.py:\d+-\d+\]", b)),
                f"Bullet point must have inline citation: {b}"
            )

        # Check key responsibilities are mentioned across bullets
        all_bullets_text = " ".join(bullet_items).lower()
        self.assertTrue(
            any(k in all_bullets_text for k in ("classification", "classify")),
            "Must describe question classification"
        )
        self.assertTrue(
            any(k in all_bullets_text for k in ("retrieval", "queries", "query")),
            "Must describe retrieval query construction"
        )
        self.assertTrue(
            any(k in all_bullets_text for k in ("pipeline", "orchestration", "ask_question", "answering")),
            "Must describe pipeline orchestration"
        )

    # =========================================================================
    # Test 2: Generic file purpose (verifying not hardcoded to rag.py)
    # =========================================================================
    def test_generic_file_purpose_and_responsibilities(self):
        """
        Verify that file-purpose detection works generically for any repository file
        (e.g. src/processor.py) with custom classes and methods.
        """
        generic_repo_id = "test-generic-repo"
        gen_col = self.mock_vs.get_or_create_collection(generic_repo_id)

        # Trivial helper in chunk 1
        gen_col.add_chunk(
            file_path="src/processor.py",
            start_line=1,
            end_line=15,
            symbol_name="log_info",
            defined_symbols="log_info",
            content="""import logging
def log_info(msg):
    \"\"\"Helper to log message.\"\"\"
    logging.info(msg)
"""
        )

        # Core class in chunk 2
        gen_col.add_chunk(
            file_path="src/processor.py",
            start_line=20,
            end_line=70,
            symbol_name="DataProcessor",
            defined_symbols="DataProcessor",
            content="""class DataProcessor:
    \"\"\"Coordinate data ingestion and transformation pipelines.\"\"\"

    def process_records(self, records: list) -> list:
        \"\"\"Process incoming records through transformation stages.\"\"\"
        pass
"""
        )

        # Core public methods in chunk 3
        gen_col.add_chunk(
            file_path="src/processor.py",
            start_line=75,
            end_line=120,
            symbol_name="DataProcessor",
            defined_symbols="validate_schema,export_results",
            content="""    def validate_schema(self, dataset: dict) -> bool:
        \"\"\"Validate schema integrity against defined rules.\"\"\"
        pass

    def export_results(self, output_path: str) -> None:
        \"\"\"Export processed records to persistent storage.\"\"\"
        pass
"""
        )

        engine = RAGEngine(vector_store=self.mock_vs, llm_provider=self.mock_llm)
        question = "What is the purpose of src/processor.py? Explain its main responsibilities in 3-5 bullet points."

        res = engine.ask_question(generic_repo_id, question, top_k=5)
        answer = res.get("answer", "")
        grounded = res.get("grounded", False)
        sources = res.get("sources", [])

        self.assertTrue(grounded)
        self.assertGreater(len(sources), 0)
        self.assertIn("DataProcessor", answer)
        self.assertNotIn("log_info", answer.lower(), "Helper log_info should not be main purpose")

        bullets = [l.strip() for l in answer.splitlines() if l.strip().startswith("- ")]
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), 5)
        for b in bullets:
            self.assertTrue(bool(re.search(r"\[src/processor\.py:\d+-\d+\]", b)))

    # =========================================================================
    # Test 3: General architecture question
    # =========================================================================
    def test_general_architecture_question_retrieval_and_grounding(self):
        """
        Verify that general architecture questions:
        1. Retrieve multi-part architectural components (frontend, backend, storage).
        2. Are not falsely blocked by missing symbol safeguards.
        3. Produce grounded responses with valid citations.
        """
        arch_repo_id = "test-arch-repo"
        arch_col = self.mock_vs.get_or_create_collection(arch_repo_id)

        arch_col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=30,
            symbol_name="",
            content="""# RepoLens Architecture
The application features a modern multi-part architecture:
- Frontend: React with Vite and Tailwind CSS
- Backend: FastAPI REST API
- Vector Store: ChromaDB with persistent embeddings
"""
        )
        arch_col.add_chunk(
            file_path="backend/app/main.py",
            start_line=1,
            end_line=40,
            symbol_name="app",
            content="""from fastapi import FastAPI
app = FastAPI(title="RepoLens AI")
# Mount API routers for chat, repository, and insights
"""
        )

        engine = RAGEngine(vector_store=self.mock_vs, llm_provider=self.mock_llm)
        question = "Explain the multi-part architecture and core system components of this application."

        res = engine.ask_question(arch_repo_id, question, top_k=5)
        answer = res.get("answer", "")
        grounded = res.get("grounded", False)
        sources = res.get("sources", [])

        self.assertTrue(grounded)
        self.assertGreater(len(sources), 0)
        self.assertNotIn("The requested implementation is not available in the retrieved evidence", answer)
        self.assertTrue(any("readme" in s["file_path"].lower() or "main.py" in s["file_path"].lower() for s in sources))

    # =========================================================================
    # Test 4: Genuinely missing symbol safeguards preserved
    # =========================================================================
    def test_genuinely_missing_symbol_safeguard_preserved(self):
        """
        Verify that explicit lookups for non-existent symbols produce explicit
        uncertainty rather than hallucinating an answer.
        """
        question = "In backend/app/ai/rag.py, find class FakeNonExistentClass and explain its purpose."
        res = self.engine.ask_question(self.repo_id, question, top_k=5)

        self.assertFalse(res.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res.get("answer", ""))


if __name__ == "__main__":
    unittest.main()
