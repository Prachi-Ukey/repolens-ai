import os
import sys
import unittest
import re
from typing import List, Dict, Any

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import (
    RAGEngine,
    extract_and_validate_citations,
    finalize_rag_response,
)
from app.ai.llm import MockLLM


class MockOverviewVectorStore:
    """Deterministic in-memory vector store for repository overview tests."""
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, repo_id: str):
        if repo_id not in self.collections:
            self.collections[repo_id] = MockOverviewCollection(repo_id)
        return self.collections[repo_id]

    def query_similar_chunks(self, repository_id: str, query_text: str, top_k: int = 20):
        col = self.get_or_create_collection(repository_id)
        return col.query_similar(query_text, top_k)


class MockOverviewCollection:
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
            "distance": 0.25,
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
        q_low = query_text.lower()
        for c in self.chunks:
            copy_c = dict(c)
            # Give closer distance to relevant chunks
            if "readme" in c["file_path"].lower():
                copy_c["distance"] = 0.15
            elif "structure" in q_low and "structure" in c.get("symbol_name", "").lower():
                copy_c["distance"] = 0.12
            else:
                copy_c["distance"] = 0.30
            results.append(copy_c)
        return results[:top_k]


class TestRAGRepositoryOverview(unittest.TestCase):
    """
    Regression test suite for repository-level RAG answers and citation accuracy.
    Phase 4 requirements:
    1. Project-purpose questions do not trigger a hardcoded FastAPI explanation.
    2. The project overview reflects the supplied repository README.
    3. Component questions return multiple distinct components when the evidence supports them.
    4. Folder-structure answers use actual indexed paths.
    5. File-purpose bullets cite the corresponding functions and line ranges.
    6. Missing evidence is acknowledged rather than fabricated.
    """

    def setUp(self):
        self.vector_store = MockOverviewVectorStore()
        self.mock_llm = MockLLM()
        self.engine = RAGEngine(vector_store=self.vector_store, llm_provider=self.mock_llm)

    # 1. Project-purpose questions do not trigger a hardcoded FastAPI explanation merely
    # because retrieved evidence contains the word "fastapi".
    def test_01_project_purpose_does_not_trigger_fastapi_explanation(self):
        repo_id = "test-analytics-repo"
        col = self.vector_store.get_or_create_collection(repo_id)

        # Repository code imports fastapi, but is an analytics service
        col.add_chunk(
            file_path="src/api.py",
            start_line=1,
            end_line=30,
            content="from fastapi import FastAPI, APIRouter\napp = FastAPI()\n@app.get('/metrics')\ndef metrics():\n    return {'status': 'ok'}",
            symbol_name="api",
            defined_symbols="app, metrics"
        )
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=45,
            content="# MetricStream Analytics\n\nMetricStream is an end-to-end real-time telemetry analytics platform designed to collect, aggregate, and visualize high-throughput system metrics.\n\nIt features real-time alerting, multi-cluster monitoring, and time-series aggregation.",
            symbol_name="",
            defined_symbols=""
        )

        res = self.engine.ask_question(
            repo_id=repo_id,
            question="What is this project about?",
            top_k=4
        )
        answer = res.get("answer", "")
        grounded = res.get("grounded", False)

        self.assertTrue(grounded)
        # Must NOT trigger generic FastAPI description
        self.assertNotIn("fastapi is a modern, fast (high-performance), web framework", answer.lower())
        # Must reflect the actual repository domain
        self.assertIn("metricstream", answer.lower())
        self.assertTrue("telemetry" in answer.lower() or "analytics" in answer.lower())

    # 2. The project overview reflects the supplied repository README.
    def test_02_project_overview_reflects_supplied_readme(self):
        repo_id = "test-taskflow-repo"
        col = self.vector_store.get_or_create_collection(repo_id)

        readme_text = (
            "# TaskFlow Engine\n\n"
            "TaskFlow Engine is a distributed workflow orchestration engine for asynchronous task graphs.\n\n"
            "It provides resilient dependency execution, automatic retry policies, and workflow telemetry across distributed workers."
        )
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=50,
            content=readme_text,
            symbol_name="readme",
            defined_symbols=""
        )

        res = self.engine.ask_question(
            repo_id=repo_id,
            question="What is the purpose of this repository?",
            top_k=3
        )
        answer = res.get("answer", "")
        sources = res.get("sources", [])

        self.assertIn("taskflow", answer.lower())
        self.assertIn("workflow orchestration", answer.lower())
        # Must include verified inline citation
        self.assertIn("[README.md:", answer)
        self.assertTrue(any("README.md" in s.get("file_path", "") for s in sources))

    # 3. Component questions return multiple distinct components when the evidence supports them.
    def test_03_component_questions_return_multiple_distinct_components(self):
        repo_id = "test-ecommerce-platform"
        col = self.vector_store.get_or_create_collection(repo_id)

        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=30,
            content="# CommerceHub Platform\n\nModular e-commerce backend and storefront.",
            symbol_name=""
        )
        col.add_chunk(
            file_path="frontend/src/App.jsx",
            start_line=1,
            end_line=40,
            content="export default function App() {\n  return <StorefrontCatalog />;\n}",
            symbol_name="App",
            defined_symbols="App"
        )
        col.add_chunk(
            file_path="backend/app/routes/orders.py",
            start_line=1,
            end_line=50,
            content="from fastapi import APIRouter\nrouter = APIRouter()\n@router.post('/checkout')\ndef create_order():\n    pass",
            symbol_name="create_order",
            defined_symbols="create_order, router"
        )
        col.add_chunk(
            file_path="backend/app/ai/recommender.py",
            start_line=1,
            end_line=45,
            content="class ProductRecommender:\n    def predict_recommendations(self, user_id):\n        pass",
            symbol_name="ProductRecommender",
            defined_symbols="ProductRecommender, predict_recommendations"
        )

        res = self.engine.ask_question(
            repo_id=repo_id,
            question="What are the main components of this project?",
            top_k=4
        )
        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # Must return multiple distinct components (at least 3)
        bullet_points = [line for line in answer.splitlines() if line.strip().startswith("- ")]
        self.assertGreaterEqual(len(bullet_points), 3, f"Expected at least 3 distinct component bullets, got: {answer}")

        # Verified citations attached to distinct component files
        source_files = {s.get("file_path") for s in sources}
        self.assertGreaterEqual(len(source_files), 3, f"Expected at least 3 distinct source files, got: {source_files}")

    # 4. Folder-structure answers use actual indexed paths.
    def test_04_folder_structure_answers_use_actual_indexed_paths(self):
        repo_id = "test-structure-repo"
        col = self.vector_store.get_or_create_collection(repo_id)

        structure_content = (
            "# Repository Structure and Main Folders:\n"
            "- Root files: README.md, docker-compose.yml, package.json\n"
            "- `frontend/` folder: Contains src, public, vite.config.js\n"
            "- `backend/` folder: Contains app, routes, ai, config.py\n"
            "- `models/` folder: Contains yolo_weights.pt, classifier.onnx\n"
        )
        col.add_chunk(
            file_path="REPOSITORY_STRUCTURE",
            start_line=1,
            end_line=25,
            content=structure_content,
            symbol_name="repository_structure",
            defined_symbols="repository_structure"
        )

        res = self.engine.ask_question(
            repo_id=repo_id,
            question="Explain the main folders and files in this project",
            top_k=3
        )
        answer = res.get("answer", "")
        grounded = res.get("grounded", False)

        self.assertTrue(grounded)
        # Must reflect the actual indexed paths from evidence
        self.assertIn("frontend/", answer)
        self.assertIn("backend/", answer)
        self.assertIn("models/", answer)
        self.assertIn("[REPOSITORY_STRUCTURE:1-25]", answer)

    # 5. File-purpose bullets cite the corresponding functions and line ranges without class overwrite.
    def test_05_file_purpose_bullets_cite_corresponding_functions_and_lines(self):
        evidence_chunks = [
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 1008,
                "end_line": 1057,
                "symbol_name": "RAGEngine",
                "content": "class RAGEngine:\n    \"\"\"Core retrieval augmented generation engine coordinating repository Q&A.\"\"\"\n    def __init__(self, vector_store, llm_provider):\n        pass",
                "defined_symbols": "RAGEngine, __init__"
            },
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 1488,
                "end_line": 1788,
                "symbol_name": "classify_question",
                "content": "def classify_question(self, question: str) -> str:\n    \"\"\"Classify a user question to guide retrieval strategy.\"\"\"\n    pass",
                "defined_symbols": "classify_question"
            },
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 1794,
                "end_line": 2040,
                "symbol_name": "build_retrieval_queries",
                "content": "def build_retrieval_queries(self, question: str) -> List[str]:\n    \"\"\"Build multiple retrieval queries for better code and doc retrieval.\"\"\"\n    pass",
                "defined_symbols": "build_retrieval_queries"
            },
            {
                "file_path": "backend/app/ai/rag.py",
                "start_line": 4260,
                "end_line": 4450,
                "symbol_name": "deduplicate_and_merge_chunks",
                "content": "def deduplicate_and_merge_chunks(self, chunks):\n    \"\"\"Deduplicates and merges contiguous evidence chunks.\"\"\"\n    pass",
                "defined_symbols": "deduplicate_and_merge_chunks"
            }
        ]

        llm_answer = (
            "The purpose of `backend/app/ai/rag.py` [backend/app/ai/rag.py:1008-1057] is to coordinate core RAGEngine operations.\n\n"
            "- **Query Classification**: Classifies incoming user questions via `classify_question` [backend/app/ai/rag.py:1488-1788].\n"
            "- **Query Construction**: Expands retrieval queries via `build_retrieval_queries` [backend/app/ai/rag.py:1794-2040].\n"
            "- **Chunk Deduplication**: Merges contiguous evidence spans via `deduplicate_and_merge_chunks` [backend/app/ai/rag.py:4260-4450]."
        )

        cleaned, valid, invalid = extract_and_validate_citations(llm_answer, evidence_chunks)

        self.assertEqual(len(invalid), 0, f"Expected 0 invalid citations, got: {invalid}")
        self.assertEqual(len(valid), 4, f"Expected 4 valid citations, got: {len(valid)}")

        # Crucial regression assertions:
        # 1. Opening class line cites RAGEngine
        self.assertIn("[backend/app/ai/rag.py:1008-1057]", cleaned)
        # 2. Bullets MUST NOT be overwritten with RAGEngine line range 1008-1057
        self.assertIn("[backend/app/ai/rag.py:1488-1788]", cleaned)
        self.assertIn("[backend/app/ai/rag.py:1794-2040]", cleaned)
        self.assertIn("[backend/app/ai/rag.py:4260-4450]", cleaned)

        # Confirm 1008-1057 only appears once (in the opening sentence) and did not overwrite the 3 bullets
        self.assertEqual(cleaned.count("[backend/app/ai/rag.py:1008-1057]"), 1)

    # 6. Missing evidence is acknowledged rather than fabricated.
    def test_06_missing_evidence_is_acknowledged_rather_than_fabricated(self):
        repo_id = "test-empty-repo"
        # Collection has no chunks matching quantum cryptography
        col = self.vector_store.get_or_create_collection(repo_id)
        col.add_chunk(
            file_path="src/utils.py",
            start_line=1,
            end_line=10,
            content="def add(a, b):\n    return a + b\n",
            symbol_name="add",
            defined_symbols="add"
        )

        res = self.engine.ask_question(
            repo_id=repo_id,
            question="Where is the quantum key distribution algorithm implemented?",
            top_k=3
        )
        answer = res.get("answer", "")
        grounded = res.get("grounded", False)

        # Must NOT claim evidence exists or invent code
        self.assertFalse(grounded)
        self.assertTrue(
            "not found" in answer.lower()
            or "couldn't find" in answer.lower()
            or "unavailable" in answer.lower()
            or "not available" in answer.lower()
            or "insufficient evidence" in answer.lower(),
            f"Expected missing evidence acknowledgment, got: {answer}"
        )


if __name__ == "__main__":
    unittest.main()
