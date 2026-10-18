import os
import sys
import unittest
from typing import List, Dict, Any

# Ensure user site-packages are accessible on Windows environment
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import (
    RAGEngine,
    extract_target_references,
)
from app.ai.llm import MockLLM


class DummyVectorStore:
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
            "distance": 0.3,
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
            results.append(copy_c)
        return results[:top_k]


class TestRAGSymbolExtractionRegression(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MockLLM()

    def test_01_reported_natural_language_queries_do_not_extract_false_symbols(self):
        """
        Regression Test 1:
        Verify that natural-language queries:
        - 'List the main Python files and explain what each one does.'
        - 'Which Python file contains the main program logic? Explain using evidence.'
        do NOT falsely extract 'List', 'Python', or 'using' as symbols.
        """
        q1 = "List the main Python files and explain what each one does."
        refs1 = extract_target_references(q1)
        syms1_lower = [s.lower() for s in refs1["symbols"]]

        self.assertNotIn("list", syms1_lower, "Word 'List' must not be extracted as symbol")
        self.assertNotIn("python", syms1_lower, "Language 'Python' must not be extracted as symbol")
        self.assertNotIn("explain", syms1_lower, "Action word 'explain' must not be extracted as symbol")
        self.assertFalse(refs1["has_explicit_symbol"], f"Expected no explicit symbols in '{q1}', got: {refs1['symbols']}")
        self.assertEqual(refs1["symbols"], [], f"Expected empty symbol list for '{q1}'")

        q2 = "Which Python file contains the main program logic? Explain using evidence."
        refs2 = extract_target_references(q2)
        syms2_lower = [s.lower() for s in refs2["symbols"]]

        self.assertNotIn("python", syms2_lower, "Language 'Python' must not be extracted as symbol")
        self.assertNotIn("using", syms2_lower, "Word 'using' following 'Explain' must not be extracted as symbol")
        self.assertNotIn("explain", syms2_lower, "Action word 'Explain' must not be extracted as symbol")
        self.assertNotIn("which", syms2_lower, "Question word 'Which' must not be extracted as symbol")
        self.assertFalse(refs2["has_explicit_symbol"], f"Expected no explicit symbols in '{q2}', got: {refs2['symbols']}")
        self.assertEqual(refs2["symbols"], [], f"Expected empty symbol list for '{q2}'")

    def test_02_genuine_explicit_symbol_lookup_preserved(self):
        """
        Regression Test 2:
        Verify that genuine explicit symbols such as calculate_damage and def process_stream
        are correctly identified as symbols.
        """
        # 1. Multi-token identifier with underscore
        refs_dmg = extract_target_references("What does calculate_damage do?")
        self.assertIn("calculate_damage", refs_dmg["symbols"])
        self.assertTrue(refs_dmg["has_explicit_symbol"])

        # 2. Explicit def keyword qualification
        refs_def = extract_target_references("Find def process_stream")
        self.assertIn("process_stream", refs_def["symbols"])
        self.assertNotIn("def", [s.lower() for s in refs_def["symbols"]], "'def' keyword must not be extracted as symbol")
        self.assertTrue(refs_def["has_explicit_symbol"])

        # 3. Explicit class keyword qualification
        refs_cls = extract_target_references("Where is class NonExistentClass defined?")
        self.assertIn("NonExistentClass", refs_cls["symbols"])
        self.assertTrue(refs_cls["has_explicit_symbol"])

        # 4. Explicit backticked symbol
        refs_bt = extract_target_references("Explain what `nonexistent_symbol_xyz` does.")
        self.assertIn("nonexistent_symbol_xyz", refs_bt["symbols"])
        self.assertTrue(refs_bt["has_explicit_symbol"])

    def test_03_end_to_end_natural_language_queries_proceed_to_retrieval(self):
        """
        Verify that the two natural language queries proceed to normal vector retrieval
        and are NOT aborted by the missing-symbol safeguard.
        """
        mock_vs = DummyVectorStore()
        col = mock_vs.get_or_create_collection("test-nl-repo")
        col.add_chunk(
            file_path="README.md",
            start_line=1,
            end_line=25,
            content="# Python Projects\nThis repository contains beginner friendly Python projects including app.py and helper.py.",
            symbol_name=""
        )
        col.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=30,
            content="def main():\n    print('Main program entry point for Python beginner projects')\n",
            symbol_name="main"
        )

        engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)

        # Query 1
        res1 = engine.ask_question("test-nl-repo", "List the main Python files and explain what each one does.")
        self.assertNotIn("The requested implementation is not available in the retrieved evidence", res1.get("answer", ""))
        self.assertTrue(res1.get("grounded", False), "Expected response to be grounded")
        self.assertGreater(len(res1.get("sources", [])), 0, "Expected sources in response")

        # Query 2
        res2 = engine.ask_question("test-nl-repo", "Which Python file contains the main program logic? Explain using evidence.")
        self.assertNotIn("The requested implementation is not available in the retrieved evidence", res2.get("answer", ""))
        self.assertTrue(res2.get("grounded", False), "Expected response to be grounded")
        self.assertGreater(len(res2.get("sources", [])), 0, "Expected sources in response")

    def test_04_missing_explicit_symbol_safeguard_remains_active(self):
        """
        Verify that a genuinely explicit symbol query that does not exist in the codebase
        still triggers the missing-symbol safeguard.
        """
        mock_vs = DummyVectorStore()
        mock_vs.get_or_create_collection("test-missing-repo")
        engine = RAGEngine(vector_store=mock_vs, llm_provider=self.mock_llm)

        # Genuinely explicit def lookup
        res_def = engine.ask_question("test-missing-repo", "Find def nonexistent_function_xyz")
        self.assertFalse(res_def.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res_def.get("answer", ""))
        self.assertEqual(res_def.get("sources", []), [])

        # Genuinely explicit backticked lookup
        res_bt = engine.ask_question("test-missing-repo", "Explain what `nonexistent_symbol_abc` does.")
        self.assertFalse(res_bt.get("grounded", True))
        self.assertIn("The requested implementation is not available in the retrieved evidence", res_bt.get("answer", ""))
        self.assertEqual(res_bt.get("sources", []), [])


if __name__ == "__main__":
    unittest.main()
