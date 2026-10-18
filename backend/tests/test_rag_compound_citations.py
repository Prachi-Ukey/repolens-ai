import unittest
from app.ai.rag import extract_and_validate_citations, finalize_rag_response


class TestRAGCompoundCitations(unittest.TestCase):
    """
    Regression and feature tests for compound citation parsing,
    evidence validation, and response grounding in RepoLens AI RAG.
    """

    def setUp(self):
        self.readme_chunks = [
            {
                "file_path": "README.md",
                "start_line": 1,
                "end_line": 50,
                "content": "# RepoLens AI\nInteractive repository analysis tool.\nFeatures and overview.",
            },
            {
                "file_path": "README.md",
                "start_line": 41,
                "end_line": 90,
                "content": "Architecture and pipeline.\nIndexing, embeddings, and chat workflow.",
            },
        ]
        self.code_chunks = [
            {
                "file_path": "backend/app/main.py",
                "start_line": 1,
                "end_line": 60,
                "content": "from fastapi import FastAPI\napp = FastAPI()",
            },
            {
                "file_path": "backend/app/routes/chat.py",
                "start_line": 20,
                "end_line": 80,
                "content": "@router.post('/chat')\ndef ask_question(): ...",
            },
        ]
        self.all_chunks = self.readme_chunks + self.code_chunks

    def test_single_valid_citation(self):
        """1. A single valid citation is extracted, validated, and normalized."""
        answer = "RepoLens AI is an interactive tool [README.md:1-12]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["file_path"], "README.md")
        self.assertEqual(valid[0]["start_line"], 1)
        self.assertEqual(valid[0]["end_line"], 12)
        self.assertEqual(len(invalid), 0)
        self.assertIn("[README.md:1-12]", cleaned)

    def test_multiple_citations_in_separate_brackets(self):
        """2. Multiple citations in separate brackets are both validated."""
        answer = "Documented in [README.md:1-12] and features at [README.md:45-60]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]["citation"], "[README.md:1-12]")
        self.assertEqual(valid[1]["citation"], "[README.md:45-60]")
        self.assertEqual(len(invalid), 0)

    def test_multiple_citations_inside_single_bracket_failing_example(self):
        """3. Multiple citations inside one bracket (exact failing user example) are parsed and validated."""
        answer = (
            "The main purpose of RepoLens AI is to provide repository analysis "
            "[README.md:1-12, README.md:30-40, README.md:75-85]."
        )
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        self.assertEqual(len(valid), 3, f"Expected 3 valid citations, got {len(valid)}: {valid}")
        citations = [v["citation"] for v in valid]
        self.assertIn("[README.md:1-12]", citations)
        self.assertIn("[README.md:30-40]", citations)
        self.assertIn("[README.md:75-85]", citations)
        self.assertEqual(len(invalid), 0)

        # Confirm the output text contains individual valid brackets
        self.assertIn("[README.md:1-12]", cleaned)
        self.assertIn("[README.md:30-40]", cleaned)
        self.assertIn("[README.md:75-85]", cleaned)
        self.assertNotIn("[README.md:1-12,", cleaned)

    def test_compound_citation_semicolon_delimited(self):
        """3b. Semicolon-delimited citations in a single bracket are parsed."""
        answer = "Components defined at [backend/app/main.py:10-25; backend/app/routes/chat.py:30-50]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.code_chunks)

        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]["file_path"], "backend/app/main.py")
        self.assertEqual(valid[1]["file_path"], "backend/app/routes/chat.py")
        self.assertEqual(len(invalid), 0)

    def test_compound_citation_shared_filename(self):
        """3c. Shared filename with comma-separated line ranges is expanded correctly."""
        answer = "Described in [README.md: 1-12, 30-40, 75-85]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        self.assertEqual(len(valid), 3)
        self.assertEqual([v["citation"] for v in valid], ["[README.md:1-12]", "[README.md:30-40]", "[README.md:75-85]"])
        self.assertEqual(len(invalid), 0)

    def test_compound_citation_with_invalid_file_or_out_of_bounds_lines(self):
        """4. Compound citation containing invalid files or out-of-bounds lines validates only true evidence."""
        answer = "Mixed [README.md:1-12, nonexistent_file.py:10-20, README.md:900-950]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        # Only README.md:1-12 is valid against chunks (1-50, 41-90)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["citation"], "[README.md:1-12]")

        # The nonexistent file and line 900-950 must be classified as invalid
        self.assertEqual(len(invalid), 2)
        self.assertIn("[nonexistent_file.py:10-20]", invalid)
        self.assertIn("[README.md:900-950]", invalid)

        # Invalid citations are stripped from the cleaned answer
        self.assertNotIn("nonexistent_file.py", cleaned)
        self.assertNotIn("900-950", cleaned)
        self.assertIn("[README.md:1-12]", cleaned)

    def test_malformed_citation_handling(self):
        """5. Malformed citation syntax is rejected and not fabricated into sources."""
        answer = "Evidence from [README.md:1-12, not_a_real_citation_syntax???]."
        cleaned, valid, invalid = extract_and_validate_citations(answer, self.readme_chunks)

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["citation"], "[README.md:1-12]")
        self.assertNotIn("not_a_real_citation_syntax", [v["citation"] for v in valid])

    def test_answer_with_no_valid_evidence_remains_ungrounded(self):
        """6. An answer with no valid evidence remains ungrounded with empty sources."""
        # Case A: Answer has a citation to a file not present in evidence chunks
        answer = "This claims something completely unsupported [unknown.py:10-20]."
        final_answer, sources, grounded = finalize_rag_response(
            raw_answer=answer,
            llm_context_chunks=self.readme_chunks,
            is_call_chain=False,
            question="How does unknown service work?",
            is_purpose=False,
        )

        self.assertFalse(grounded, "Answer with unverified citation must be ungrounded")
        self.assertEqual(sources, [])
        self.assertIn("Accurate line citations could not be established", final_answer)

        # Case B: Answer provided with completely empty evidence
        answer_no_ev = "General explanation with citation [README.md:1-12]."
        final_answer_no_ev, sources_no_ev, grounded_no_ev = finalize_rag_response(
            raw_answer=answer_no_ev,
            llm_context_chunks=[],
            is_call_chain=False,
            question="What is the purpose of this repository?",
            is_purpose=True,
        )
        self.assertFalse(grounded_no_ev, "Answer with empty evidence must remain ungrounded")
        self.assertEqual(sources_no_ev, [])

    def test_full_response_finalization_purpose_query(self):
        """7. Full response finalization path produces grounded=True and sources for compound citation."""
        answer = (
            "RepoLens AI is an interactive repository analysis tool designed to inspect, index, and query codebases "
            "[README.md:1-12, README.md:30-40, README.md:75-85]."
        )
        final_answer, sources, grounded = finalize_rag_response(
            raw_answer=answer,
            llm_context_chunks=self.readme_chunks,
            is_call_chain=False,
            question="What is the main purpose of this repository?",
            is_purpose=True,
        )

        self.assertTrue(grounded, "Expected grounded=True when all citations are verified in README evidence")
        self.assertEqual(len(sources), 3, f"Expected 3 sources, got {len(sources)}: {sources}")

        source_citations = [s["citation"] for s in sources]
        self.assertIn("[README.md:1-12]", source_citations)
        self.assertIn("[README.md:30-40]", source_citations)
        self.assertIn("[README.md:75-85]", source_citations)

        # Every source must have file_path, start_line, end_line, and non-empty snippet
        for s in sources:
            self.assertEqual(s["file_path"], "README.md")
            self.assertGreater(s["start_line"], 0)
            self.assertGreaterEqual(s["end_line"], s["start_line"])
            self.assertTrue(bool(s.get("snippet")), "Source snippet must be populated from evidence")

    def test_unrelated_bracketed_text_preserved(self):
        """8. Unrelated bracketed text with commas is preserved and not altered."""
        text = "Check the options [option A, option B, option C] before proceeding."
        cleaned, valid, invalid = extract_and_validate_citations(text, self.readme_chunks)

        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 0)
        self.assertEqual(cleaned, text)


if __name__ == "__main__":
    unittest.main()
