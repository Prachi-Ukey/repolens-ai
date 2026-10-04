import logging
import re
from typing import Any, Dict, List, Tuple

from .llm import LLMProvider


logger = logging.getLogger(__name__)


# ============================================================
# UI / FILE EXTENSIONS
# ============================================================

UI_EXTENSIONS = {
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".jsx",
    ".tsx",
}


# ============================================================
# CITATION VALIDATION
# ============================================================

def extract_and_validate_citations(
    answer: str,
    evidence_chunks: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Extract citations from the LLM answer and validate them
    against the repository evidence supplied to the LLM.
    """

    citation_pattern = r"\[([^\[\]:]+):(\d+)-(\d+)\]"

    matches = re.findall(
        citation_pattern,
        answer
    )

    evidence_map: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in evidence_chunks:

        file_path = chunk.get(
            "file_path",
            ""
        )

        start_line = int(
            chunk.get(
                "start_line",
                1
            )
        )

        end_line = int(
            chunk.get(
                "end_line",
                start_line
            )
        )

        evidence_map.setdefault(
            file_path,
            []
        ).append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "chunk": chunk
            }
        )

    valid_citations = []
    invalid_citations = []

    for file_path, cited_start, cited_end in matches:

        cited_start = int(cited_start)
        cited_end = int(cited_end)

        valid = False
        matched_chunk = None

        for evidence in evidence_map.get(
            file_path,
            []
        ):

            evidence_start = evidence[
                "start_line"
            ]

            evidence_end = evidence[
                "end_line"
            ]

            if (
                cited_start >= evidence_start
                and cited_end <= evidence_end
            ):

                valid = True

                matched_chunk = evidence[
                    "chunk"
                ]

                break

        citation_text = (
            f"[{file_path}:"
            f"{cited_start}-"
            f"{cited_end}]"
        )

        if valid:

            valid_citations.append(
                {
                    "citation": citation_text,
                    "file_path": file_path,
                    "start_line": cited_start,
                    "end_line": cited_end,
                    "chunk": matched_chunk
                }
            )

        else:

            invalid_citations.append(
                citation_text
            )

    cleaned_answer = answer

    for citation in invalid_citations:

        cleaned_answer = (
            cleaned_answer.replace(
                citation,
                ""
            )
        )

    return (
        cleaned_answer.strip(),
        valid_citations,
        invalid_citations
    )


# ============================================================
# RAG ENGINE
# ============================================================

class RAGEngine:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        vector_store,
        llm_provider: LLMProvider
    ):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    # ========================================================
    # QUESTION CLASSIFICATION
    # ========================================================

    def classify_question(
        self,
        question: str
    ) -> str:
        """
        Classify a user question to guide retrieval strategy.
        """

        question_lower = question.lower().strip()

        runtime_terms = {
            "how does",
            "how is",
            "how are",
            "what happens",
            "what happens after",
            "process",
            "processing",
            "runtime",
            "detected",
            "detect",
            "prediction",
            "predict",
            "execute",
            "execution",
        }

        function_terms = {
            "function",
            "method",
            "class",
            "variable",
            "where is",
            "which function",
            "which method",
            "defined",
            "implemented",
        }

        architecture_terms = {
            "database",
            "technology",
            "technologies",
            "framework",
            "library",
            "libraries",
            "stack",
            "architecture",
            "backend",
            "frontend",
            "api",
        }

        structure_terms = {
            "folder",
            "directory",
            "file",
            "files",
            "structure",
            "repository structure",
            "project structure",
        }

        if any(
            term in question_lower
            for term in runtime_terms
        ):
            return "runtime"

        if any(
            term in question_lower
            for term in function_terms
        ):
            return "code"

        if any(
            term in question_lower
            for term in architecture_terms
        ):
            return "architecture"

        if any(
            term in question_lower
            for term in structure_terms
        ):
            return "structure"

        return "general"

    # ========================================================
    # BUILD RETRIEVAL QUERIES
    # ========================================================

    def build_retrieval_queries(
        self,
        question: str
    ) -> List[str]:
        """
        Build multiple retrieval queries for better code retrieval.

        The first query preserves the user's natural language.
        Additional queries add code-specific and runtime-specific
        terminology for implementation questions.
        """

        question_lower = question.lower()

        implementation_terms = {
            "how",
            "detect",
            "detection",
            "detected",
            "implement",
            "implementation",
            "process",
            "work",
            "works",
            "calculate",
            "calculated",
            "predict",
            "prediction",
            "classify",
            "classification",
            "upload",
            "login",
            "authenticate",
            "store",
            "save",
            "database",
            "model",
            "train",
            "training",
        }

        is_implementation_question = any(
            term in question_lower
            for term in implementation_terms
        )

        queries = []

        # Query 1: Original natural language

        queries.append(
            question
        )

        if is_implementation_question:

            # Query 2: General code implementation

            queries.append(
                question
                + " source code implementation "
                + "function logic runtime code"
            )

            # Query 3: Code-specific retrieval

            queries.append(
                question
                + " code function method variable "
                + "model input output execution"
            )

            # Query 4: Runtime-specific retrieval

            queries.append(
                question
                + " runtime source code "
                + "function call result objects "
                + "processing logic"
            )

        # Remove duplicate queries.

        unique_queries = []

        seen = set()

        for query in queries:

            normalized = query.strip().lower()

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            unique_queries.append(
                query.strip()
            )

        return unique_queries

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(
        self,
        repo_id: str,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:

        logger.info(
            "RAG question | repo=%s | question=%s",
            repo_id,
            question
        )

        # ====================================================
        # 1. CLASSIFY QUESTION
        # ====================================================

        question_type = self.classify_question(
            question
        )

        logger.info(
            "Question type: %s",
            question_type
        )

        # ====================================================
        # 2. BUILD RETRIEVAL QUERIES
        # ====================================================

        retrieval_queries = (
            self.build_retrieval_queries(
                question
            )
        )

        print(
            "\n========== VECTOR SEARCH QUERIES =========="
        )

        for i, query in enumerate(
            retrieval_queries,
            start=1
        ):

            print(
                f"{i}. {query}"
            )

        print(
            "============================================\n"
        )

        # ====================================================
        # 3. SEARCH VECTOR STORE
        # ====================================================

        merged_results: Dict[
            str,
            Dict[str, Any]
        ] = {}

        for query in retrieval_queries:

            search_results = (
                self.vector_store.query_similar_chunks(
                    repository_id=repo_id,
                    query_text=query,
                    top_k=90
                )
            )

            for chunk in search_results:

                chunk_key = (
                    f"{chunk.get('file_path', '')}:"
                    f"{chunk.get('start_line', 1)}-"
                    f"{chunk.get('end_line', 1)}"
                )

                distance = float(
                    chunk.get(
                        "distance",
                        1.0
                    )
                )

                if chunk_key not in merged_results:

                    chunk_copy = dict(
                        chunk
                    )

                    chunk_copy[
                        "_best_distance"
                    ] = distance

                    chunk_copy[
                        "_query_matches"
                    ] = 1

                    merged_results[
                        chunk_key
                    ] = chunk_copy

                else:

                    existing = (
                        merged_results[
                            chunk_key
                        ]
                    )

                    existing_distance = float(
                        existing.get(
                            "_best_distance",
                            1.0
                        )
                    )

                    if distance < existing_distance:

                        existing[
                            "_best_distance"
                        ] = distance

                        existing[
                            "distance"
                        ] = distance

                    existing[
                        "_query_matches"
                    ] = existing.get(
                        "_query_matches",
                        1
                    ) + 1

        search_results = list(
            merged_results.values()
        )

        # Make sure the best distance is used.

        for chunk in search_results:

            chunk["distance"] = chunk.get(
                "_best_distance",
                chunk.get(
                    "distance",
                    1.0
                )
            )

        if not search_results:

            return {
                "answer": (
                    "I couldn't find enough evidence in the "
                    "repository to answer this confidently."
                ),
                "sources": [],
                "grounded": False
            }

        # ====================================================
        # 4. DISPLAY RETRIEVED CHUNKS
        # ====================================================

        print(
            "\n========== MERGED RETRIEVED CHUNKS =========="
        )

        for i, chunk in enumerate(
            sorted(
                search_results,
                key=lambda item: item.get(
                    "distance",
                    1.0
                )
            )[:30],
            start=1
        ):

            print(
                f"{i}. FILE: {chunk.get('file_path')} | "
                f"LINES: {chunk.get('start_line')}-"
                f"{chunk.get('end_line')} | "
                f"DISTANCE: {chunk.get('distance')} | "
                f"QUERY MATCHES: "
                f"{chunk.get('_query_matches', 1)} | "
                f"SYMBOL: {chunk.get('symbol_name')}"
            )

        print(
            "=============================================\n"
        )

        # ====================================================
        # 5. QUERY KEYWORDS
        # ====================================================

        question_lower = question.lower()

        query_keywords = set(
            re.findall(
                r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
                question_lower
            )
        )

        # ====================================================
        # 6. SCORE RETRIEVED CHUNKS
        # ====================================================

        scored_chunks = []

        for chunk in search_results:

            distance = float(
                chunk.get(
                    "distance",
                    1.0
                )
            )

            semantic_score = 1.0 / (
                1.0 + distance
            )

            file_path = (
                chunk.get(
                    "file_path",
                    ""
                )
                or ""
            )

            content = (
                chunk.get(
                    "content",
                    chunk.get(
                        "code",
                        ""
                    )
                )
                or ""
            )

            symbol_name = (
                chunk.get(
                    "symbol_name",
                    ""
                )
                or ""
            )

            searchable_text = (
                f"{file_path} "
                f"{symbol_name} "
                f"{content}"
            ).lower()

            # ------------------------------------------------
            # Keyword matching
            # ------------------------------------------------

            keyword_matches = 0

            for keyword in query_keywords:

                if len(keyword) < 3:
                    continue

                if keyword in searchable_text:
                    keyword_matches += 1

            keyword_score = min(
                keyword_matches / max(
                    len(query_keywords),
                    1
                ),
                1.0
            )

            # ------------------------------------------------
            # Implementation boost
            # ------------------------------------------------

            implementation_score = 0.0

            if file_path.lower().endswith(
                (
                    ".py",
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                    ".java",
                    ".cpp",
                    ".c"
                )
            ):

                implementation_score += 0.15

            # README/docs should not dominate implementation.

            if file_path.lower().endswith(
                (
                    "readme.md",
                    "readme.txt",
                    ".md"
                )
            ):

                implementation_score -= 0.15

            # Runtime source files get additional boost.

            if file_path.lower() in {
                "app.py",
                "main.py",
                "server.py",
                "api.py",
            }:

                implementation_score += 0.15

            # ------------------------------------------------
            # Multi-query match boost
            # ------------------------------------------------

            query_match_count = int(
                chunk.get(
                    "_query_matches",
                    1
                )
            )

            query_match_score = min(
                query_match_count * 0.05,
                0.15
            )

            # ------------------------------------------------
            # Symbol boost
            # ------------------------------------------------

            symbol_score = 0.0

            if symbol_name:

                symbol_lower = symbol_name.lower()

                for keyword in query_keywords:

                    if (
                        len(keyword) >= 3
                        and keyword in symbol_lower
                    ):

                        symbol_score += 0.10

            # ------------------------------------------------
            # Runtime / execution logic boost
            # ------------------------------------------------

            runtime_score = 0.0

            runtime_intent_keywords = (
                "how is",
                "how does",
                "detected",
                "detect",
                "implemented",
                "implementation",
                "runtime",
                "execute",
                "processed",
                "processing",
            )

            runtime_intent = any(
                phrase in question_lower
                for phrase in runtime_intent_keywords
            )

            if runtime_intent:

                runtime_patterns = [
                    "model(",
                    "model (",
                    "result[",
                    "result [",
                    ".boxes",
                    "class_ids",
                    "class_counts",
                    "image_path",
                    "predict(",
                    "prediction",
                    "detect(",
                    "inference",
                ]

                runtime_hits = 0

                for pattern in runtime_patterns:

                    if pattern in content.lower():

                        runtime_hits += 1

                runtime_score = min(
                    runtime_hits * 0.08,
                    0.40
                )

            # ------------------------------------------------
            # Final hybrid score
            # ------------------------------------------------

            hybrid_score = (
                semantic_score
                + keyword_score * 0.35
                + implementation_score
                + symbol_score
                + query_match_score
                + runtime_score
            )

            chunk["semantic_score"] = (
                semantic_score
            )

            chunk["keyword_score"] = (
                keyword_score
            )

            chunk["implementation_score"] = (
                implementation_score
            )

            chunk["query_match_score"] = (
                query_match_score
            )

            chunk["symbol_score"] = (
                symbol_score
            )

            chunk["runtime_score"] = (
                runtime_score
            )

            chunk["hybrid_score"] = (
                hybrid_score
            )

            scored_chunks.append(
                chunk
            )

        # ====================================================
        # 7. SORT BY HYBRID SCORE
        # ====================================================

        scored_chunks.sort(
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True
        )

        print(
            "\n========== HYBRID SCORES =========="
        )

        for i, chunk in enumerate(
            scored_chunks[:15],
            start=1
        ):

            print(
                f"{i}. "
                f"{chunk.get('file_path')} | "
                f"LINES "
                f"{chunk.get('start_line')}-"
                f"{chunk.get('end_line')} | "
                f"SEMANTIC "
                f"{chunk.get('semantic_score'):.3f} | "
                f"KEYWORD "
                f"{chunk.get('keyword_score'):.3f} | "
                f"IMPL "
                f"{chunk.get('implementation_score'):.3f} | "
                f"QUERY MATCH "
                f"{chunk.get('query_match_score'):.3f} | "
                f"RUNTIME "
                f"{chunk.get('runtime_score', 0.0):.3f} | "
                f"FINAL "
                f"{chunk.get('hybrid_score'):.3f}"
            )

        print(
            "==================================\n"
        )

        # ====================================================
        # 8. SELECT RELEVANT UNIQUE CHUNKS
        # ====================================================

        top_selected_chunks = []

        seen_ids = set()

        RELEVANCE_THRESHOLD = 0.45

        for chunk in scored_chunks:

            if (
                chunk["hybrid_score"]
                < RELEVANCE_THRESHOLD
            ):
                continue

            chunk_key = (
                f"{chunk.get('file_path', '')}:"
                f"{chunk.get('start_line', 1)}-"
                f"{chunk.get('end_line', 1)}"
            )

            if chunk_key in seen_ids:
                continue

            seen_ids.add(
                chunk_key
            )

            top_selected_chunks.append(
                chunk
            )

            if len(
                top_selected_chunks
            ) >= min(
                top_k,
                5
            ):

                break

        # ====================================================
        # 9. CHECK USEFUL EVIDENCE
        # ====================================================

        if not top_selected_chunks:

            logger.warning(
                "No relevant evidence found | repo=%s | question=%s",
                repo_id,
                question
            )

            return {
                "answer": (
                    "I couldn't find enough evidence in the "
                    "repository to answer this confidently."
                ),
                "sources": [],
                "grounded": False
            }

        # ====================================================
        # 10. LOG SELECTED CHUNKS
        # ====================================================

        logger.info(
            "Selected %s RAG chunks",
            len(top_selected_chunks)
        )

        for chunk in top_selected_chunks:

            logger.info(
                "RAG chunk | file=%s | lines=%s-%s | score=%.4f",
                chunk.get("file_path"),
                chunk.get("start_line"),
                chunk.get("end_line"),
                chunk.get("hybrid_score", 0)
            )

        # ====================================================
        # 11. PREPARE LLM CONTEXT
        # ====================================================

        # Currently send only the highest-ranked chunk.

        llm_context_chunks = (
            top_selected_chunks[:1]
        )

        evidence_text_parts = []

        for chunk in llm_context_chunks:

            file_path = chunk.get(
                "file_path",
                "unknown"
            )

            start_line = chunk.get(
                "start_line",
                1
            )

            end_line = chunk.get(
                "end_line",
                start_line
            )

            symbol = chunk.get(
                "symbol_name",
                ""
            )

            code = chunk.get(
                "content",
                chunk.get(
                    "code",
                    ""
                )
            )

            # Limit context size.

            code_lines = code.splitlines()

            if len(code_lines) > 40:

                code = "\n".join(
                    code_lines[:40]
                )

            evidence_text_parts.append(
                f"""
FILE: {file_path}
LINES: {start_line}-{end_line}
SYMBOL: {symbol}

CODE:
{code}
"""
            )

        evidence_text = "\n".join(
            evidence_text_parts
        )

        # ====================================================
        # 12. BUILD USER PROMPT
        # ====================================================

        user_prompt = f"""
Question:
{question}

Repository evidence:
{evidence_text}

Use only this evidence.
Every technical claim must include a citation.
"""

        # ====================================================
        # 13. DEBUGGING OUTPUT
        # ====================================================

        print(
            "\n========== RAG EVIDENCE SENT TO LLM =========="
        )

        print(
            evidence_text
        )

        print(
            "==============================================\n"
        )

        print(
            "\n========== FULL USER PROMPT SENT TO LLM =========="
        )

        print(
            user_prompt
        )

        print(
            "===================================================\n"
        )

        # ====================================================
        # 14. GENERATE LLM RESPONSE
        # ====================================================

        answer = (
            self.llm_provider.generate_response(
                SYSTEM_RAG_PROMPT,
                user_prompt
            )
        )

        # ====================================================
        # 15. VALIDATE LLM CITATIONS
        # ====================================================

        (
            answer,
            valid_citations,
            invalid_citations
        ) = extract_and_validate_citations(
            answer,
            llm_context_chunks
        )

        if invalid_citations:

            logger.warning(
                "Invalid LLM citations detected: "
                f"{invalid_citations}"
            )

        # ====================================================
        # 16. EXTRACT VERIFIED SOURCES
        # ====================================================

        sources = []

        seen_sources = set()

        for citation in valid_citations:

            source_key = (
                f"{citation['file_path']}:"
                f"{citation['start_line']}-"
                f"{citation['end_line']}"
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            sources.append(
                {
                    "file_path": citation["file_path"],
                    "start_line": citation["start_line"],
                    "end_line": citation["end_line"],
                    "citation": citation["citation"],
                    "snippet": citation["chunk"].get(
                        "content",
                        citation["chunk"].get(
                            "code",
                            ""
                        )
                    )
                }
            )

        # ====================================================
        # 17. DETERMINE GROUNDING STATUS
        # ====================================================

        grounded = (
            len(valid_citations) > 0
            and len(invalid_citations) == 0
            and (
                "I couldn't find enough evidence"
                not in answer
            )
        )

        # ====================================================
        # 18. RETURN RESPONSE
        # ====================================================

        return {
            "answer": answer,
            "sources": sources,
            "grounded": grounded
        }


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_RAG_PROMPT = """
You answer questions about source code using the provided repository evidence.

Read the code carefully and answer the user's question directly.

Rules:
- Use only the provided repository evidence.
- Explain what the code actually does.
- Do not say the evidence is insufficient if the code contains enough information to answer.
- Do not invent details.
- Include a citation for the technical claim.
- Citation format: [file_path:start_line-end_line]
- Use only the file path and line range provided in the evidence.
- Never create a Sources section.
- Answer in one or two short sentences.
"""
