import os
import re
import logging
from typing import List, Dict, Any, Tuple

from app.ai.vector_store import get_vector_store
from app.ai.llm import get_llm_provider
from app.ai.prompts import (
    SYSTEM_RAG_PROMPT,
    format_rag_user_prompt
)
from app.schemas.chat import SourceCitation
from app.config import settings


logger = logging.getLogger("repolens.ai.rag")


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cpp", ".c", ".cs", ".go",
    ".rs", ".php", ".rb", ".sql"
}


UI_EXTENSIONS = {
    ".html", ".css", ".scss", ".svg"
}


KEYWORD_BOOST_MAP = [
    (
        "detection",
        [
            "detect",
            "detection",
            "detected",
            "vehicle damage",
            "damage detection",
            "image",
            "images",
            "uploaded image",
            "object detection",
            "prediction",
            "yolo",
            "model",
            "boxes",
            "class_ids",
            "class_counts",
            "annotated",
            "best.pt"
        ]
    ),

    (
        "auth",
        [
            "auth",
            "login",
            "logout",
            "signup",
            "bcrypt",
            "password",
            "session",
            "checkpw",
            "user",
            "authenticate"
        ]
    ),

    (
        "login",
        [
            "login",
            "bcrypt",
            "session",
            "password",
            "checkpw",
            "email"
        ]
    ),

    (
        "database",
        [
            "db",
            "database",
            "sql",
            "cursor",
            "query",
            "sqlite",
            "table",
            "execute"
        ]
    ),

    (
        "model",
        [
            "yolo",
            "cv2",
            "torch",
            "model",
            "predict",
            "predict_damage",
            "detect",
            "boxes",
            "class_ids",
            "class_counts",
            "plot",
            "best.pt",
            "onnx",
            "load_model",
            "pickle",
            "pkl"
        ]
    ),

    (
        "cost",
        [
            "cost",
            "estimate",
            "price",
            "insurance",
            "parts",
            "total"
        ]
    ),
]


def compute_chunk_hybrid_score(
    chunk: Dict[str, Any],
    query: str
) -> float:
    """
    Computes a hybrid relevance score using:
    - Semantic similarity
    - Exact phrase matching
    - Keyword matching
    - Filename matching
    - Symbol matching
    - Domain-specific technical terms
    - Code-file preference
    """

    query_lower = query.lower().strip()

    content_lower = (
        chunk.get("content", "")
        .lower()
    )

    file_path = (
        chunk.get("file_path", "")
        .lower()
    )

    symbol_name = (
        chunk.get("symbol_name") or ""
    ).lower()

    ext = os.path.splitext(file_path)[1].lower()

    # ---------------------------------------------------------
    # 1. Semantic score
    # ---------------------------------------------------------

    distance = float(
        chunk.get("distance", 1.0)
    )

    semantic_score = (
        1.0 / (1.0 + distance)
    )

    # ---------------------------------------------------------
    # 2. File type awareness
    # ---------------------------------------------------------

    file_type_bonus = 0.0

    if ext in CODE_EXTENSIONS:
        file_type_bonus += 0.20

    elif ext in UI_EXTENSIONS:
        file_type_bonus -= 0.10

    # ---------------------------------------------------------
    # 3. Query words
    # ---------------------------------------------------------

    query_words = [
        word
        for word in re.split(
            r"\W+",
            query_lower
        )
        if len(word) > 2
    ]

    # ---------------------------------------------------------
    # 4. Basic lexical matching
    # ---------------------------------------------------------

    content_matches = sum(
        1
        for word in query_words
        if word in content_lower
    )

    filename_matches = sum(
        1
        for word in query_words
        if word in file_path
    )

    symbol_matches = sum(
        1
        for word in query_words
        if word in symbol_name
    )

    lexical_score = (
        content_matches * 0.06
        + filename_matches * 0.15
        + symbol_matches * 0.15
    )

    # ---------------------------------------------------------
    # 5. Exact phrase matching
    # ---------------------------------------------------------

    exact_phrase_bonus = 0.0

    if query_lower in content_lower:
        exact_phrase_bonus += 0.80

    # ---------------------------------------------------------
    # 6. Important technical/code operations
    # ---------------------------------------------------------

    technical_terms = [
        "model(",
        "predict(",
        "detect(",
        ".boxes",
        "class_ids",
        "class_counts",
        "cv2.imwrite",
        ".plot(",
        "yolo",
        "best.pt",
        "load_model",
        "pickle",
        "pkl",
        "bcrypt.checkpw",
        "session",
        "sqlalchemy",
        "cursor.execute",
        "execute(",
        "commit(",
        "jwt",
    ]

    technical_matches = sum(
        1
        for term in technical_terms
        if term in query_lower
        and term in content_lower
    )

    technical_bonus = (
        technical_matches * 0.35
    )

    # ---------------------------------------------------------
    # 7. Domain-specific matching
    # ---------------------------------------------------------

    domain_bonus = 0.0

    for topic_key, terms in KEYWORD_BOOST_MAP:

        query_matches_topic = (
            topic_key in query_lower
            or any(
                term in query_lower
                for term in terms
            )
        )

        if not query_matches_topic:
            continue

        content_term_matches = sum(
            1
            for term in terms
            if term in content_lower
        )

        if content_term_matches > 0:
            domain_bonus += min(
                0.30,
                content_term_matches * 0.08
            )

        if ext in CODE_EXTENSIONS:
            domain_bonus += 0.05

    # ---------------------------------------------------------
    # 8. Special boost for detection-related code
    # ---------------------------------------------------------

    detection_terms = [
        "model(",
        "result[0].boxes",
        "detected_objects",
        "class_ids",
        "class_counts",
        "result[0].plot",
        "cv2.imwrite",
        "best.pt",
    ]

    detection_query = any(
        term in query_lower
        for term in [
            "vehicle damage",
            "damage detection",
            "damage detected",
            "detect damage",
            "detection",
            "yolo",
            "object detection",
            "detected",
        ]
    )

    if detection_query:

        detection_matches = sum(
            1
            for term in detection_terms
            if term in content_lower
        )

        if detection_matches > 0:
            domain_bonus += min(
                0.80,
                detection_matches * 0.15
            )

    # ---------------------------------------------------------
    # 9. Final hybrid score
    # ---------------------------------------------------------

    final_score = (
        (semantic_score * 0.30)
        + (lexical_score * 0.20)
        + exact_phrase_bonus
        + technical_bonus
        + file_type_bonus
        + domain_bonus
    )

    return final_score


class RAGEngine:

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm_provider = get_llm_provider()

    def ask_question(
        self,
        repository_id: str,
        question: str,
        top_k: int = 5
    ) -> Tuple[
        str,
        List[SourceCitation],
        bool
    ]:

        """
        Executes grounded RAG pipeline
        with hybrid retrieval and re-ranking.
        """

        logger.info(
            f"RAG Request - Repository ID: "
            f"{repository_id} | "
            f"Provider: {settings.LLM_PROVIDER} | "
            f"Question: '{question}'"
        )

        # -----------------------------------------------------
        # 1. Candidate Pool Retrieval
        # -----------------------------------------------------

        candidate_chunks = (
            self.vector_store.query_similar_chunks(
                repository_id,
                question,
                top_k=50
            )
        )

        logger.info(
            f"Retrieved {len(candidate_chunks)} "
            f"candidate chunks from ChromaDB "
            f"collection repo_"
            f"{repository_id.replace('-', '_')}"
        )

        if not candidate_chunks:

            logger.warning(
                f"No candidate chunks retrieved "
                f"for repository {repository_id}"
            )

            return (
                "I couldn't find enough evidence "
                "in the repository to answer "
                "this confidently.",
                [],
                False
            )

        # -----------------------------------------------------
        # 2. Hybrid Re-Ranking
        # -----------------------------------------------------

        scored_chunks = []

        for chunk in candidate_chunks:

            hybrid_score = (
                compute_chunk_hybrid_score(
                    chunk,
                    question
                )
            )

            chunk_copy = chunk.copy()
            chunk_copy["hybrid_score"] = hybrid_score

            scored_chunks.append(
                chunk_copy
            )

        # Sort highest score first
        scored_chunks.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        # -----------------------------------------------------
        # Select top unique chunks
        # -----------------------------------------------------

        top_selected_chunks = []
        seen_ids = set()

        for chunk in scored_chunks:

            chunk_key = (
                f"{chunk['file_path']}:"
                f"{chunk['start_line']}-"
                f"{chunk['end_line']}"
            )

            if chunk_key not in seen_ids:

                seen_ids.add(chunk_key)

                top_selected_chunks.append(
                    chunk
                )

            if len(top_selected_chunks) >= min(
                top_k,
                5
            ):
                break

        # -----------------------------------------------------
        # Log selected chunks
        # -----------------------------------------------------

        log_selected = [
            (
                f"{chunk['file_path']}:"
                f"L{chunk['start_line']}-"
                f"L{chunk['end_line']} "
                f"(score="
                f"{chunk['hybrid_score']:.3f})"
            )
            for chunk in top_selected_chunks
        ]

        logger.info(
            "Top selected chunks after "
            f"hybrid re-ranking: {log_selected}"
        )

        # -----------------------------------------------------
        # DEBUG: Print selected RAG context
        # -----------------------------------------------------

        print(
            "\n===== SELECTED RAG CONTEXT ====="
        )

        for chunk in top_selected_chunks:

            print(
                f"\n--- {chunk['file_path']}:"
                f"{chunk['start_line']}-"
                f"{chunk['end_line']} ---"
            )

            print(chunk["content"])

        print(
            "\n===== END SELECTED RAG CONTEXT =====\n"
        )

        # -----------------------------------------------------
        # 3. Select strongest evidence for LLM
        # -----------------------------------------------------

        llm_context_chunks = (
            top_selected_chunks[:1]
        )

        if not llm_context_chunks:

            return (
                "I couldn't find enough evidence "
                "in the repository to answer "
                "this confidently.",
                [],
                False
            )

        strongest_chunk = (
            llm_context_chunks[0]
        )

        logger.info(
            "Using strongest evidence chunk for LLM: "
            f"{strongest_chunk['file_path']}:"
            f"{strongest_chunk['start_line']}-"
            f"{strongest_chunk['end_line']}"
        )

        # -----------------------------------------------------
        # 4. Format safe RAG prompt
        # -----------------------------------------------------

        user_prompt = format_rag_user_prompt(
            question,
            llm_context_chunks
        )

        # -----------------------------------------------------
        # 5. Query LLM
        # -----------------------------------------------------

        answer = (
            self.llm_provider.generate_response(
                SYSTEM_RAG_PROMPT,
                user_prompt
            )
        )

        # -----------------------------------------------------
        # 6. Extract ONLY cited sources
        # -----------------------------------------------------

        sources = []
        seen_keys = set()

        for chunk in top_selected_chunks:

            file_path = chunk["file_path"]
            start_line = chunk["start_line"]
            end_line = chunk["end_line"]

            citation = (
                f"[{file_path}:"
                f"{start_line}-"
                f"{end_line}]"
            )

            key = (
                f"{file_path}:"
                f"{start_line}-"
                f"{end_line}"
            )

            # Only return this source if the LLM
            # actually cited it in the answer.
            if (
                citation in answer
                and key not in seen_keys
            ):

                seen_keys.add(key)

                sources.append(
                    SourceCitation(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        symbol_name=chunk.get(
                            "symbol_name"
                        ),
                        snippet="\n".join(
                            chunk["content"]
                            .splitlines()[:5]
                        )
                    )
                )

        logger.info(
            f"Final cited sources returned: "
            f"{len(sources)}"
        )

        # -----------------------------------------------------
        # 7. Grounding status
        # -----------------------------------------------------

        grounded = (
            "I couldn't find enough evidence"
            not in answer
        )

        return (
            answer,
            sources,
            grounded
        )