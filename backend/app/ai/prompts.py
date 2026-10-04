SYSTEM_RAG_PROMPT = """You are RepoLens AI, an expert codebase assistant.

Answer the user's question using ONLY the repository code provided
in the user message.

GROUNDING RULES:

1. Never use information that is not present in the provided code.
2. Every technical statement MUST include an inline citation.
3. Citation format MUST be exactly:
   [file_path:start_line-end_line]
4. Use the FILE and LINES metadata provided with each chunk.
5. Never invent files, functions, line numbers, dependencies, or behavior.
6. Prefer runtime implementation code over README files, training scripts,
   documentation, or unrelated helper code.
7. If the provided code contains the implementation that directly performs
   the requested operation, explain that implementation directly.
8. Do not replace the requested operation with a related operation.
9. If there is insufficient evidence, respond exactly:

I couldn't find enough evidence in the repository to answer this confidently.

ANSWER STYLE:

- Be concise and direct.
- Explain what the code actually does.
- Use the most relevant implementation chunk.
- Every technical claim must have a citation.
- Do not create a Sources section.
"""


def format_rag_user_prompt(
    question: str,
    context_chunks: list
) -> str:
    """Format repository evidence for the LLM."""

    formatted_context = []

    for idx, chunk in enumerate(context_chunks, 1):

        file_path = chunk.get("file_path", "unknown")
        start_line = chunk.get("start_line", 1)
        end_line = chunk.get("end_line", 1)

        symbol = (
            chunk.get("symbol_name")
            or "Global Scope"
        )

        content = chunk.get("content", "")

        lines = content.splitlines()

        if len(lines) > 60:
            content = (
                "\n".join(lines[:60])
                + f"\n... [truncated {len(lines) - 60} lines]"
            )

        chunk_str = (
            f"--- EVIDENCE {idx} ---\n"
            f"FILE: {file_path}\n"
            f"LINES: {start_line}-{end_line}\n"
            f"SYMBOL: {symbol}\n"
            f"CODE:\n"
            f"{content}\n"
            f"--- END EVIDENCE {idx} ---\n"
        )

        formatted_context.append(chunk_str)

    context_block = (
        "\n".join(formatted_context)
        if formatted_context
        else "No repository code was found."
    )

    user_prompt = f"""
USER QUESTION:
{question}

REPOSITORY CODE EVIDENCE:
{context_block}

IMPORTANT:

First determine which evidence chunk directly performs the operation
asked about.

The answer must be based on the most direct implementation.

For example, if the question asks how vehicle damage is detected,
code containing:

- an uploaded image
- model(image_path)
- result[0].boxes
- class_ids
- class_counts
- result[0].plot()
- cv2.imwrite()

is direct detection evidence.

Code that only:

- trains the YOLO model
- calculates repair prices
- predicts insurance coverage
- describes the project in README

is NOT the primary answer when runtime detection code is available.

If direct implementation evidence exists, explain THAT code.

Do not answer using a related operation merely because it appears
in another evidence chunk.

CITATION REQUIREMENT:

Every technical statement must have an inline citation.

Use exactly:

[file_path:start_line-end_line]

Only use file paths and line numbers shown in the evidence.

Do not invent citations.

Now answer the user's question concisely.
"""

    return user_prompt