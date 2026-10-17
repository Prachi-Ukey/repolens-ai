import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import ast
import logging

logger = logging.getLogger("repolens.github.parser")


# ============================================================
# DIRECTORIES TO IGNORE COMPLETELY
# ============================================================

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "pycache",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    "venv",
    "env",
    "target",
    "vendor",
    ".idea",
    ".vscode",
    "coverage",
    ".next",
    ".nuxt",
    ".output",
    "bin",
    "obj",
    "out",
    ".cargo",
    ".gradle",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".turbo",
    ".parcel-cache",
    ".serverless",
    ".terraform",
}


# ============================================================
# DEPENDENCY LOCKFILES TO IGNORE COMPLETELY
# ============================================================

LOCK_FILES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "gemfile.lock",
    "composer.lock",
    "poetry.lock",
    "pipfile.lock",
    "packages.lock.json",
    "bun.lockb",
    "flake.lock",
}


# ============================================================
# FILE EXTENSIONS TO IGNORE
# ============================================================

IGNORE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".iso",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".pyc",
    ".pyo",
    ".pyd",
    ".class",
    ".o",
    ".obj",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".log",
    ".ds_store",
    ".lock",
    ".lockb",
    ".map",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
    ".hdf5",
    ".weights",
}


# ============================================================
# FILES TO IGNORE
# ============================================================

IGNORE_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    "id_rsa",
    "id_rsa.pub",
    "credentials.json",
    "service-account.json",
}


# ============================================================
# LANGUAGE DETECTION
# ============================================================

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sh": "bash",
    ".dockerfile": "dockerfile",
}


# ============================================================
# PATH FILTERING
# ============================================================

def is_ignored_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")

    for part in parts[:-1]:
        part_lower = part.lower()
        if part_lower in IGNORE_DIRS or part.startswith("."):
            return True

    filename = parts[-1].lower()

    if filename in IGNORE_FILES:
        return True

    if filename in LOCK_FILES:
        return True

    if filename.endswith(".lock") or filename.endswith(".lockb"):
        return True

    # Minified / generated bundle files / maps
    if (
        filename.endswith(".min.js")
        or filename.endswith(".min.css")
        or filename.endswith(".map")
        or filename.endswith(".bundle.js")
        or filename.endswith(".chunk.js")
    ):
        return True

    ext = os.path.splitext(filename)[1].lower()

    if ext in IGNORE_EXTENSIONS:
        return True

    return False


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(file_path: str) -> str:
    filename = os.path.basename(file_path).lower()

    if filename == "dockerfile":
        return "dockerfile"

    ext = os.path.splitext(filename)[1].lower()

    return EXTENSION_LANGUAGE_MAP.get(ext, "text")


# ============================================================
# SHA-256
# ============================================================

def compute_sha256(content: str) -> str:
    return hashlib.sha256(
        content.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


# ============================================================
# SYMBOL EXTRACTION FOR NON-PYTHON LANGUAGES
# ============================================================

def extract_symbol_name(
    line: str,
    language: str,
) -> Optional[str]:
    """
    Try to extract a function/class name from a source line.
    """

    line_clean = line.strip()

    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    if language == "python":

        match = re.match(
            r"^(?:async\s+)?def\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)",
            line_clean,
        )

        if match:
            return match.group(1) or match.group(2)

    # --------------------------------------------------------
    # JAVASCRIPT / TYPESCRIPT
    # --------------------------------------------------------

    elif language in [
        "javascript",
        "typescript",
    ]:

        match = re.match(
            r"^(?:export\s+)?"
            r"(?:async\s+)?"
            r"function\s+([a-zA-Z0-9_]+)"
            r"|(?:export\s+)?class\s+([a-zA-Z0-9_]+)"
            r"|(?:const|let|var)\s+([a-zA-Z0-9_]+)"
            r"\s*=\s*(?:async\s*)?\(",
            line_clean,
        )

        if match:
            return (
                match.group(1)
                or match.group(2)
                or match.group(3)
            )

    # --------------------------------------------------------
    # JAVA / C / C++ / C#
    # --------------------------------------------------------

    elif language in [
        "java",
        "csharp",
        "cpp",
        "c",
    ]:

        match = re.match(
            r"^(?:public|private|protected|static|\s)*"
            r"(?:class|interface|enum|void|int|string|boolean|auto)"
            r"\s+([a-zA-Z0-9_]+)",
            line_clean,
        )

        if match:
            return match.group(1)

    # --------------------------------------------------------
    # GO
    # --------------------------------------------------------

    elif language == "go":

        match = re.match(
            r"^func\s+(?:\([^)]*\)\s+)?([a-zA-Z0-9_]+)"
            r"|type\s+([a-zA-Z0-9_]+)\s+struct",
            line_clean,
        )

        if match:
            return match.group(1) or match.group(2)

    # --------------------------------------------------------
    # RUST
    # --------------------------------------------------------

    elif language == "rust":

        match = re.match(
            r"^(?:pub\s+)?fn\s+([a-zA-Z0-9_]+)"
            r"|^(?:pub\s+)?struct\s+([a-zA-Z0-9_]+)",
            line_clean,
        )

        if match:
            return match.group(1) or match.group(2)

    return None


# ============================================================
# PYTHON AST METADATA
# ============================================================

def extract_python_metadata(content: str):
    """
    Extract:

    - defined symbols
    - called symbols
    - imported symbols

    from Python source using AST.
    """

    defined = []
    called = []
    imported = []

    try:

        tree = ast.parse(content)

        class Visitor(ast.NodeVisitor):

            def visit_FunctionDef(self, node):

                defined.append(
                    (
                        node.name,
                        node.lineno,
                    )
                )

                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):

                defined.append(
                    (
                        node.name,
                        node.lineno,
                    )
                )

                self.generic_visit(node)

            def visit_ClassDef(self, node):

                defined.append(
                    (
                        node.name,
                        node.lineno,
                    )
                )

                self.generic_visit(node)

            def visit_Call(self, node):

                if isinstance(
                    node.func,
                    ast.Name,
                ):

                    called.append(
                        (
                            node.func.id,
                            node.lineno,
                        )
                    )

                elif isinstance(
                    node.func,
                    ast.Attribute,
                ):

                    called.append(
                        (
                            node.func.attr,
                            node.lineno,
                        )
                    )

                self.generic_visit(node)

            def visit_Import(self, node):

                for n in node.names:

                    imported.append(
                        (
                            n.name,
                            node.lineno,
                        )
                    )

                self.generic_visit(node)

            def visit_ImportFrom(self, node):

                for n in node.names:

                    imported.append(
                        (
                            n.name,
                            node.lineno,
                        )
                    )

                self.generic_visit(node)

        Visitor().visit(tree)

    except Exception as e:

        logger.warning(
            "Python metadata extraction failed: %s",
            e,
        )

    return (
        defined,
        called,
        imported,
    )


# ============================================================
# GET TOP-LEVEL PYTHON AST NODES
# ============================================================

def _get_python_top_level_nodes(
    content: str,
):
    """
    Return top-level Python functions/classes with
    accurate source ranges.

    Decorators are included in the function/class chunk.
    """

    try:

        tree = ast.parse(content)

        nodes = []

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):

                # ------------------------------------------------
                # Include decorators in the semantic chunk.
                # ------------------------------------------------

                decorator_list = getattr(
                    node,
                    "decorator_list",
                    [],
                )

                if decorator_list:

                    start_line = min(
                        getattr(
                            decorator,
                            "lineno",
                            node.lineno,
                        )
                        for decorator in decorator_list
                    )

                else:

                    start_line = getattr(
                        node,
                        "lineno",
                        None,
                    )

                end_line = getattr(
                    node,
                    "end_lineno",
                    None,
                )

                if (
                    start_line is not None
                    and end_line is not None
                ):

                    nodes.append(
                        {
                            "node": node,
                            "start_line": start_line,
                            "end_line": end_line,
                            "symbol_name": getattr(
                                node,
                                "name",
                                None,
                            ),
                            "chunk_type": (
                                "class"
                                if isinstance(
                                    node,
                                    ast.ClassDef,
                                )
                                else "function"
                            ),
                        }
                    )

        return nodes

    except SyntaxError as e:

        logger.warning(
            "Python AST parsing failed during semantic chunking: %s",
            e,
        )

        return []

    except Exception as e:

        logger.warning(
            "Unexpected AST semantic chunking error: %s",
            e,
        )

        return []


# ============================================================
# BUILD PYTHON SEMANTIC RANGES
# ============================================================

def _build_python_semantic_ranges(
    content: str,
    max_function_lines: int = 500,
    fallback_chunk_lines: int = 50,
):
    """
    Build semantic source ranges for Python code.

    Normal top-level functions/classes become individual chunks.

    Very large functions/classes are split into smaller sections.

    Module-level code is grouped separately.
    """

    lines = content.splitlines()

    total_lines = len(lines)

    if total_lines == 0:
        return []

    nodes = _get_python_top_level_nodes(
        content
    )

    ranges = []

    semantic_nodes = []

    # ========================================================
    # FUNCTIONS / CLASSES
    # ========================================================

    for item in nodes:

        start = item["start_line"]
        end = item["end_line"]

        semantic_nodes.append(item)

        node_line_count = (
            end - start + 1
        )

        # ----------------------------------------------------
        # NORMAL FUNCTION / CLASS
        # ----------------------------------------------------

        if node_line_count <= max_function_lines:

            ranges.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "symbol_name": item[
                        "symbol_name"
                    ],
                    "chunk_type": item[
                        "chunk_type"
                    ],
                    "parent_symbol": None,
                }
            )

        # ----------------------------------------------------
        # VERY LARGE FUNCTION / CLASS
        # ----------------------------------------------------

        else:

            current = start

            while current <= end:

                section_end = min(
                    current
                    + fallback_chunk_lines
                    - 1,
                    end,
                )

                ranges.append(
                    {
                        "start_line": current,
                        "end_line": section_end,
                        "symbol_name": item[
                            "symbol_name"
                        ],
                        "chunk_type": (
                            "function_section"
                            if item[
                                "chunk_type"
                            ]
                            == "function"
                            else "class_section"
                        ),
                        "parent_symbol": item[
                            "symbol_name"
                        ],
                    }
                )

                current = (
                    section_end + 1
                )

    # ========================================================
    # MODULE-LEVEL CODE
    # ========================================================

    occupied = []

    for item in semantic_nodes:

        occupied.append(
            (
                item["start_line"],
                item["end_line"],
            )
        )

    def is_inside_semantic_node(
        line_number: int,
    ) -> bool:

        for start, end in occupied:

            if (
                start
                <= line_number
                <= end
            ):
                return True

        return False

    module_lines = []

    for line_number in range(
        1,
        total_lines + 1,
    ):

        if not is_inside_semantic_node(
            line_number
        ):

            module_lines.append(
                line_number
            )

    # ========================================================
    # GROUP CONTIGUOUS MODULE LINES
    # ========================================================

    if module_lines:

        groups = []

        current_group = [
            module_lines[0]
        ]

        for line_number in module_lines[1:]:

            if (
                line_number
                == current_group[-1] + 1
            ):

                current_group.append(
                    line_number
                )

            else:

                groups.append(
                    current_group
                )

                current_group = [
                    line_number
                ]

        groups.append(current_group)

        # ----------------------------------------------------
        # CREATE MODULE CHUNKS
        # ----------------------------------------------------

        for group in groups:

            group_start = group[0]
            group_end = group[-1]

            group_content = "\n".join(
                lines[
                    group_start - 1 : group_end
                ]
            )

            # Skip whitespace-only regions.
            if not group_content.strip():
                continue

            # ------------------------------------------------
            # SMALL MODULE REGION
            # ------------------------------------------------

            if (
                len(group)
                <= fallback_chunk_lines
            ):

                ranges.append(
                    {
                        "start_line": group_start,
                        "end_line": group_end,
                        "symbol_name": None,
                        "chunk_type": "module",
                        "parent_symbol": None,
                    }
                )

            # ------------------------------------------------
            # LARGE MODULE REGION
            # ------------------------------------------------

            else:

                current = group_start

                while current <= group_end:

                    section_end = min(
                        current
                        + fallback_chunk_lines
                        - 1,
                        group_end,
                    )

                    section_content = "\n".join(
                        lines[
                            current - 1 : section_end
                        ]
                    )

                    if section_content.strip():

                        ranges.append(
                            {
                                "start_line": current,
                                "end_line": section_end,
                                "symbol_name": None,
                                "chunk_type": "module",
                                "parent_symbol": None,
                            }
                        )

                    current = (
                        section_end + 1
                    )

    # ========================================================
    # SORT BY SOURCE POSITION
    # ========================================================

    ranges.sort(
        key=lambda item: (
            item["start_line"],
            item["end_line"],
        )
    )

    return ranges


# ============================================================
# PARSE FILE INTO CHUNKS
# ============================================================

def parse_file_into_chunks(
    file_path: str,
    content: str,
    language: str,
    target_lines_per_chunk: int = 50,
    overlap_lines: int = 10,
) -> List[Dict[str, Any]]:
    """
    Parse source code into chunks.

    Python:
        AST-aware semantic chunking.

    Other languages:
        Fixed-size sliding-window chunking.
    """

    lines = content.splitlines()

    total_lines = len(lines)

    if total_lines == 0:
        return []

    chunks = []

    # ========================================================
    # PYTHON
    # ========================================================

    if language == "python":

        defined, called, imported = (
            extract_python_metadata(content)
        )

        semantic_ranges = (
            _build_python_semantic_ranges(
                content,
                max_function_lines=500,
                fallback_chunk_lines=target_lines_per_chunk,
            )
        )

        for semantic_range in semantic_ranges:

            start_line = (
                semantic_range["start_line"]
            )

            end_line = (
                semantic_range["end_line"]
            )

            chunk_lines = lines[
                start_line - 1 : end_line
            ]

            chunk_content = "\n".join(
                chunk_lines
            )

            if not chunk_content.strip():
                continue

            sha = compute_sha256(
                chunk_content
            )

            chunk_id = (
                f"{file_path}:"
                f"{start_line}-"
                f"{end_line}:"
                f"{sha[:8]}"
            )

            # ------------------------------------------------
            # AST METADATA FOR THIS RANGE
            # ------------------------------------------------

            c_defined = sorted(
                set(
                    symbol
                    for symbol, line_number
                    in defined
                    if (
                        start_line
                        <= line_number
                        <= end_line
                    )
                )
            )

            c_called = sorted(
                set(
                    symbol
                    for symbol, line_number
                    in called
                    if (
                        start_line
                        <= line_number
                        <= end_line
                    )
                )
            )

            c_imported = sorted(
                set(
                    symbol
                    for symbol, line_number
                    in imported
                    if (
                        start_line
                        <= line_number
                        <= end_line
                    )
                )
            )

            chunks.append(
                {
                    "file_path": file_path,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_name": semantic_range[
                        "symbol_name"
                    ],
                    "chunk_type": semantic_range[
                        "chunk_type"
                    ],
                    "parent_symbol": semantic_range[
                        "parent_symbol"
                    ],
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "sha_hash": sha,
                    "defined_symbols": ",".join(
                        c_defined
                    ),
                    "called_symbols": ",".join(
                        c_called
                    ),
                    "imported_symbols": ",".join(
                        c_imported
                    ),
                }
            )

        logger.info(
            "AST semantic chunking: %s -> %d chunks",
            file_path,
            len(chunks),
        )

        return chunks

    # ========================================================
    # NON-PYTHON
    # ========================================================

    i = 0

    while i < total_lines:

        start_line = i + 1

        end_line = min(
            i + target_lines_per_chunk,
            total_lines,
        )

        chunk_lines = lines[
            i:end_line
        ]

        chunk_content = "\n".join(
            chunk_lines
        )

        # ----------------------------------------------------
        # Detect first symbol in the chunk.
        # ----------------------------------------------------

        current_symbol = None

        for line in chunk_lines:

            sym = extract_symbol_name(
                line,
                language,
            )

            if sym:

                current_symbol = sym

                break

        sha = compute_sha256(
            chunk_content
        )

        chunk_id = (
            f"{file_path}:"
            f"{start_line}-"
            f"{end_line}:"
            f"{sha[:8]}"
        )

        chunks.append(
            {
                "file_path": file_path,
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "symbol_name": current_symbol,
                "chunk_type": "fixed_window",
                "parent_symbol": None,
                "chunk_id": chunk_id,
                "content": chunk_content,
                "sha_hash": sha,
                "defined_symbols": "",
                "called_symbols": "",
                "imported_symbols": "",
            }
        )

        if end_line == total_lines:
            break

        i += (
            target_lines_per_chunk
            - overlap_lines
        )

    return chunks


# ============================================================
# PROCESS REPOSITORY FILES
# ============================================================

def process_repository_files(
    repo_dir: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Traverse repository directory.

    Returns:

        discovered_files
        all_chunks
    """

    discovered_files = []

    all_chunks = []

    MAX_FILES_LIMIT = int(os.environ.get("MAX_REPO_FILES", "1000"))
    MAX_CHUNKS_LIMIT = int(os.environ.get("MAX_REPO_CHUNKS", "5000"))
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(1024 * 1024)))

    for root, dirs, files in os.walk(
        repo_dir
    ):
        if len(discovered_files) >= MAX_FILES_LIMIT:
            logger.info("Reached maximum repository file limit (%s). Terminating file discovery.", MAX_FILES_LIMIT)
            break

        # ----------------------------------------------------
        # Remove ignored directories.
        # ----------------------------------------------------

        dirs[:] = [
            d
            for d in dirs
            if (
                d not in IGNORE_DIRS
                and not d.startswith(".")
            )
        ]

        for filename in files:
            if len(discovered_files) >= MAX_FILES_LIMIT:
                break

            full_path = os.path.join(
                root,
                filename,
            )

            rel_path = os.path.relpath(
                full_path,
                repo_dir,
            ).replace("\\", "/")

            # ------------------------------------------------
            # Ignore unwanted files.
            # ------------------------------------------------

            if is_ignored_path(
                rel_path
            ):
                continue

            try:

                with open(
                    full_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:

                    content = f.read()

                size_bytes = len(
                    content.encode("utf-8")
                )

                # ------------------------------------------------
                # Ignore empty or oversized files.
                # ------------------------------------------------

                if (
                    size_bytes == 0
                    or size_bytes > MAX_FILE_SIZE
                ):
                    continue

                line_count = len(
                    content.splitlines()
                )

                language = detect_language(
                    rel_path
                )

                sha = compute_sha256(
                    content
                )

                file_info = {
                    "file_path": rel_path,
                    "file_name": filename,
                    "language": language,
                    "size_bytes": size_bytes,
                    "line_count": line_count,
                    "sha_hash": sha,
                    "content": content,
                }

                discovered_files.append(
                    file_info
                )

                # ------------------------------------------------
                # Create semantic/fixed chunks if under budget.
                # ------------------------------------------------

                if len(all_chunks) < MAX_CHUNKS_LIMIT:
                    file_chunks = (
                        parse_file_into_chunks(
                            rel_path,
                            content,
                            language,
                        )
                    )

                    all_chunks.extend(
                        file_chunks[: MAX_CHUNKS_LIMIT - len(all_chunks)]
                    )

            except Exception as e:

                logger.warning(
                    "Skipping unreadable file '%s': %s",
                    rel_path,
                    e,
                )

                continue

    return (
        discovered_files,
        all_chunks,
    )
