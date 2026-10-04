import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger("repolens.github.parser")

# Directories to ignore completely
IGNORE_DIRS = {
    ".git", "node_modules", "pycache", "__pycache__", "dist", "build", ".venv",
    "venv", "env", "target", "vendor", ".idea", ".vscode", "coverage", ".next",
    ".nuxt", ".output", "bin", "obj", "out", ".cargo", ".gradle"
}

# File extensions to ignore
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".pdf", ".zip",
    ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".iso", ".mp3", ".mp4", ".wav", ".avi", ".mov", ".pyc", ".pyo", ".pyd",
    ".class", ".o", ".obj", ".db", ".sqlite", ".sqlite3", ".log", ".DS_Store",
    ".lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock"
}

# Ignore secret files
IGNORE_FILES = {
    ".env", ".env.local", ".env.development", ".env.production", "id_rsa",
    "id_rsa.pub", "credentials.json", "service-account.json"
}

# File extension to language mapping
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
    ".dockerfile": "dockerfile"
}

def is_ignored_path(relative_path: str) -> bool:
    parts = relative_path.replace("\\", "/").split("/")
    # Check directory names
    for part in parts[:-1]:
        if part in IGNORE_DIRS or part.startswith("."):
            return True
    
    filename = parts[-1]
    if filename in IGNORE_FILES:
        return True

    ext = os.path.splitext(filename)[1].lower()
    if ext in IGNORE_EXTENSIONS:
        return True

    return False

def detect_language(file_path: str) -> str:
    filename = os.path.basename(file_path).lower()
    if filename == "dockerfile":
        return "dockerfile"
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "text")

def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

def extract_symbol_name(line: str, language: str) -> Optional[str]:
    """
    Tries to extract class/function name from code lines using regex.
    """
    line_clean = line.strip()
    if language in ["python"]:
        match = re.match(r"^(?:async\s+)?def\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)", line_clean)
        if match:
            return match.group(1) or match.group(2)
    elif language in ["javascript", "typescript"]:
        match = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)|(?:export\s+)?class\s+([a-zA-Z0-9_]+)|(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\(", line_clean)
        if match:
            return match.group(1) or match.group(2) or match.group(3)
    elif language in ["java", "csharp", "cpp", "c"]:
        match = re.match(r"^(?:public|private|protected|static|\s)*(?:class|interface|enum|void|int|string|boolean|auto)\s+([a-zA-Z0-9_]+)", line_clean)
        if match:
            return match.group(1)
    elif language in ["go"]:
        match = re.match(r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)|type\s+([a-zA-Z0-9_]+)\s+struct", line_clean)
        if match:
            return match.group(1) or match.group(2)
    elif language in ["rust"]:
        match = re.match(r"^(?:pub\s+)?fn\s+([a-zA-Z0-9_]+)|^(?:pub\s+)?struct\s+([a-zA-Z0-9_]+)", line_clean)
        if match:
            return match.group(1) or match.group(2)

    return None

def parse_file_into_chunks(
    file_path: str,
    content: str,
    language: str,
    target_lines_per_chunk: int = 50,
    overlap_lines: int = 10
) -> List[Dict[str, Any]]:
    """
    Splits file content into code-aware chunks preserving line numbers and detected symbols.
    """
    lines = content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    chunks = []
    i = 0

    while i < total_lines:
        start_line = i + 1
        end_line = min(i + target_lines_per_chunk, total_lines)
        chunk_lines = lines[i:end_line]
        chunk_content = "\n".join(chunk_lines)

        # Look for symbol in the chunk lines
        current_symbol = None
        for l in chunk_lines:
            sym = extract_symbol_name(l, language)
            if sym:
                current_symbol = sym
                break

        sha = compute_sha256(chunk_content)
        chunk_id = f"{file_path}:{start_line}-{end_line}:{sha[:8]}"

        chunks.append({
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": current_symbol,
            "chunk_id": chunk_id,
            "content": chunk_content,
            "sha_hash": sha
        })

        if end_line == total_lines:
            break
        i += (target_lines_per_chunk - overlap_lines)

    return chunks

def process_repository_files(repo_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Traverses repository directory, filters relevant code files, and parses them into files & chunks.
    """
    discovered_files = []
    all_chunks = []

    for root, dirs, files in os.walk(repo_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_dir).replace("\\", "/")

            if is_ignored_path(rel_path):
                continue

            try:
                # Read file content safely
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip empty or excessively huge generated files (> 2MB)
                size_bytes = len(content.encode("utf-8"))
                if size_bytes == 0 or size_bytes > 2 * 1024 * 1024:
                    continue

                line_count = len(content.splitlines())
                language = detect_language(rel_path)
                sha = compute_sha256(content)

                file_info = {
                    "file_path": rel_path,
                    "file_name": filename,
                    "language": language,
                    "size_bytes": size_bytes,
                    "line_count": line_count,
                    "sha_hash": sha,
                    "content": content
                }
                discovered_files.append(file_info)

                # Parse into chunks
                file_chunks = parse_file_into_chunks(rel_path, content, language)
                all_chunks.extend(file_chunks)

            except Exception as e:
                logger.warning(f"Skipping unreadable file '{rel_path}': {e}")
                continue

    return discovered_files, all_chunks
