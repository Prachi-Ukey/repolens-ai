import pytest
from app.github.client import validate_github_url, parse_github_url
from app.github.parser import is_ignored_path, detect_language, parse_file_into_chunks

def test_github_url_validation():
    valid_urls = [
        "https://github.com/expressjs/express",
        "http://github.com/fastapi/fastapi",
        "https://www.github.com/torvalds/linux.git"
    ]
    for url in valid_urls:
        assert validate_github_url(url) is True
        owner, repo = parse_github_url(url)
        assert len(owner) > 0
        assert len(repo) > 0

    invalid_urls = ["not_a_url", "https://google.com/foo/bar", "ftp://github.com/a/b"]
    for url in invalid_urls:
        assert validate_github_url(url) is False

def test_file_filtering_and_language():
    assert is_ignored_path("node_modules/express/index.js") is True
    assert is_ignored_path(".git/config") is True
    assert is_ignored_path(".env") is True
    assert is_ignored_path("src/components/App.jsx") is False

    assert detect_language("app/main.py") == "python"
    assert detect_language("src/index.ts") == "typescript"
    assert detect_language("Dockerfile") == "dockerfile"

def test_code_chunking():
    sample_code = """def hello():
    print("Hello world")

class UserAuth:
    def login(self):
        return True
"""
    chunks = parse_file_into_chunks("app/auth.py", sample_code, "python", target_lines_per_chunk=3)
    assert len(chunks) > 0
    assert chunks[0]["file_path"] == "app/auth.py"
    assert "start_line" in chunks[0]
    assert "end_line" in chunks[0]
