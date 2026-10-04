import re
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.file import RepositoryFile
from app.schemas.insights import InsightItem, CodeInsightsResponse, DependencyItem, DependencyAnalysisResponse

SECRET_PATTERNS = [
    (re.compile(r"""(?:api_key|apikey|secret|password|auth_token)\s*=\s*['"][A-Za-z0-9_\-]{16,}['"]""", re.IGNORECASE), "Potential hardcoded credential or API secret"),
    (re.compile(r"""(http://[^\s'"]+)""", re.IGNORECASE), "Insecure HTTP connection string (prefer HTTPS)"),
    (re.compile(r"""eval\s*\(""", re.IGNORECASE), "Usage of dynamic eval() function poses potential code execution risk"),
    (re.compile(r"""SELECT\s+.*\s+FROM\s+.*%.*""", re.IGNORECASE), "Potential SQL injection vulnerability via string formatting")
]

def analyze_code_insights(db: Session, repository_id: str) -> CodeInsightsResponse:
    files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repository_id).all()
    insights: List[InsightItem] = []

    for f in files:
        # Check 1: Large file
        if f.line_count > 300:
            insights.append(InsightItem(
                severity="medium",
                category="complexity",
                file_path=f.file_path,
                line_number=1,
                title="Large File Detected",
                explanation=f"File contains {f.line_count} lines of code. Large files increase cognitive load and refactoring difficulty.",
                suggestion="Consider splitting modular components or functions into smaller sub-modules."
            ))

        content = f.content or ""
        lines = content.splitlines()

        # Check 2: Missing docstring / documentation in Python/JS entrypoints
        if f.line_count > 50 and not content.startswith(('"""', "'''", "//", "/*", "#")):
            insights.append(InsightItem(
                severity="low",
                category="documentation",
                file_path=f.file_path,
                line_number=1,
                title="Missing Module Documentation",
                explanation="No top-level module docstring or comment header found.",
                suggestion="Add a brief header explaining the module purpose and exported functionality."
            ))

        # Check 3: Security patterns
        for line_idx, line in enumerate(lines, 1):
            for pattern, desc in SECRET_PATTERNS:
                if pattern.search(line):
                    insights.append(InsightItem(
                        severity="high",
                        category="security",
                        file_path=f.file_path,
                        line_number=line_idx,
                        title="Potential Security Concern Detected",
                        explanation=f"{desc} on line {line_idx}.",
                        suggestion="Store sensitive keys and credentials in environment variables (`.env`) rather than source code."
                    ))

        # Check 4: Bare try-except or catch-all in Python
        if f.language == "python":
            for line_idx, line in enumerate(lines, 1):
                if re.search(r"except\s*:", line):
                    insights.append(InsightItem(
                        severity="medium",
                        category="maintenance",
                        file_path=f.file_path,
                        line_number=line_idx,
                        title="Bare Exception Clause",
                        explanation="Bare 'except:' catches all exceptions, including KeyboardInterrupt and SystemExit.",
                        suggestion="Catch specific exception types (e.g. `except ValueError:`) to prevent silent failures."
                    ))

    return CodeInsightsResponse(
        repository_id=repository_id,
        total_issues=len(insights),
        insights=insights
    )

def analyze_dependencies(db: Session, repository_id: str) -> DependencyAnalysisResponse:
    files = db.query(RepositoryFile).filter(RepositoryFile.repository_id == repository_id).all()
    dependencies: List[DependencyItem] = []
    ecosystems_found = set()

    for f in files:
        fname = f.file_name.lower()
        content = f.content or ""

        # Node / NPM (package.json)
        if fname == "package.json":
            ecosystems_found.add("npm")
            try:
                data = json.loads(content)
                deps = data.get("dependencies", {})
                dev_deps = data.get("devDependencies", {})
                for pkg, ver in deps.items():
                    dependencies.append(DependencyItem(
                        name=pkg,
                        version=str(ver),
                        type="production",
                        ecosystem="npm",
                        outdated=False,
                        details="Runtime dependency"
                    ))
                for pkg, ver in dev_deps.items():
                    dependencies.append(DependencyItem(
                        name=pkg,
                        version=str(ver),
                        type="development",
                        ecosystem="npm",
                        outdated=False,
                        details="Development dependency"
                    ))
            except Exception:
                pass

        # Python (requirements.txt)
        elif fname == "requirements.txt":
            ecosystems_found.add("pip")
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = re.split(r"[==>=<=~=]", line)
                    pkg = parts[0].strip()
                    ver = parts[1].strip() if len(parts) > 1 else "latest"
                    dependencies.append(DependencyItem(
                        name=pkg,
                        version=ver,
                        type="production",
                        ecosystem="pip",
                        outdated=False,
                        details="Python package dependency"
                    ))

    return DependencyAnalysisResponse(
        repository_id=repository_id,
        ecosystems_found=list(ecosystems_found) if ecosystems_found else ["custom"],
        total_dependencies=len(dependencies),
        dependencies=dependencies
    )
