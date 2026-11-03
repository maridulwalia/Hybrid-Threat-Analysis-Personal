"""
ml_service/tests/test_llm_isolation.py
──────────────────────────────────────
Static AST import-graph isolation test for Phase 11.
Verifies that Phases 1-10 modules maintain zero hard dependencies on ml_service.src.llm.
"""

import ast
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def test_static_import_graph_isolation():
    """Ensure no file outside ml_service/src/llm/ imports from llm or external LLM SDKs."""
    src_dir = _ROOT / "ml_service" / "src"
    violations = []

    for root, _, files in os.walk(src_dir):
        # Skip llm module itself
        if "llm" in Path(root).parts:
            continue

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=str(file_path))
                except Exception as e:
                    violations.append(f"SyntaxError in {file_path}: {e}")
                    continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "llm" in alias.name or "google.genai" in alias.name:
                            violations.append(f"{file_path.name} imports forbidden module: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if "llm" in mod or "google.genai" in mod:
                        violations.append(f"{file_path.name} imports from forbidden module: {mod}")

    assert not violations, f"LLM Isolation Violations detected:\n" + "\n".join(violations)


if __name__ == "__main__":
    test_static_import_graph_isolation()
    print("ALL ISOLATION ASSERTIONS PASSED: Zero dependencies on ml_service.src.llm found in upstream phases!")

