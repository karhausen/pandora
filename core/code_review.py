from __future__ import annotations
import ast
from pathlib import Path
from .config import PROTECTED_CORE_FILES

class CodeReview:
    def review_file_change(self, file_path: str, new_content: str) -> dict:
        issues: list[str] = []
        warnings: list[str] = []
        path = Path(file_path)
        if path.name in PROTECTED_CORE_FILES:
            issues.append(f"Protected core file requires explicit approval: {path.name}")
        if path.suffix == ".py":
            try:
                ast.parse(new_content)
            except SyntaxError as exc:
                issues.append(f"SyntaxError: {exc}")
        for token in ["os.system(", "subprocess.", "eval(", "exec(", "__import__(", "shutil.rmtree", "socket."]:
            if token in new_content:
                warnings.append(f"Risky token found: {token}")
        risk = "HIGH" if issues else ("MEDIUM" if warnings else "LOW")
        return {"ok": not issues, "risk": risk, "issues": issues, "warnings": warnings}

    def review_many(self, changes: dict[str, str]) -> dict:
        reviews = {path: self.review_file_change(path, content) for path, content in changes.items()}
        ok = all(r["ok"] for r in reviews.values())
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        risk = max((r["risk"] for r in reviews.values()), key=lambda r: order[r], default="LOW")
        return {"ok": ok, "risk": risk, "files": reviews}
