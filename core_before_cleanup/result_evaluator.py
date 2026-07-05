from __future__ import annotations

class ResultEvaluator:
    def evaluate(self, action: dict, result) -> dict:
        if action.get("type") == "reject":
            return {"success": False, "quality": "rejected", "reason": "Action rejected."}
        if isinstance(result, dict) and result.get("success") is False:
            return {"success": False, "quality": "failed", "reason": result.get("error")}
        if hasattr(result, "success"):
            return {"success": bool(result.success), "quality": "ok" if result.success else "failed", "reason": None if result.success else result.error}
        return {"success": True, "quality": "ok", "reason": None}
