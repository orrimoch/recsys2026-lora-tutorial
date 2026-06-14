"""GeminiClient.thinking_budget wiring (no API call): default None = SDK default
(unchanged for reranker/responder); explicit 0 disables thinking tokens (cheaper)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))
from mcrs.query_rewriters.gemini_propose import GeminiClient


def test_thinking_budget_defaults_to_none():
    assert GeminiClient(model="gemini-2.5-flash-lite").thinking_budget is None


def test_thinking_budget_can_be_disabled():
    assert GeminiClient(model="gemini-2.5-flash-lite", thinking_budget=0).thinking_budget == 0
