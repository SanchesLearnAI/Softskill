from __future__ import annotations

import math
import re
from typing import Any


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(str(text))
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _as_number(value: Any) -> float | None:
    text = _normalize_text(value).replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def relaxed_accuracy(predicted: Any, gold: Any, *, tolerance: float = 0.05) -> float:
    """ChartQA relaxed accuracy: exact text or 5% relative numeric tolerance."""
    predicted_text = _normalize_text(predicted)
    gold_text = _normalize_text(gold)
    if predicted_text == gold_text:
        return 1.0
    predicted_number = _as_number(predicted_text)
    gold_number = _as_number(gold_text)
    if predicted_number is None or gold_number is None:
        return 0.0
    if gold_number == 0.0:
        return float(predicted_number == 0.0)
    return float(abs(predicted_number - gold_number) <= abs(gold_number) * float(tolerance))


def evaluate(prediction_text: str, gold_answer: Any) -> dict[str, Any]:
    predicted_answer = extract_answer(prediction_text)
    score = relaxed_accuracy(predicted_answer, gold_answer)
    return {
        "relaxed_accuracy": score,
        "predicted_answer": predicted_answer,
        "gold_answer": str(gold_answer),
    }
