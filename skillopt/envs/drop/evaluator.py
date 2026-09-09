"""Dependency-free implementation of the official DROP EM/F1 semantics.

The normalization, numeric matching, per-span F1 and maximum-over-validated-
answers behavior follow AllenAI's official ``drop_eval.py``.  The official
implementation uses SciPy's Hungarian solver; here a small dynamic program
computes the same optimal one-to-one alignment without adding a dependency.
"""
from __future__ import annotations

import ast
import json
import re
import string
from functools import lru_cache
from typing import Any, Iterable, Sequence

from skillopt.envs.drop.data import answer_json_to_strings


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_EXCLUDE = set(string.punctuation)


def extract_answer(text: str) -> str | list[str]:
    matches = _ANSWER_RE.findall(str(text))
    answer = matches[-1].strip() if matches else str(text).strip()
    if answer.startswith("["):
        parsed: Any = None
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)) and all(
            isinstance(item, (str, int, float)) for item in parsed
        ):
            return [str(item) for item in parsed]
    return answer


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_answer(text: str) -> str:
    parts = []
    for token in re.split(" |-", str(text).lower()):
        if not _is_number(token):
            token = "".join(char for char in token if char not in _EXCLUDE)
        if _is_number(token):
            token = str(float(token))
        token = re.sub(r"\b(a|an|the)\b", " ", token)
        token = " ".join(token.split())
        if token.strip():
            parts.append(token)
    return " ".join(parts).strip()


def _answer_to_bags(answer: str | Sequence[str]) -> tuple[list[str], list[set[str]]]:
    spans = list(answer) if isinstance(answer, (list, tuple)) else [answer]
    normalized = [_normalize_answer(str(span)) for span in spans]
    return normalized, [set(span.split()) for span in normalized]


def _match_numbers_if_present(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {word for word in gold if _is_number(word)}
    predicted_numbers = {word for word in predicted if _is_number(word)}
    return not gold_numbers or bool(gold_numbers & predicted_numbers)


def _bag_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(gold & predicted)
    precision = intersection / float(len(predicted)) if predicted else 1.0
    recall = intersection / float(len(gold)) if gold else 1.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _optimal_alignment_scores(
    predicted: list[set[str]],
    gold: list[set[str]],
) -> list[float]:
    size = max(len(predicted), len(gold))
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for gold_index, gold_bag in enumerate(gold):
        for pred_index, pred_bag in enumerate(predicted):
            if _match_numbers_if_present(gold_bag, pred_bag):
                matrix[gold_index][pred_index] = _bag_f1(pred_bag, gold_bag)

    @lru_cache(maxsize=None)
    def solve(row: int, used_columns: int) -> tuple[float, tuple[float, ...]]:
        if row == size:
            return 0.0, ()
        best_total = -1.0
        best_scores: tuple[float, ...] = ()
        for column in range(size):
            if used_columns & (1 << column):
                continue
            tail_total, tail_scores = solve(row + 1, used_columns | (1 << column))
            score = matrix[row][column]
            total = score + tail_total
            if total > best_total:
                best_total = total
                best_scores = (score, *tail_scores)
        return best_total, best_scores

    return list(solve(0, 0)[1])


def get_metrics(
    predicted: str | Sequence[str],
    gold: str | Sequence[str],
) -> tuple[float, float]:
    predicted_normalized, predicted_bags = _answer_to_bags(predicted)
    gold_normalized, gold_bags = _answer_to_bags(gold)
    exact_match = float(
        set(predicted_normalized) == set(gold_normalized)
        and len(predicted_normalized) == len(gold_normalized)
    )
    aligned = _optimal_alignment_scores(predicted_bags, gold_bags)
    f1 = round(sum(aligned) / max(len(aligned), 1), 2)
    return exact_match, f1


def evaluate(
    prediction_text: str,
    candidate_answers: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    predicted = extract_answer(prediction_text)
    best_em = 0.0
    best_f1 = 0.0
    best_type = ""
    gold_values: list[list[str]] = []
    for answer in candidate_answers:
        gold, answer_type = answer_json_to_strings(answer)
        gold_values.append(gold)
        em, f1 = get_metrics(predicted, gold)
        if em > best_em or (em == best_em and f1 >= best_f1):
            best_em = em
            best_f1 = f1
            best_type = answer_type
    return {
        "em": best_em,
        "f1": best_f1,
        "predicted_answer": predicted,
        "gold_answers": gold_values,
        "answer_type": best_type,
    }
