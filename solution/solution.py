"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    # Retrieval-side diagnostics. None means "retrieval was not evaluated" —
    # deliberately distinct from 0.0, which means "evaluated and scored zero".
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _clamp(value: float) -> float:
    """Keep a metric inside [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            # Nothing was claimed, so nothing is ungrounded.
            return 1.0
        context_tokens = _tokenize(context)
        overlap = len(answer_tokens & context_tokens)
        return _clamp(overlap / len(answer_tokens))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        overlap = len(_tokenize(answer) & question_tokens)
        return _clamp(overlap / len(question_tokens))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        overlap = len(_tokenize(answer) & expected_tokens)
        return _clamp(overlap / len(expected_tokens))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0

        union_tokens: set[str] = set()
        for chunk in contexts or []:
            union_tokens |= _tokenize(chunk)

        overlap = len(expected_tokens & union_tokens)
        return _clamp(overlap / len(expected_tokens))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        # Step 1 — mark each chunk relevant/not, keeping retriever order.
        flags = [
            len(_tokenize(chunk) & expected_tokens) / len(expected_tokens)
            >= relevance_threshold
            for chunk in contexts
        ]
        total_relevant = sum(flags)
        if total_relevant == 0:
            return 0.0

        # Steps 2-3 — Average Precision: a relevant chunk contributes the
        # precision at its own rank, so the same chunk placed earlier scores
        # higher than when it is buried under noise.
        hits = 0
        precision_sum = 0.0
        for k, is_relevant in enumerate(flags, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / k

        return _clamp(precision_sum / total_relevant)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        # Retrieval metrics are diagnostics only: they are computed when chunks
        # are available but never feed `passed` or overall_score().
        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
                retrieved_contexts=list(contexts) if contexts else [],
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    # TODO (Bonus — Exercise 3.5): implement the reranker
    raise NotImplementedError("Implement rerank_by_overlap")


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    # A judge that scores 4/5 where a human scores 3/5 is not "wrong" so much
    # as uncalibrated; keeping the raw text lets us re-audit rationales later.
    DEFAULT_SCORE = 0.5

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        rubric = rubric or {}
        criteria_block = "\n".join(
            f"- {name}: {description}" for name, description in rubric.items()
        ) or "- quality: Is this a good answer?"

        prompt = (
            "You are an impartial evaluator. Score the ANSWER against each "
            "rubric criterion on a 0.0-1.0 scale.\n"
            "Judge only the content: a longer answer is not a better answer, "
            "and unsupported claims must be penalised.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"RUBRIC:\n{criteria_block}\n\n"
            "Respond with a JSON object mapping each criterion name to its "
            'score, e.g. {"accuracy": 0.8}. Then explain your reasoning.'
        )

        raw_response = self.judge_llm_fn(prompt)
        parsed = self._parse_scores(raw_response)

        # Always answer for exactly the criteria that were asked about; an
        # unparseable judge falls back to 0.5 rather than silently dropping one.
        criteria = list(rubric) or ["quality"]
        scores = {
            name: parsed.get(name, self.DEFAULT_SCORE) for name in criteria
        }

        return {"scores": scores, "reasoning": raw_response}

    @staticmethod
    def _parse_scores(raw_response: str) -> dict[str, float]:
        """Pull a criterion → score mapping out of a judge response.

        Tolerates the usual LLM wrappers (prose, ```json fences) and both
        score conventions: 0-1 floats and the rubric's 1-5 integers.
        """
        if not raw_response:
            return {}

        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return {}

        if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
            payload = payload["scores"]
        if not isinstance(payload, dict):
            return {}

        scores: dict[str, float] = {}
        for name, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            score = float(value)
            if score > 1.0:
                score = score / 5.0  # rubric returned the 1-5 scale
            scores[name] = _clamp(score)
        return scores

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        per_item_means: list[float] = []
        for entry in scores_batch or []:
            values = [
                float(v) for v in (entry.get("scores") or {}).values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if values:
                per_item_means.append(sum(values) / len(values))

        if not per_item_means:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
                "average_score": None,
                "sample_size": 0,
            }

        average = sum(per_item_means) / len(per_item_means)

        # Positional bias: the answer shown first scoring higher than the rest.
        # With a single observation there is nothing to compare against, so we
        # report False rather than inventing a signal from n=1.
        positional_bias = False
        if len(per_item_means) > 1:
            rest = per_item_means[1:]
            positional_bias = per_item_means[0] - (sum(rest) / len(rest)) > 0.1

        return {
            "positional_bias": positional_bias,
            "leniency_bias": average > 0.8,
            "severity_bias": average < 0.3,
            "average_score": average,
            "sample_size": len(per_item_means),
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                # A pair with no chunks was never scored by a retriever, so it
                # reports None rather than a misleading recall/precision of 0.
                contexts=pair.retrieved_contexts or None,
            )
            # Keep the original pair so downstream analysis still has the
            # metadata (difficulty, category, source docs) attached.
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        passed = sum(1 for r in results if r.passed)

        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        def _avg(values: list[float]) -> float:
            return sum(values) / len(values)

        def _avg_optional(attr: str) -> float | None:
            # Skip results that never went through retrieval evaluation, so a
            # partially-instrumented run does not drag the average toward zero.
            scored = [
                getattr(r, attr) for r in results if getattr(r, attr, None) is not None
            ]
            return _avg(scored) if scored else None

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": _avg([r.faithfulness for r in results]),
            "avg_relevance": _avg([r.relevance for r in results]),
            "avg_completeness": _avg([r.completeness for r in results]),
            "avg_context_recall": _avg_optional("context_recall"),
            "avg_context_precision": _avg_optional("context_precision"),
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        """
        REGRESSION_TOLERANCE = 0.05

        def _avg(items: list, attr: str) -> float:
            if not items:
                return 0.0
            return sum(getattr(r, attr) for r in items) / len(items)

        report: dict[str, Any] = {}
        regressions: list[str] = []

        for metric in ("faithfulness", "relevance", "completeness"):
            new_avg = _avg(new_results, metric)
            baseline_avg = _avg(baseline_results, metric)
            report[f"new_avg_{metric}"] = new_avg
            report[f"baseline_avg_{metric}"] = baseline_avg
            # Strictly greater: a drop of exactly 0.05 is noise, not a regression.
            if baseline_avg - new_avg > REGRESSION_TOLERANCE:
                regressions.append(metric)

        report["regressions"] = regressions
        report["passed"] = not regressions
        return report

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            r for r in results
            if min(r.faithfulness, r.relevance, r.completeness) < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        counts: dict[str, int] = {}
        for failure in failures or []:
            label = failure.failure_type or "unknown"
            counts[label] = counts.get(label, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        causes = {
            "faithfulness": "Context is missing or irrelevant — improve retrieval",
            "relevance": "Answer does not address the question — improve prompt clarity",
            "completeness": (
                "Answer is missing key information — "
                "increase context window or improve generation"
            ),
        }
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }

        # Two or more metrics below the bar means no single fix will clear it;
        # naming one root cause there would send the team down the wrong path.
        weak = [name for name, score in scores.items() if score < 0.5]
        if len(weak) >= 2:
            return "Multiple issues detected — review full pipeline"

        lowest = min(scores, key=lambda name: scores[name])
        return causes[lowest]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]

        for index, failure in enumerate(failures or []):
            failure_id = f"F{index + 1:03d}"
            failure_type = failure.failure_type or "unknown"
            root_cause = self.find_root_cause(failure)
            # The suggestion list may be shorter than the failure list; reuse the
            # last one rather than dropping the row.
            if suggestions:
                fix = suggestions[index] if index < len(suggestions) else suggestions[-1]
            else:
                fix = "TBD"
            lines.append(
                f"| {failure_id} | {failure_type} | {root_cause} | {fix} | Open |"
            )

        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        by_type = self.categorize_failures(failures)

        playbook = {
            "hallucination": (
                "Implement a hallucination checker that drops claims absent from "
                "the retrieved context, and require the model to cite source_doc"
            ),
            "irrelevant": (
                "Rewrite the system prompt to restate the question before "
                "answering, and add query rewriting so retrieval matches intent"
            ),
            "incomplete": (
                "Increase chunk size and top-k to reduce context fragmentation, "
                "and add few-shot examples that list every condition and exception"
            ),
            "off_topic": (
                "Add intent classification to route out-of-scope questions instead "
                "of answering them from the wrong document set"
            ),
            "refusal": (
                "Loosen the guardrail so in-scope questions are answered; "
                "restrict refusals to topics genuinely outside the corpus"
            ),
            "unknown": (
                "Label the uncategorised failures manually before choosing a fix — "
                "an unnamed failure mode cannot be prioritised"
            ),
        }

        # Highest-count failure type first: fixing one root cause usually clears
        # a whole cluster, so the ordering IS the prioritisation.
        ranked = sorted(by_type.items(), key=lambda item: item[1], reverse=True)
        suggestions = [
            playbook.get(name, f"Investigate recurring '{name}' failures")
            for name, _ in ranked
        ]

        fallbacks = [
            "Add the failing cases to the golden dataset so the fix is regression-tested",
            "Review retrieval quality first (Context Recall) before touching prompts — "
            "low recall cannot be fixed by generation changes",
            "Calibrate evaluation thresholds against human labels on a sample of failures",
        ]
        for suggestion in fallbacks:
            if len(suggestions) >= 3:
                break
            suggestions.append(suggestion)

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
