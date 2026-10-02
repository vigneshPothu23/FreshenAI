"""Sprint 6 — the evaluation harness.

A fixed test set, scored the same way every time. This is what separates a
demo from a system: without it, "the chatbot seems better now" is an opinion,
and a change that quietly breaks intent routing ships unnoticed.

Five suites, each gating a different sprint's output:

==========  ================================================================
Suite       What it protects
==========  ================================================================
nlu         Intent classification and entity extraction (Sprint 5 planner)
routing     That an intent produces the right agents and actions
rag         Retrieval accuracy, groundedness and correct refusal (Sprint 4)
forecast    That model error has not regressed against the baseline (Sprint 2)
agents      End-to-end orchestration completes and uses the right specialists
==========  ================================================================

Two properties matter more than the metrics themselves:

**Missing dependencies are skipped, not failed.** With no vector index built,
the RAG suite reports *skipped* rather than 0%. Scoring an absent component as
a failure buries real regressions under noise that clears itself when someone
runs a setup script.

**Refusal is scored as a capability.** A question outside the corpus that gets
answered anyway is a failure, exactly as a question inside it that gets refused
is. Systems that only measure answer quality reward confident fabrication.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from freshsense.config import SETTINGS
from freshsense.db.repository import MonitoringRepository
from freshsense.db.session import Database, get_database
from freshsense.logging_config import get_logger
from freshsense.paths import PATHS

LOG = get_logger(__name__)

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"

#: Minimum score for a suite to pass, as a percentage. Overridable by a
#: ``"thresholds"`` object in ``evaluation/eval_set.json`` so the gate can be
#: tightened without a code change.
PASS_THRESHOLDS: dict[str, float] = {
    "nlu": 80.0,
    "routing": 80.0,
    "rag": 70.0,
    "forecast": 75.0,
    "agents": 80.0,
}

#: Forecast regression gates: mean MAPE ceiling and the share of products that
#: must beat the seasonal-naive baseline.
FORECAST_MAX_MAPE = 20.0
FORECAST_MIN_BEATING_BASELINE = 70.0

#: Built-in cases, used when ``evaluation/eval_set.json`` is absent so the
#: harness is runnable before that file exists. The JSON file, when present,
#: replaces these entirely rather than extending them — a partially-overridden
#: test set is impossible to reason about.
DEFAULT_CASES: list[dict[str, Any]] = [
    # ── nlu ───────────────────────────────────────────────────────────
    {"id": "nlu-01", "suite": "nlu", "query": "What is at risk today in T Nagar?",
     "expected_intent": "at_risk_review", "expected_entities": {"zone": "T Nagar"}},
    {"id": "nlu-02", "suite": "nlu", "query": "How much should I discount tomatoes expiring in 1 day?",
     "expected_intent": "pricing",
     "expected_entities": {"product": "Tomato", "days": 1}},
    {"id": "nlu-03", "suite": "nlu", "query": "Forecast paneer demand for next week",
     "expected_intent": "forecast_demand", "expected_entities": {"product": "Paneer"}},
    {"id": "nlu-04", "suite": "nlu", "query": "Who will buy 40 kg of brinjal?",
     "expected_intent": "find_buyers",
     "expected_entities": {"product": "Brinjal", "quantity": 40.0}},
    {"id": "nlu-05", "suite": "nlu", "query": "I need 60 kg tomatoes cheapest supplier near Adyar",
     "expected_intent": "source_supply",
     "expected_entities": {"product": "Tomato", "quantity": 60.0, "zone": "Adyar"}},
    {"id": "nlu-06", "suite": "nlu", "query": "What should I reorder this week?",
     "expected_intent": "restocking"},
    {"id": "nlu-07", "suite": "nlu", "query": "What temperature must paneer be stored at under FSSAI rules?",
     "expected_intent": "knowledge", "expected_entities": {"product": "Paneer"}},
    {"id": "nlu-08", "suite": "nlu", "query": "Do I have any whitebread in stock?",
     "expected_intent": "inventory_lookup",
     "expected_entities": {"product": "White Bread"}},
    {"id": "nlu-09", "suite": "nlu", "query": "Walk me through my morning priorities",
     "expected_intent": "daily_briefing"},
    {"id": "nlu-10", "suite": "nlu", "query": "Show me brown bread batches in Velachery",
     "expected_intent": "inventory_lookup",
     "expected_entities": {"product": "Brown Bread", "zone": "Velachery"}},

    # ── routing ───────────────────────────────────────────────────────
    {"id": "route-01", "suite": "routing", "query": "What is at risk today?",
     "expected_agents": ["inventory", "recommendation"]},
    {"id": "route-02", "suite": "routing", "query": "Forecast demand for milk",
     "expected_agents": ["forecast"], "expected_actions": ["forecast_demand"]},
    {"id": "route-03", "suite": "routing", "query": "What does FSSAI require for cold storage?",
     "expected_agents": ["rag"], "expected_actions": ["answer_question"]},
    {"id": "route-04", "suite": "routing", "query": "I need 30 kg of onions",
     "expected_agents": ["recommendation"], "expected_actions": ["match_sellers"]},
    {"id": "route-05", "suite": "routing", "query": "Walk me through my morning priorities",
     "expected_agents": ["inventory", "forecast", "recommendation"]},
    {"id": "route-06", "suite": "routing", "query": "What should I reorder?",
     "expected_actions": ["recommend_restocking"]},

    # ── rag ───────────────────────────────────────────────────────────
    {"id": "rag-01", "suite": "rag", "question": "What temperature must paneer be stored at?",
     "expected_keywords": ["2", "4"], "should_refuse": False},
    {"id": "rag-02", "suite": "rag", "question": "How long do coriander leaves last at ambient temperature?",
     "expected_keywords": ["two", "2"], "should_refuse": False},
    {"id": "rag-03", "suite": "rag", "question": "What discount applies on the final day before expiry?",
     "expected_keywords": ["seventy", "70"], "should_refuse": False},
    {"id": "rag-04", "suite": "rag", "question": "What are the criteria for grade A?",
     "expected_keywords": ["two days", "thirty five", "35"], "should_refuse": False},
    {"id": "rag-05", "suite": "rag", "question": "What happens to stock past its expiry date?",
     "expected_keywords": ["ngo", "compost", "not be sold"], "should_refuse": False},
    {"id": "rag-06", "suite": "rag", "question": "What is the capital of Iceland?",
     "should_refuse": True},
    {"id": "rag-07", "suite": "rag", "question": "Who won the 2019 cricket world cup?",
     "should_refuse": True},
    {"id": "rag-08", "suite": "rag", "question": "Explain quantum entanglement",
     "should_refuse": True},

    # ── agents (end to end) ───────────────────────────────────────────
    {"id": "agent-01", "suite": "agents", "query": "What is at risk today in Adyar?",
     "expected_agents": ["inventory"]},
    {"id": "agent-02", "suite": "agents", "query": "I need 20 kg of tomatoes near Adyar",
     "expected_agents": ["recommendation"]},
    {"id": "agent-03", "suite": "agents", "query": "Walk me through my morning priorities",
     "expected_agents": ["inventory", "recommendation"]},
    {"id": "agent-04", "suite": "agents", "query": "What should I reorder this week?",
     "expected_agents": ["recommendation"]},
]


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class EvalCase:
    """One test case. Fields not relevant to a suite are simply unset."""

    id: str
    suite: str
    query: str = ""
    question: str = ""
    expected_intent: str = ""
    expected_entities: dict[str, Any] = field(default_factory=dict)
    expected_agents: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    expected_source: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    should_refuse: bool = False
    notes: str = ""

    @property
    def prompt(self) -> str:
        return self.query or self.question

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class CaseResult:
    """Outcome of one case."""

    case_id: str
    suite: str
    status: str = PASSED
    score: float = 1.0
    detail: str = ""
    latency_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "suite": self.suite, "status": self.status,
            "score": round(self.score, 3), "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


@dataclass
class SuiteResult:
    """Aggregate outcome for one suite."""

    name: str
    results: list[CaseResult] = field(default_factory=list)
    threshold: float = 80.0
    skipped_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if r.status != SKIPPED]

    @property
    def score(self) -> float:
        if not self.scored:
            return 0.0
        return round(sum(r.score for r in self.scored) / len(self.scored) * 100, 1)

    @property
    def status(self) -> str:
        if self.skipped_reason or not self.scored:
            return SKIPPED
        return PASSED if self.score >= self.threshold else FAILED

    @property
    def summary(self) -> str:
        if self.status == SKIPPED:
            return self.skipped_reason or "No scorable case ran."
        passing = sum(1 for r in self.scored if r.passed)
        return (
            f"{passing}/{len(self.scored)} case(s) passed — {self.score:.1f}% "
            f"against a {self.threshold:.0f}% threshold."
        )

    def frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=["case_id", "status", "score", "detail"])
        return pd.DataFrame([r.as_dict() for r in self.results])[
            ["case_id", "status", "score", "detail", "latency_ms"]
        ]


@dataclass
class EvaluationReport:
    """The whole run."""

    suites: dict[str, SuiteResult] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    duration_ms: int = 0

    @property
    def scored_suites(self) -> list[SuiteResult]:
        return [s for s in self.suites.values() if s.status != SKIPPED]

    @property
    def overall_score(self) -> float:
        scored = self.scored_suites
        if not scored:
            return 0.0
        return round(sum(s.score for s in scored) / len(scored), 1)

    @property
    def passed(self) -> bool:
        scored = self.scored_suites
        return bool(scored) and all(s.status == PASSED for s in scored)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "suite": name,
                "status": suite.status,
                "score": suite.score if suite.status != SKIPPED else None,
                "threshold": suite.threshold,
                "cases": len(suite.scored),
                "detail": suite.summary,
            }
            for name, suite in self.suites.items()
        ])

    def failures(self) -> pd.DataFrame:
        rows = [
            r.as_dict() | {"suite": name}
            for name, suite in self.suites.items()
            for r in suite.results if r.status == FAILED
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["case_id", "suite", "status", "score", "detail"]
        )


# ══════════════════════════════════════════════════════════════════════════
def load_eval_set(path: Path | None = None) -> tuple[list[EvalCase], dict[str, float]]:
    """Load the test set, falling back to the built-in cases.

    Returns:
        ``(cases, thresholds)``. A JSON file may be either a bare list of cases
        or an object with ``"cases"`` and optional ``"thresholds"``.
    """
    path = path or (PATHS.root / "evaluation" / "eval_set.json")
    thresholds = dict(PASS_THRESHOLDS)

    if not path.is_file():
        LOG.info("No eval set at %s — using %d built-in case(s)",
                 PATHS.relative(path), len(DEFAULT_CASES))
        return [EvalCase.from_dict(c) for c in DEFAULT_CASES], thresholds

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOG.error("Eval set at %s is not valid JSON (%s) — using built-in cases",
                  PATHS.relative(path), exc)
        return [EvalCase.from_dict(c) for c in DEFAULT_CASES], thresholds

    if isinstance(payload, dict):
        raw_cases = payload.get("cases", [])
        thresholds.update(payload.get("thresholds", {}))
    else:
        raw_cases = payload

    cases = [EvalCase.from_dict(c) for c in raw_cases if isinstance(c, dict)]
    LOG.info("Loaded %d case(s) from %s", len(cases), PATHS.relative(path))
    return cases, thresholds


# ══════════════════════════════════════════════════════════════════════════
class Evaluator:
    """Runs the fixed test set and scores each suite."""

    def __init__(
        self,
        cases: Sequence[EvalCase] | None = None,
        *,
        thresholds: dict[str, float] | None = None,
        db: Database | None = None,
        coordinator: Any = None,
    ) -> None:
        loaded, loaded_thresholds = load_eval_set()
        self.cases = list(cases) if cases is not None else loaded
        self.thresholds = {**loaded_thresholds, **(thresholds or {})}
        self.db = db or get_database()
        self.monitoring = MonitoringRepository(self.db)
        self._coordinator = coordinator

    def suite_cases(self, suite: str) -> list[EvalCase]:
        return [c for c in self.cases if c.suite == suite]

    @property
    def coordinator(self) -> Any:
        if self._coordinator is None:
            from freshsense.agents import get_coordinator

            self._coordinator = get_coordinator()
        return self._coordinator

    # ── nlu ───────────────────────────────────────────────────────────
    def run_nlu(self) -> SuiteResult:
        """Intent classification and entity extraction."""
        suite = SuiteResult("nlu", threshold=self.thresholds.get("nlu", 80.0))
        cases = self.suite_cases("nlu")
        if not cases:
            suite.skipped_reason = "No NLU cases defined."
            return suite

        from freshsense.agents.planner_agent import PlannerAgent

        planner = PlannerAgent()
        for case in cases:
            started = time.perf_counter()
            intent, _, _ = planner.classify(case.prompt)
            entities = {
                "product": planner.extract_product(case.prompt),
                "zone": planner.extract_zone(case.prompt),
                "quantity": planner.extract_quantity(case.prompt),
                "days": planner.extract_days(case.prompt),
            }

            checks: list[bool] = []
            problems: list[str] = []

            if case.expected_intent:
                correct = intent == case.expected_intent
                checks.append(correct)
                if not correct:
                    problems.append(f"intent {intent!r} != {case.expected_intent!r}")

            for key, expected in case.expected_entities.items():
                actual = entities.get(key)
                correct = str(actual).lower() == str(expected).lower()
                checks.append(correct)
                if not correct:
                    problems.append(f"{key} {actual!r} != {expected!r}")

            score = sum(checks) / len(checks) if checks else 0.0
            suite.results.append(CaseResult(
                case.id, "nlu",
                PASSED if score == 1.0 else FAILED,
                score,
                "; ".join(problems) or f"intent={intent}",
                int((time.perf_counter() - started) * 1000),
            ))
        return suite

    # ── routing ───────────────────────────────────────────────────────
    def run_routing(self) -> SuiteResult:
        """That an intent produces a plan naming the right agents and actions.

        Scored on the *plan*, not on execution, so routing regressions are
        isolated from data or model problems downstream.
        """
        suite = SuiteResult("routing", threshold=self.thresholds.get("routing", 80.0))
        cases = self.suite_cases("routing")
        if not cases:
            suite.skipped_reason = "No routing cases defined."
            return suite

        from freshsense.agents.planner_agent import PlannerAgent
        from freshsense.agents.state import AgentState, AgentTask

        planner = PlannerAgent()
        for case in cases:
            started = time.perf_counter()
            state = AgentState(run_id="EVAL", query=case.prompt)
            result = planner.execute(state, AgentTask(
                agent="planner", action="plan", params={"query": case.prompt}
            ))
            tasks = result.data.get("tasks", []) if result.data else []
            agents = {t.agent for t in tasks}
            actions = {t.action for t in tasks}

            checks, problems = [], []
            for expected in case.expected_agents:
                correct = expected in agents
                checks.append(correct)
                if not correct:
                    problems.append(f"missing agent {expected!r}")
            for expected in case.expected_actions:
                correct = expected in actions
                checks.append(correct)
                if not correct:
                    problems.append(f"missing action {expected!r}")

            score = sum(checks) / len(checks) if checks else 0.0
            suite.results.append(CaseResult(
                case.id, "routing",
                PASSED if score == 1.0 else FAILED,
                score,
                "; ".join(problems) or f"planned {sorted(agents)}",
                int((time.perf_counter() - started) * 1000),
            ))
        return suite

    # ── rag ───────────────────────────────────────────────────────────
    def run_rag(self) -> SuiteResult:
        """Groundedness and refusal against the document corpus."""
        suite = SuiteResult("rag", threshold=self.thresholds.get("rag", 70.0))
        cases = self.suite_cases("rag")
        if not cases:
            suite.skipped_reason = "No RAG cases defined."
            return suite

        try:
            from freshsense.rag.chain import get_rag_chain

            chain = get_rag_chain()
        except Exception as exc:                         # pragma: no cover
            suite.skipped_reason = f"RAG chain unavailable: {exc}"
            return suite

        if not chain.retriever.is_ready:
            suite.skipped_reason = (
                "The knowledge index has not been built. Run "
                "`python scripts/build_knowledge_base.py`, then re-run the "
                "evaluation."
            )
            return suite

        refusal_correct = 0
        refusal_total = 0

        for case in cases:
            started = time.perf_counter()
            answer = chain.answer(case.prompt, persist=False)
            latency = int((time.perf_counter() - started) * 1000)

            # Refusal is scored as a capability in its own right: answering an
            # out-of-corpus question is a failure, not a bonus.
            refusal_total += 1
            refused_correctly = answer.refused == case.should_refuse
            refusal_correct += int(refused_correctly)

            if case.should_refuse:
                suite.results.append(CaseResult(
                    case.id, "rag",
                    PASSED if refused_correctly else FAILED,
                    1.0 if refused_correctly else 0.0,
                    "correctly refused" if refused_correctly
                    else f"answered an out-of-corpus question "
                         f"(similarity {answer.max_similarity:.2f})",
                    latency,
                ))
                continue

            if answer.refused:
                suite.results.append(CaseResult(
                    case.id, "rag", FAILED, 0.0,
                    f"refused a covered question "
                    f"(similarity {answer.max_similarity:.2f})",
                    latency,
                ))
                continue

            text = answer.answer.lower()
            keywords = [k.lower() for k in case.expected_keywords]
            # Any keyword matching is enough: the corpus may phrase a figure as
            # a numeral or a word, and both are correct.
            matched = [k for k in keywords if k in text]
            grounded = bool(matched) if keywords else True

            source_ok = True
            if case.expected_source:
                sources = " ".join(
                    str(s.get("source", "")) for s in answer.sources
                ).lower()
                source_ok = case.expected_source.lower() in sources

            score = (0.7 if grounded else 0.0) + (0.3 if source_ok else 0.0)
            suite.results.append(CaseResult(
                case.id, "rag",
                PASSED if score >= 0.7 else FAILED,
                score,
                f"matched {matched or 'no'} keyword(s); "
                f"similarity {answer.max_similarity:.2f}",
                latency,
            ))

        suite.extra = {
            "refusal_accuracy": round(refusal_correct / max(refusal_total, 1) * 100, 1),
        }
        return suite

    # ── forecast ──────────────────────────────────────────────────────
    def run_forecast(self) -> SuiteResult:
        """Regression gate on model error and baseline superiority.

        Not case-driven: it scores the trained artefact itself, so a retrain
        that quietly degrades accuracy fails the suite before it reaches a user.
        """
        suite = SuiteResult("forecast", threshold=self.thresholds.get("forecast", 75.0))
        try:
            from freshsense.models.forecasting import get_forecast_service

            service = get_forecast_service()
        except Exception as exc:                         # pragma: no cover
            suite.skipped_reason = f"Forecast service unavailable: {exc}"
            return suite

        if not service.is_ready:
            suite.skipped_reason = (
                "No trained forecasting model. Run "
                "`python scripts/train_models.py`, then re-run the evaluation."
            )
            return suite

        metrics = service.metrics_frame()
        if metrics.empty:
            suite.skipped_reason = "The trained bundle holds no backtest metrics."
            return suite

        mean_mape = float(metrics["model_mape"].mean())
        beating_pct = float(metrics["beats_baseline"].mean() * 100)

        mape_ok = mean_mape <= FORECAST_MAX_MAPE
        baseline_ok = beating_pct >= FORECAST_MIN_BEATING_BASELINE

        suite.results.append(CaseResult(
            "forecast-mape", "forecast",
            PASSED if mape_ok else FAILED, 1.0 if mape_ok else 0.0,
            f"mean backtest error {mean_mape:.2f}% MAPE against a "
            f"{FORECAST_MAX_MAPE:.0f}% ceiling",
        ))
        suite.results.append(CaseResult(
            "forecast-baseline", "forecast",
            PASSED if baseline_ok else FAILED, 1.0 if baseline_ok else 0.0,
            f"{beating_pct:.1f}% of products beat seasonal-naive against a "
            f"{FORECAST_MIN_BEATING_BASELINE:.0f}% floor",
        ))

        # The weakest products are named, because a good mean can hide a
        # handful of models that should not be shipped at all.
        worst = metrics.nlargest(3, "model_mape")[["product", "model_mape"]]
        suite.extra = {
            "mean_mape": round(mean_mape, 2),
            "pct_beating_baseline": round(beating_pct, 1),
            "n_products": int(len(metrics)),
            "worst_products": worst.to_dict("records"),
        }
        return suite

    # ── agents ────────────────────────────────────────────────────────
    def run_agents(self) -> SuiteResult:
        """End-to-end orchestration: does a request complete, using the right agents?"""
        suite = SuiteResult("agents", threshold=self.thresholds.get("agents", 80.0))
        cases = self.suite_cases("agents")
        if not cases:
            suite.skipped_reason = "No agent cases defined."
            return suite

        try:
            coordinator = self.coordinator
        except Exception as exc:                         # pragma: no cover
            suite.skipped_reason = f"Coordinator unavailable: {exc}"
            return suite

        if coordinator.capability_gaps:
            suite.skipped_reason = (
                f"Agents unavailable: {', '.join(coordinator.capability_gaps)}."
            )
            return suite

        for case in cases:
            started = time.perf_counter()
            run = coordinator.run(case.prompt, session_id="EVAL")
            latency = int((time.perf_counter() - started) * 1000)

            checks, problems = [], []
            completed = run.status in ("completed", "awaiting_confirmation")
            checks.append(completed)
            if not completed:
                problems.append(f"status {run.status}")

            used = set(run.agents_used)
            for expected in case.expected_agents:
                correct = expected in used
                checks.append(correct)
                if not correct:
                    problems.append(f"{expected!r} not used")

            answered = bool(run.answer.strip())
            checks.append(answered)
            if not answered:
                problems.append("empty answer")

            score = sum(checks) / len(checks) if checks else 0.0
            suite.results.append(CaseResult(
                case.id, "agents",
                PASSED if score == 1.0 else FAILED,
                score,
                "; ".join(problems)
                or f"used {sorted(used)}, {run.replan_count} replan(s)",
                latency,
            ))
        return suite

    # ── orchestration ─────────────────────────────────────────────────
    def run_all(self, *, persist: bool = True) -> EvaluationReport:
        """Run every suite and optionally record the scores."""
        started = time.perf_counter()
        report = EvaluationReport()

        for name, runner in (
            ("nlu", self.run_nlu),
            ("routing", self.run_routing),
            ("rag", self.run_rag),
            ("forecast", self.run_forecast),
            ("agents", self.run_agents),
        ):
            try:
                report.suites[name] = runner()
            except Exception as exc:                     # pragma: no cover
                LOG.exception("Suite '%s' raised", name)
                suite = SuiteResult(name, threshold=self.thresholds.get(name, 80.0))
                suite.skipped_reason = f"Suite raised: {exc}"
                report.suites[name] = suite

        report.duration_ms = int((time.perf_counter() - started) * 1000)

        if persist:
            self.persist(report)

        LOG.info(
            "Evaluation complete in %d ms — overall %.1f%% (%s)",
            report.duration_ms, report.overall_score,
            "PASS" if report.passed else "FAIL",
        )
        return report

    def persist(self, report: EvaluationReport) -> int:
        """Record suite scores so the dashboard can show them over time."""
        scores = {
            f"eval_{name}": suite.score
            for name, suite in report.suites.items()
            if suite.status != SKIPPED
        }
        if not scores:
            return 0
        scores["eval_overall"] = report.overall_score
        try:
            return self.monitoring.record_model_metrics(
                "evaluation_harness", scores, dataset="fixed_eval_set"
            )
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Could not persist evaluation scores: %s", exc)
            return 0


def run_evaluation(*, persist: bool = True) -> EvaluationReport:
    """Convenience entry point used by ``scripts/run_eval.py`` and the UI."""
    return Evaluator().run_all(persist=persist)


__all__ = [
    "EvalCase", "CaseResult", "SuiteResult", "EvaluationReport", "Evaluator",
    "load_eval_set", "run_evaluation", "DEFAULT_CASES", "PASS_THRESHOLDS",
    "FORECAST_MAX_MAPE", "FORECAST_MIN_BEATING_BASELINE",
    "PASSED", "FAILED", "SKIPPED",
]