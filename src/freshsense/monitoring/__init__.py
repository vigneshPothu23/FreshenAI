"""
FreshSense AI Monitoring Package.

Central exports for the monitoring subsystem.

The monitoring package provides:

• Runtime metrics
• System and business KPIs
• Drift monitoring
• Instrumentation and tracing
• Evaluation framework
"""

from __future__ import annotations

from freshsense.monitoring.metrics import (
    MonitoringService,
    MonitoringSnapshot,
    DriftSummary,
    HealthSummary,
)

from freshsense.monitoring.tracker import (
    Timer,
    track,
    track_block,
    instrument,
    configure_buffer,
    record_event,
)

from freshsense.monitoring.evaluator import (
    Evaluator,
    EvaluationReport,
    EvaluationSuite,
    EvaluationResult,
    EvalCase,
    run_evaluation,
)

__all__ = [
    # Metrics
    "MonitoringService",
    "MonitoringSnapshot",
    "DriftSummary",
    "HealthSummary",

    # Tracker
    "Timer",
    "track",
    "track_block",
    "instrument",
    "configure_buffer",
    "record_event",

    # Evaluator
    "Evaluator",
    "EvaluationReport",
    "EvaluationSuite",
    "EvaluationResult",
    "EvalCase",
    "run_evaluation",
]