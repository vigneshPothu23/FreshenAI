"""Sprint 6 — instrumentation.

Feeds ``system_logs``, which is what the latency, success-rate and component
breakdowns in :mod:`freshsense.monitoring.metrics` are computed from.

Four rules shape this module, each learned from a way instrumentation normally
goes wrong:

**Observability must never break the thing it observes.** Every failure inside
the tracker is swallowed. An unreachable database or a malformed metadata
payload must not turn a working forecast into a stack trace.

**It must not recurse.** Writing an event calls the repository, which — if the
repository were itself instrumented — would write an event. A thread-local
guard makes re-entrant tracking a no-op rather than a stack overflow.

**It must not dominate the cost of what it measures.** A synchronous insert per
call would make a 2 ms repository read a 3 ms one, which changes the number
being reported. Events are buffered and flushed in batches, with an ``atexit``
hook so nothing is lost at shutdown.

**It must work without editing the code it measures.** :func:`instrument` wraps
methods on an existing object in place, so services written before monitoring
existed can be traced without touching them.
"""

from __future__ import annotations

import atexit
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, TypeVar

from freshsense.config import SETTINGS
from freshsense.db.repository import MonitoringRepository
from freshsense.db.session import Database
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

#: Re-entrancy guard. Set while an event is being written, so instrumentation
#: triggered by the write itself is ignored.
_LOCAL = threading.local()


def _reentrant() -> bool:
    return getattr(_LOCAL, "writing", False)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Timer:
    """Elapsed-time context manager. Pure measurement, no persistence.

    Useful where a duration is needed for a narrative rather than for the log::

        with Timer() as t:
            result = expensive()
        print(f"took {t.elapsed_ms} ms")
    """

    started: float = 0.0
    stopped: float = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stopped = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        end = self.stopped or time.perf_counter()
        return int((end - self.started) * 1000)


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class EventBuffer:
    """Batches system events and flushes them periodically.

    Thread-safe. Flushes when the buffer reaches ``max_size``, when
    ``max_age_seconds`` has elapsed since the last flush, or at interpreter
    exit. Errors are absorbed: a monitoring backlog is preferable to a failed
    user request.
    """

    max_size: int = 25
    max_age_seconds: float = 30.0
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_flush: float = field(default_factory=time.monotonic, repr=False)
    _repository: MonitoringRepository | None = field(default=None, repr=False)
    dropped: int = 0

    @property
    def repository(self) -> MonitoringRepository:
        if self._repository is None:
            self._repository = MonitoringRepository()
        return self._repository

    def bind(self, db: Database) -> "EventBuffer":
        """Point the buffer at a specific database. Used by tests."""
        self._repository = MonitoringRepository(db)
        return self

    def add(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            due = (
                len(self.events) >= self.max_size
                or (time.monotonic() - self._last_flush) >= self.max_age_seconds
            )
        if due:
            self.flush()

    def flush(self) -> int:
        """Write buffered events. Returns the number persisted."""
        with self._lock:
            pending, self.events = self.events, []
            self._last_flush = time.monotonic()

        if not pending:
            return 0

        _LOCAL.writing = True
        try:
            for event in pending:
                self.repository.log_event(**event)
            return len(pending)
        except Exception as exc:                         # pragma: no cover
            self.dropped += len(pending)
            LOG.debug("Dropped %d monitoring event(s): %s", len(pending), exc)
            return 0
        finally:
            _LOCAL.writing = False

    @property
    def pending(self) -> int:
        return len(self.events)


#: Process-wide buffer. Flushed at exit so a short-lived script still records.
BUFFER = EventBuffer()
atexit.register(BUFFER.flush)


def configure_buffer(
    *, max_size: int | None = None, max_age_seconds: float | None = None,
    db: Database | None = None,
) -> EventBuffer:
    """Adjust buffering, or bind it to a specific database.

    Setting ``max_size=1`` makes tracking synchronous, which tests want and
    production does not.
    """
    if max_size is not None:
        BUFFER.max_size = int(max_size)
    if max_age_seconds is not None:
        BUFFER.max_age_seconds = float(max_age_seconds)
    if db is not None:
        BUFFER.bind(db)
    return BUFFER


def record_event(
    *,
    component: str,
    action: str,
    status: str = "ok",
    latency_ms: int = 0,
    detail: str = "",
    metadata: Any = None,
) -> None:
    """Queue one system event. Never raises."""
    if _reentrant():
        return
    try:
        BUFFER.add({
            "component": component,
            "action": action,
            "status": status,
            "latency_ms": int(latency_ms),
            "detail": str(detail)[:500],
            "metadata": metadata,
        })
    except Exception:                                    # pragma: no cover
        pass


# ══════════════════════════════════════════════════════════════════════════
@contextmanager
def track_block(
    component: str,
    action: str,
    *,
    detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Time and record an arbitrary block of work.

    Yields a mutable context dictionary the block may write to, so the recorded
    detail can describe the *outcome* rather than only the intent::

        with track_block("pipeline", "clean_inventory") as ctx:
            frame = clean(raw)
            ctx["detail"] = f"{len(frame)} rows retained"

    An exception is recorded as a failure and then re-raised — the caller still
    sees it.
    """
    context: dict[str, Any] = {
        "detail": detail, "metadata": dict(metadata or {}), "status": "ok"
    }
    started = time.perf_counter()
    try:
        yield context
    except Exception as exc:
        context["status"] = "error"
        context["detail"] = f"{type(exc).__name__}: {exc}"[:500]
        record_event(
            component=component, action=action, status="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=context["detail"], metadata=context["metadata"],
        )
        raise
    else:
        record_event(
            component=component, action=action, status=context["status"],
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=context["detail"], metadata=context["metadata"],
        )


def track(
    component: str,
    action: str | None = None,
    *,
    describe: Callable[[Any], str] | None = None,
    sample_rate: float = 1.0,
) -> Callable[[F], F]:
    """Decorator recording latency and success for a callable.

    Args:
        component: Logical subsystem, e.g. ``"forecast"`` or ``"pipeline"``.
        action: Operation name. Defaults to the function's own name.
        describe: Optional function mapping the return value to a detail string.
            Any exception it raises is ignored — a description helper must never
            be able to fail the call it describes.
        sample_rate: Fraction of calls to record, for hot paths. Latency
            percentiles remain valid under uniform sampling; call counts do not,
            so the sampled rate is recorded in the metadata.

    The wrapped function's exceptions propagate unchanged; only the recording is
    suppressed on failure.
    """
    def decorator(function: F) -> F:
        operation = action or function.__name__
        counter = {"n": 0}

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counter["n"] += 1
            # Deterministic sampling: every Nth call rather than a random draw,
            # so a test can predict what gets recorded.
            step = max(1, int(round(1.0 / max(sample_rate, 1e-9))))
            should_record = (counter["n"] % step) == 0

            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                if should_record:
                    record_event(
                        component=component, action=operation, status="error",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise

            if should_record:
                detail = ""
                if describe is not None:
                    try:
                        detail = str(describe(result))[:500]
                    except Exception:
                        detail = ""
                record_event(
                    component=component, action=operation, status="ok",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    detail=detail,
                    metadata={"sample_rate": sample_rate} if sample_rate < 1 else None,
                )
            return result

        return wrapper                                   # type: ignore[return-value]

    return decorator


def instrument(
    target: Any,
    component: str,
    methods: Sequence[str],
    *,
    sample_rate: float = 1.0,
) -> Any:
    """Wrap named methods on an existing object with tracking, in place.

    Lets services written before monitoring existed be traced without editing
    them — which matters when those files are already reviewed and shipped.
    Unknown or already-instrumented methods are skipped rather than raising, so
    the call is safe to make unconditionally at start-up.

    Returns:
        The same object, for chaining.
    """
    for name in methods:
        original = getattr(target, name, None)
        if not callable(original) or getattr(original, "_freshsense_tracked", False):
            continue
        wrapped = track(component, name, sample_rate=sample_rate)(original)
        wrapped._freshsense_tracked = True               # type: ignore[attr-defined]
        try:
            setattr(target, name, wrapped)
        except AttributeError:                           # pragma: no cover
            LOG.debug("Could not instrument %s.%s (read-only attribute)",
                      component, name)
    return target


# ══════════════════════════════════════════════════════════════════════════
def prune_old_logs(db: Database | None = None, *, retention_days: int | None = None) -> int:
    """Delete system events older than the configured retention window.

    Without this the log grows without bound and the dashboard's percentile
    queries slow down over the life of the deployment.

    Returns:
        The number of rows removed.
    """
    days = int(retention_days or SETTINGS.monitoring.get("retention_days", 90))
    from freshsense.db.session import get_database

    database = db or get_database()
    removed = database.execute(
        "DELETE FROM system_logs WHERE created_at < datetime('now', ?)",
        (f"-{days} days",),
    )
    if removed:
        LOG.info("Pruned %d system log row(s) older than %d day(s)", removed, days)
    return removed


def instrumentation_status() -> dict[str, Any]:
    """Buffer state, for the Settings and Monitoring pages."""
    return {
        "pending_events": BUFFER.pending,
        "flush_threshold": BUFFER.max_size,
        "flush_interval_seconds": BUFFER.max_age_seconds,
        "dropped_events": BUFFER.dropped,
    }


__all__ = [
    "Timer", "EventBuffer", "BUFFER", "configure_buffer", "record_event",
    "track", "track_block", "instrument", "prune_old_logs",
    "instrumentation_status",
]