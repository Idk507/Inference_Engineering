"""A dependency-free simulator for continuous (in-flight) LLM batching.

It models the scheduler control plane, not transformer math.  At every
iteration it retires finished work, keeps ongoing decode work responsive, and
uses any remaining token/KV-cache budget to admit or continue prompt prefills.
Run ``python continuous_batching_scheduler.py`` to see an execution trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional


class RequestState(str, Enum):
    """Lifecycle states owned by the scheduler."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Request:
    """One generation request and its scheduler-visible progress.

    ``prompt_tokens`` is the amount of uncached prompt work. ``max_new_tokens``
    is deliberately deterministic in this simulator; a real engine would stop
    on EOS, stop strings, token limits, cancellation, or tool-call boundaries.
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_iteration: int = 0
    state: RequestState = RequestState.WAITING
    prompt_processed: int = 0
    output_tokens: int = 0
    generated: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None

    @property
    def remaining_prompt(self) -> int:
        """Return prompt tokens that still require prefill computation."""
        return self.prompt_tokens - self.prompt_processed

    @property
    def in_decode(self) -> bool:
        """Return whether prefill is complete and generation remains."""
        return self.remaining_prompt == 0 and self.output_tokens < self.max_new_tokens

    @property
    def needs_kv_slots(self) -> int:
        """Return simulated KV slots currently retained by this request."""
        # Prompt tokens and emitted tokens both occupy causal-attention history.
        return self.prompt_processed + self.output_tokens


@dataclass(frozen=True)
class ScheduledWork:
    """A request's work selected for one GPU iteration."""

    request_id: str
    phase: str  # "decode" or "prefill"
    token_count: int


@dataclass
class SchedulerConfig:
    """Hard limits evaluated during every scheduling iteration."""

    max_running_requests: int = 3
    max_batched_tokens: int = 10
    max_kv_slots: int = 100
    prefill_chunk_size: int = 8

    def __post_init__(self) -> None:
        """Reject invalid capacity settings early."""
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; received {value}")


class ContinuousBatchScheduler:
    """Token-budgeted continuous batching scheduler.

    Scheduling policy: retain active decodes first, advance active prefills
    second, then admit FIFO waiting requests using chunked prefill. This policy
    avoids interrupting streamed responses merely because a new long prompt
    arrived. It deliberately uses a conservative KV admission check: a request
    is admitted only if its full prompt plus maximum output can fit in cache.
    That makes this example ``guaranteed no evict`` rather than a preemptive
