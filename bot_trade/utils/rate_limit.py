from __future__ import annotations
"""Simple shared rate limiter with weighted buckets and retry backoff.

This module implements a token bucket limiter used by sandbox gateways to
respect exchange rate limits. The limiter is intentionally lightweight and
synchronous; it tracks a single bucket with configurable capacity and refill
interval. Each acquisition may specify a weight representing the cost of a
request.

A convenience :func:`retry` helper retries callables when exchanges respond
with temporary rate limit errors (HTTP 429/418/1003).
"""

from dataclasses import dataclass, field
import random
import threading
import time
from typing import Callable, Dict


@dataclass
class RateLimitBucket:
    capacity: int
    refill_time: float
    tokens: float = field(init=False)
    updated: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def refill(self) -> None:
        now = time.time()
        delta = now - self.updated
        if delta <= 0:
            return
        self.tokens = min(
            self.capacity,
            self.tokens + (delta / self.refill_time) * self.capacity,
        )
        self.updated = now

    def consume(self, weight: int) -> bool:
        self.refill()
        if self.tokens >= weight:
            self.tokens -= weight
            return True
        return False


class RateLimiter:
    """Token bucket rate limiter with blocking acquire."""

    def __init__(self, capacity: int, refill_time: float = 1.0) -> None:
        self.bucket = RateLimitBucket(capacity=capacity, refill_time=refill_time)
        self._lock = threading.Lock()

    def acquire(self, weight: int = 1) -> None:
        backoff = 0.1
        while True:
            with self._lock:
                if self.bucket.consume(weight):
                    return
            time.sleep(backoff + random.random() * 0.1)
            backoff = min(backoff * 2, 1.0)


def retry(
    func: Callable[[], Dict[str, int]],
    *,
    is_retryable: Callable[[Dict[str, int]], bool],
    max_attempts: int = 5,
    base_delay: float = 0.5,
) -> Dict[str, int]:
    """Retry helper with exponential backoff and jitter."""
    attempt = 0
    had_retry = False
    while True:
        resp = func()
        retryable = is_retryable(resp)
        if not retryable or attempt >= max_attempts - 1:
            if had_retry and not retryable:
                print("RLIMIT_CLEAR")
            return resp
        had_retry = True
        sleep = base_delay * (2 ** attempt) + random.random() * 0.1
        print(f"RLIMIT_SLEEP={sleep:.2f}s")
        time.sleep(sleep)
        attempt += 1


__all__ = ["RateLimiter", "retry"]
