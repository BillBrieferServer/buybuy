"""
rate_limit.py
Simple in-memory sliding-window rate limiter for AI endpoints.
"""

import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding window rate limiter per key (typically IP address)."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        q = self._requests[key]

        # Remove expired entries
        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= self.max_requests:
            return False

        q.append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        q = self._requests[key]
        while q and q[0] < window_start:
            q.popleft()
        return max(0, self.max_requests - len(q))

    def seconds_until_next(self, key: str) -> float:
        """Seconds until the oldest request expires from the window."""
        q = self._requests[key]
        if not q:
            return 0
        return max(0, self.window_seconds - (time.time() - q[0]))
