"""HTTP client with throttle, retry-on-transient-error and shared session.

Hand-rolled to avoid a `tenacity` dependency. Retries on:
- network errors (`requests.exceptions.ConnectionError`, `Timeout`)
- HTTP 429 (Too Many Requests) and 5xx responses

Backoff is exponential with optional jitter, capped at `max_sleep`.
Inter-request throttle is a hard floor: the next request waits at least
`min_interval_seconds` since the last successful response.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "mibel-derivatives/0.1 (research; +https://github.com/cvilches-mibel)"
)
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class ThrottledSession:
    """Thin wrapper around `requests.Session` with throttling and retries."""

    user_agent: str = DEFAULT_USER_AGENT
    min_interval_seconds: float = 1.0
    max_retries: int = 4
    initial_backoff: float = 1.5
    max_sleep: float = 30.0
    timeout: float = 30.0
    _session: requests.Session = field(init=False)
    _last_request_monotonic: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def _wait_throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_monotonic
        deficit = self.min_interval_seconds - elapsed
        if deficit > 0:
            time.sleep(deficit)

    def _sleep_backoff(self, attempt: int) -> None:
        base = self.initial_backoff * (2 ** (attempt - 1))
        jitter = random.uniform(0, base * 0.25)
        time.sleep(min(self.max_sleep, base + jitter))

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        """GET with retry. Raises on the last failed attempt."""
        last_exc: Exception | None = None
        last_resp: requests.Response | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_throttle()
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout if timeout is not None else self.timeout,
                )
                self._last_request_monotonic = time.monotonic()
                if resp.status_code in RETRYABLE_STATUS:
                    last_resp = resp
                    logger.warning(
                        "GET %s -> %d (attempt %d/%d), backing off",
                        url, resp.status_code, attempt, self.max_retries,
                    )
                    if attempt < self.max_retries:
                        self._sleep_backoff(attempt)
                    continue
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                self._last_request_monotonic = time.monotonic()
                logger.warning(
                    "GET %s raised %s (attempt %d/%d)",
                    url, type(exc).__name__, attempt, self.max_retries,
                )
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
        if last_resp is not None:
            last_resp.raise_for_status()
            return last_resp
        assert last_exc is not None
        raise last_exc
