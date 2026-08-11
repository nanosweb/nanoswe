"""Lean OpenAI-compatible client for a single local vLLM endpoint.

WHY THIS EXISTS
---------------
litellm abstracts 100+ API providers; the OPD rollout (and SWE-eval) only ever POST
to ONE vLLM `/v1/chat/completions`. litellm's per-call Python — provider routing,
pydantic response coercion, cost lookup against a multi-MB price map, logging callbacks
— is CPU-bound and GIL-serialized. Measured against a warm vLLM endpoint, 91 concurrent
first-queries: litellm p50 latency 3.6s vs raw HTTP 0.4s (~8x), scaling linearly with
concurrency (0.40s @ N=1 -> 1.54s @ N=30 -> 3.61s @ N=91) — the GIL signature. That
overhead IS the rollout's round-start ramp and a per-turn tax on every query. vLLM itself
serves all 91 concurrently in ~0.4s (raw client, run reaches ~N, wait==0); the bottleneck
is purely litellm client-side.

This class drops all of it: a shared httpx.Client POST, minimal parse, cost==0.

PARITY (same YAML the old litellm client used still works):
  * Error classification raises vendored exception shims (models/exceptions.py) whose
    class NAMES match litellm's. The agent classifies context-window by
    `type(e).__name__ == "ContextWindowExceededError"` (agents/default.py), so this
    gives a byte-identical `context_window` termination — with no litellm import.
  * Scheduler hooks (limiter.before_query / after_query) preserved verbatim — same
    per-endpoint TokenScheduler singleton, same acquire() call site in swebench.py.
  * Hard wall-clock deadline (DeadlineReached) preserved: request timeout is bounded to
    the remaining deadline; DeadlineReached is retry-excluded so backoff can't overshoot.
  * last_usage exposes .prompt_tokens / .completion_tokens / .prompt_tokens_details
    .cached_tokens for the agent's cache-eviction observability.

Opt-in via `model_class: vllm`. Default model selection is unchanged (LitellmModel).
"""
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
# Vendored exception shims (names match litellm) so the eval path never imports litellm.
from minisweagent.models.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout as LiteLLMTimeout,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.adaptive_limit import AdaptiveLLMLimit
from minisweagent.models.token_scheduler import DeadlineReached, TokenScheduler

logger = logging.getLogger("vllm_model")

# One shared connection pool per process. Limits are set high so the pool never
# serializes requests (vLLM serves 91-concurrent fine; the raw-HTTP burst measured
# ~0.4s flat at N=91). trust_env=False: ignore HTTP(S)_PROXY — we hit a local socket.
_CLIENT: "httpx.Client | None" = None
_CLIENT_LOCK = threading.Lock()


def _client() -> "httpx.Client":
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(
                    limits=httpx.Limits(max_connections=512, max_keepalive_connections=512),
                    timeout=httpx.Timeout(600.0),
                    trust_env=False,
                )
    return _CLIENT


def _strip_provider(model_name: str) -> str:
    """litellm wants 'hosted_vllm/<id>'; the raw API wants '<id>'."""
    for p in ("hosted_vllm/", "openai/"):
        if model_name.startswith(p):
            return model_name[len(p):]
    return model_name


class _Details:
    __slots__ = ("cached_tokens",)

    def __init__(self, d: dict):
        self.cached_tokens = int((d or {}).get("cached_tokens", 0) or 0)


class _Usage:
    """Mimics the litellm usage object the agent loop reads (getattr-based)."""
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details")

    def __init__(self, d: dict):
        d = d or {}
        self.prompt_tokens = int(d.get("prompt_tokens", 0) or 0)
        self.completion_tokens = int(d.get("completion_tokens", 0) or 0)
        self.total_tokens = int(d.get("total_tokens", 0) or 0)
        ptd = d.get("prompt_tokens_details")
        self.prompt_tokens_details = _Details(ptd) if ptd else None


@dataclass
class VLLMModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # Accepted-but-ignored so the same YAML loads under either model class:
    litellm_model_registry: Path | str | None = None
    adaptive_limit: dict[str, Any] | None = None
    token_scheduler: dict[str, Any] | None = None


class VLLMModel:
    # Per-(api_base) limiter singleton — one TokenScheduler per vLLM endpoint per
    # process, exactly as LitellmModel. swebench.py still does `with model.limiter.acquire():`.
    _limiters: dict[str, Any] = {}
    _limiters_lock = threading.Lock()

    def __init__(self, **kwargs):
        self.config = VLLMModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.last_usage = None
        self._served_model = _strip_provider(self.config.model_name)
        mk = self.config.model_kwargs if isinstance(self.config.model_kwargs, dict) else {}
        api_base = mk.get("api_base")
        self._url = (api_base.rstrip("/") + "/chat/completions") if api_base else None
        self._api_key = mk.get("api_key") or "x"           # vLLM ignores it; send a dummy
        self._temperature = mk.get("temperature", 0.0)
        self._max_tokens = mk.get("max_tokens")
        self._cfg_timeout = mk.get("timeout")              # per-request cap (s), e.g. 90
        self.limiter: Any = None
        if api_base:
            with VLLMModel._limiters_lock:
                if api_base in VLLMModel._limiters:
                    self.limiter = VLLMModel._limiters[api_base]
                elif self.config.token_scheduler is not None:
                    kw = dict(self.config.token_scheduler or {})
                    kw.setdefault("model_name", self.config.model_name)
                    self.limiter = TokenScheduler(endpoint_url=api_base, **kw)
                    VLLMModel._limiters[api_base] = self.limiter
                elif self.config.adaptive_limit is not None:
                    self.limiter = AdaptiveLLMLimit(endpoint_url=api_base, **(self.config.adaptive_limit or {}))
                    VLLMModel._limiters[api_base] = self.limiter

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                NotFoundError,
                PermissionDeniedError,
                ContextWindowExceededError,
                APIError,
                AuthenticationError,
                DeadlineReached,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], _deadline: float | None = None, **kwargs):
        # request timeout = min(remaining deadline, configured cap). Deadline is
        # retry-excluded so the 4-60s backoff can't overshoot the per-traj wall-clock cap.
        timeout = self._cfg_timeout
        if _deadline is not None:
            rem = _deadline - time.perf_counter()
            if rem <= 0:
                raise DeadlineReached()
            timeout = rem if timeout is None else min(timeout, rem)
        to = float(timeout) if timeout is not None else 600.0
        payload = {
            "model": self._served_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
        }
        # Optional deterministic sampling: NANOSWE_SEED pins vLLM's per-request RNG so
        # results are reproducible run-to-run (removes the unseeded arrival-order-RNG
        # nondeterminism that makes pass@1 timing/load-dependent). Set per-eval to sweep.
        import os as _os
        _seed = _os.environ.get("NANOSWE_SEED")
        if _seed not in (None, ""):
            payload["seed"] = int(_seed)
        if self._max_tokens is not None:
            payload["max_tokens"] = kwargs.get("max_tokens", self._max_tokens)
        try:
            r = _client().post(self._url, json=payload,
                               headers={"Authorization": f"Bearer {self._api_key}"}, timeout=to)
        except httpx.TimeoutException as e:
            raise LiteLLMTimeout(message=f"Request timed out: {e}", model=self.config.model_name,
                                 llm_provider="hosted_vllm") from e
        if r.status_code >= 400:
            body = r.text or ""
            bl = body.lower()
            if ("maximum context length" in bl or "context length is only" in bl
                    or "maximum input length" in bl or "context_length" in bl):
                raise ContextWindowExceededError(model=self.config.model_name,
                                                 llm_provider="hosted_vllm", message=body)
            if r.status_code == 401:
                raise AuthenticationError(message=body, model=self.config.model_name, llm_provider="hosted_vllm")
            if r.status_code == 429:
                raise RateLimitError(message=body, model=self.config.model_name, llm_provider="hosted_vllm")
            if r.status_code == 400:
                raise BadRequestError(message=body, model=self.config.model_name, llm_provider="hosted_vllm")
            raise APIError(status_code=r.status_code, message=body,
                           model=self.config.model_name, llm_provider="hosted_vllm")
        return r.json()

    def query(self, messages: list[dict[str, str]], _deadline: float | None = None, **kwargs) -> dict:
        self.last_usage = None
        if self.limiter is not None and hasattr(self.limiter, "before_query"):
            try:
                self.limiter.before_query(deadline=_deadline)
            except DeadlineReached:
                raise
            except Exception:
                pass
        resp = self._query(messages, _deadline=_deadline, **kwargs)
        self.n_calls += 1
        usage = resp.get("usage") if isinstance(resp, dict) else None
        self.last_usage = _Usage(usage) if usage else None
        if self.limiter is not None and hasattr(self.limiter, "after_query"):
            try:
                pt = int((usage or {}).get("prompt_tokens", 0) or 0)
                ct = int((usage or {}).get("completion_tokens", 0) or 0)
                self.limiter.after_query(pt, ct, messages=list(messages))
            except Exception:
                pass
        GLOBAL_MODEL_STATS.add(0.0)
        try:
            content = resp["choices"][0]["message"]["content"] or ""
        except Exception:
            content = ""
        return {"content": content}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
