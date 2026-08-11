"""Adaptive per-endpoint concurrency limit (AIMD on vLLM scheduler metrics).

Workers acquire a per-trajectory slot before the first LLM call and release after
the trajectory completes. The limit shrinks proportionally on preemption events
(or queue formation) and grows additively on trajectory completion when no
recent preempt has fired. The result is a self-tuning concurrency cap that sits
just below the vLLM endpoint's actual KV capacity.

Granularity rationale: vLLM's prefix cache holds a trajectory's KV between
turns (between LLM calls of the same conversation). So KV pressure scales with
the number of in-flight TRAJECTORIES, not the number of simultaneous LLM calls.
A per-trajectory acquire is what bounds KV occupancy correctly.

Acquire after setup (env creation + env_startup_command) and release before
cleanup (save_traj, env.cleanup) so the slot is held exactly for the duration
when vLLM has prefix cache for this conversation.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

logger = logging.getLogger("adaptive_limit")


class AdaptiveLLMLimit:
    """AIMD concurrency limiter driven by per-trajectory eviction reports.

    Each agent reports `evicted: bool` per turn (in `default.py:query()`),
    derived from `cached_tokens < 0.5 * expected_cached`. This is a direct
    causal signal — an active conversation's KV blocks were dropped — unlike
    proxies (hit_rate, KV%, preempts) which conflate eviction with cold
    starts, healthy warm caches, and downstream capacity events.

    Rule (per 5s poll window):
      - ≥3 evictions → shrink by 2 (severe burst)
      - ≥1 eviction → shrink by 1
      - 0 evictions for `grow_confirm_samples` consecutive polls → grow by 1
      - else hold

    The grow-confirm window subsumes the old preempt_cooldown_s: with default
    12 samples × 5s = 60s of sustained clean polls before we test upward.
    """

    def __init__(
        self,
        endpoint_url: str,
        initial: int = 64,
        min_limit: int = 25,
        max_limit: int = 72,
        poll_interval_s: float = 5.0,
        grow_confirm_samples: int = 12,   # 12 × 5s = 60s clean before grow
        severe_burst_evictions: int = 3,  # ≥this in one window = severe shrink (-2)
        shrink_cooldown_polls: int = 2,   # min poll windows between consecutive shrinks
        low_kv_skip_thresh: float = 0.30, # skip n=1 shrinks when kv below this
        preempt_shrink_factor: float = 0.90,  # preempt → limit = limit * this (10% drop)
        preempt_cooldown_polls: int = 6,  # 6 × 5s = 30s freeze after preempt
    ):
        """endpoint_url: e.g. "http://g188:33509/v1" — `/metrics` is appended."""
        self.endpoint_url = endpoint_url.rstrip("/")
        # Derive the bare-host metrics URL (vLLM exposes /metrics on the same port)
        # api_base typically ends in /v1; metrics is at /metrics (sibling).
        if self.endpoint_url.endswith("/v1"):
            base = self.endpoint_url[: -len("/v1")]
        else:
            base = self.endpoint_url
        self.metrics_url = f"{base}/metrics"
        # Extract host for no_proxy hint (cluster proxy chokes on internal hosts)
        self.host = self.endpoint_url.replace("http://", "").split(":", 1)[0]

        self.limit = initial
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.poll_interval_s = poll_interval_s
        self.grow_confirm_samples = grow_confirm_samples
        self.severe_burst_evictions = severe_burst_evictions
        self.shrink_cooldown_polls = shrink_cooldown_polls
        self.low_kv_skip_thresh = low_kv_skip_thresh
        self.preempt_shrink_factor = preempt_shrink_factor
        self.preempt_cooldown_polls = preempt_cooldown_polls
        # Cooldown counter: polls remaining before next shrink can fire.
        # Decremented each poll; set after any shrink (longer after preempt).
        self.shrink_cooldown_remaining = 0

        self.in_flight = 0
        self.cond = threading.Condition()

        # Eviction signal — agents call report_turn(evicted) per LLM turn.
        # Drained once per poll window.
        self._evictions_in_window = 0
        self._turns_in_window = 0
        self._signal_lock = threading.Lock()

        self.consecutive_clean = 0          # poll windows with zero evictions
        self.n_grows = 0
        self.n_shrinks = 0
        self.peak_limit = initial

        # Preempt backstop — rare but unambiguous capacity exhaustion. Tracked
        # via /metrics; baseline so we only react to NEW preempts.
        self.last_preempt_total = 0
        baseline = self._fetch_metrics()
        if baseline is not None:
            self.last_preempt_total = int(baseline.get("preempts", 0) or 0)

        self._watcher = threading.Thread(target=self._poll_loop, daemon=True)
        self._watcher.start()
        logger.info(
            f"AdaptiveLLMLimit({self.host}) start: limit={initial} "
            f"min={min_limit} max={max_limit} grow_confirm={grow_confirm_samples}"
        )

    def report_turn(self, evicted: bool) -> None:
        """Called by the agent loop after each LLM turn. `evicted=True` means
        the response showed `cached_tokens` significantly below what the prior
        turn's conversation should have left cached — a direct sign of KV
        block eviction for an in-flight conversation."""
        with self._signal_lock:
            self._turns_in_window += 1
            if evicted:
                self._evictions_in_window += 1

    @contextmanager
    def acquire(self):
        """Per-trajectory slot. Acquire after env setup, release after agent.run()."""
        with self.cond:
            while self.in_flight >= self.limit:
                self.cond.wait()
            self.in_flight += 1
        try:
            yield
        finally:
            with self.cond:
                self.in_flight -= 1
                # Growth happens in the poll loop, NOT here — otherwise with N
                # workers completing fast, growth fires N times per polling
                # interval and limit explodes faster than shrink can react.
                self.cond.notify(1)

    def _fetch_metrics(self) -> dict | None:
        """Pull cumulative cache hits/queries + preempts from vLLM /metrics."""
        import os
        original_no_proxy = os.environ.get("no_proxy", "")
        os.environ["no_proxy"] = (original_no_proxy + "," + self.host).lstrip(",")
        os.environ["NO_PROXY"] = os.environ["no_proxy"]
        try:
            req = urllib.request.Request(self.metrics_url)
            body = urllib.request.urlopen(req, timeout=4).read().decode()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return None
        hits = queries = preempts = kv = None
        for ln in body.splitlines():
            if ln.startswith("vllm:prefix_cache_hits_total{"):
                try: hits = float(ln.split()[-1])
                except (ValueError, IndexError): pass
            elif ln.startswith("vllm:prefix_cache_queries_total{"):
                try: queries = float(ln.split()[-1])
                except (ValueError, IndexError): pass
            elif ln.startswith("vllm:num_preemptions_total{"):
                try: preempts = float(ln.split()[-1])
                except (ValueError, IndexError): pass
            elif ln.startswith("vllm:kv_cache_usage_perc{"):
                # Gauge in [0, 1]. Multiple labels (per dp_rank, etc.); take
                # the max so we react to the most-pressured shard.
                try:
                    v = float(ln.split()[-1])
                    kv = v if kv is None else max(kv, v)
                except (ValueError, IndexError): pass
        return {"hits": hits, "queries": queries, "preempts": preempts, "kv": kv}

    def _poll_loop(self):
        """5s poll. Reads eviction counts reported by agents and reacts:
          - ≥severe_burst_evictions in window → shrink by 2
          - ≥1 eviction in window           → shrink by 1
          - 0 evictions for grow_confirm_samples consecutive windows → grow by 1

        Preempts (rare) are still treated as an emergency backstop on top.
        Hit-rate and KV are still fetched and logged for observability but no
        longer drive control — they were proxies that conflated eviction with
        cold-start bursts.
        """
        while True:
            try:
                time.sleep(self.poll_interval_s)

                # Drain reported evictions since last poll
                with self._signal_lock:
                    n_evict = self._evictions_in_window
                    n_turns = self._turns_in_window
                    self._evictions_in_window = 0
                    self._turns_in_window = 0

                # Observability snapshot (best-effort; doesn't drive control)
                m = self._fetch_metrics() or {}
                kv = m.get("kv")
                preempt_total = m.get("preempts") or 0
                new_preempts = int(preempt_total) - self.last_preempt_total
                self.last_preempt_total = int(preempt_total)

                # Decrement shrink cooldown each poll (regardless of evictions).
                if self.shrink_cooldown_remaining > 0:
                    self.shrink_cooldown_remaining -= 1

                # EMERGENCY backstop: vLLM had to preempt a request — true
                # capacity exhaustion. Drop limit to a fraction of current
                # (= 10% drop with default 0.90) and freeze further shrinks
                # for preempt_cooldown_polls windows. Cuts faster than the
                # per-window mechanism without overshooting to floor.
                if new_preempts > 0:
                    with self.cond:
                        target = int(self.limit * self.preempt_shrink_factor)
                        new_limit = max(self.min_limit, target)
                        if new_limit < self.limit:
                            logger.warning(
                                f"AdaptiveLLMLimit({self.host}) PREEMPT +{new_preempts}: "
                                f"limit {self.limit}→{new_limit} "
                                f"(cooldown {self.preempt_cooldown_polls} polls)"
                            )
                            self.limit = new_limit
                            self.n_shrinks += 1
                            self.cond.notify_all()
                    self.consecutive_clean = 0
                    self.shrink_cooldown_remaining = self.preempt_cooldown_polls
                    continue

                # PRIMARY: per-window eviction count from agent reports
                shrink = 0
                if n_evict >= self.severe_burst_evictions:
                    shrink = 2
                elif n_evict >= 1:
                    # Skip single-eviction shrinks at very low KV — those are
                    # typically a single bad request, not real pressure. Real
                    # cache thrashing requires the cache to actually be near full.
                    if n_evict == 1 and kv is not None and kv < self.low_kv_skip_thresh:
                        shrink = 0
                    else:
                        shrink = 1

                # Honor shrink cooldown: even if we want to shrink, wait until
                # the previous shrink has had time to take effect. Lets each
                # shrink absorb before potentially compounding (prevents the
                # cascading overshoot to floor on bursty evictions).
                if shrink > 0 and self.shrink_cooldown_remaining > 0:
                    self.consecutive_clean = 0   # bursty period, no growth
                    continue

                if shrink > 0:
                    with self.cond:
                        new_limit = max(self.min_limit, self.limit - shrink)
                        if new_limit < self.limit:
                            logger.warning(
                                f"AdaptiveLLMLimit({self.host}) EVICT n={n_evict}/{n_turns} "
                                f"kv={(kv or 0)*100:.0f}%: limit {self.limit}→{new_limit}"
                            )
                            self.limit = new_limit
                            self.n_shrinks += 1
                            self.cond.notify_all()
                    self.consecutive_clean = 0
                    self.shrink_cooldown_remaining = self.shrink_cooldown_polls
                else:
                    # Clean (or filtered) window. Only count toward growth if we
                    # actually saw some traffic (else we'd grow during idle periods).
                    if n_turns > 0:
                        self.consecutive_clean += 1
                    if (self.consecutive_clean >= self.grow_confirm_samples
                            and self.limit < self.max_limit):
                        with self.cond:
                            self.limit += 1
                            self.peak_limit = max(self.peak_limit, self.limit)
                            self.n_grows += 1
                            logger.warning(
                                f"AdaptiveLLMLimit({self.host}) GROW "
                                f"({self.consecutive_clean} clean polls, kv={(kv or 0)*100:.0f}%): "
                                f"limit {self.limit-1}→{self.limit}"
                            )
                            self.cond.notify(2)
                        self.consecutive_clean = 0   # reset after acting
            except Exception:
                logger.exception("AdaptiveLLMLimit poll loop error")
