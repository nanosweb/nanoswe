"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    time_limit: float = 0  # per-trajectory wall-clock cap (seconds); 0 = disabled. A
    # time-limited trajectory raises LimitsExceeded -> _submit_salvage, so its partial
    # diff is still captured (used by the online-OPD rollout to bound vLLM slot-holding).


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class ContextWindowExceeded(TerminatingException):
    """Raised when the LM call rejects the prompt for exceeding the context
    window. The harness handles this by force-submitting whatever the agent
    has produced so far (running the same submit command the agent would
    have issued itself), so partial work isn't lost as a graded failure.
    """


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._deadline = 0.0                 # set per-run from config.time_limit (0 = off)
        # Phase telemetry (seconds). Read by callers to attribute trajectory time.
        self.t_llm = 0.0
        self.t_bash = 0.0
        # Cache-eviction signal: track how many tokens of the prior conversation
        # the next turn SHOULD find cached on the endpoint. Compare against the
        # response's actual cached_tokens; if << expected, KV blocks for this
        # conversation got evicted between turns → real cache pressure signal
        # reported to model.limiter (see adaptive_limit.py for the AIMD rule).
        self._prev_total_tokens = 0          # prompt_tokens + completion_tokens of prior turn
        self._eviction_threshold = 0.5       # actual_cached < threshold * expected → evicted

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        # Backward-compatible alias: some templates expect `working_dir`; environments expose `cwd`
        if "cwd" in template_vars and "working_dir" not in template_vars:
            template_vars["working_dir"] = template_vars["cwd"]
        return Template(template).render(**kwargs, **template_vars, **self.extra_template_vars)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._deadline = (time.perf_counter() + self.config.time_limit) if self.config.time_limit else 0.0
        # Skip the system message when the configured template renders to empty.
        # Used by configs targeting models trained on boilerplate-stripped data
        # (no msg[0]=system, trajectory starts on user). Keeps eval-time prompt
        # byte-identical to training-time prompt.
        system_content = self.render_template(self.config.system_template)
        if system_content.strip():
            self.add_message("system", system_content)
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except (ContextWindowExceeded, LimitsExceeded) as e:
                # Don't lose the agent's work-in-progress: run the same submit
                # command the agent would have issued itself, capture the diff,
                # and treat it as a normal Submitted termination. Covers both
                # context exhaustion and hitting the step/turn cap (step_limit /
                # cost_limit) — a turn-capped trajectory is identifiable by
                # n_calls == step_limit.
                #
                # Stamp a parseable termination marker so downstream consumers
                # (e.g. the online-OPD rollout filter) can tell WHY a trajectory
                # ended — the inner salvage otherwise collapses step/ctx/time/cost
                # exits all into exit_status="Submitted". Eval-neutral: this message
                # is never sent back to the model (salvage runs a bash submit and
                # returns), and grading is patch-based.
                _reason = "context_window" if isinstance(e, ContextWindowExceeded) \
                    else getattr(self, "_limit_reason", "limit")
                self.add_message("user", f"[MSWEA_TERMINATION:{_reason}]")
                return self._submit_salvage(e)
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def _submit_salvage(self, e) -> tuple[str, str]:
        """Force the agent's submit command to capture WIP; return ("Submitted", diff).

        Shared by the ContextWindowExceeded and LimitsExceeded handlers so a
        forcibly-terminated trajectory still contributes its partial diff
        instead of being discarded and re-attempted on every future sweep.
        On submit failure returns ("<ExcName>SubmitFailed", details).
        """
        submit_cmd = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
        # Same timeout bump as execute_action: the runner-issued submit can hit
        # a big-repo `git add -A` that exceeds the default per-step timeout.
        _saved_timeout = getattr(self.env.config, "timeout", None)
        _submit_to = getattr(self.env.config, "submit_timeout", None)
        if _submit_to is None:
            _submit_to = getattr(self.env.config, "startup_timeout", _saved_timeout)
        if _submit_to is not None and _saved_timeout is not None:
            self.env.config.timeout = _submit_to
        # Free KV budget early (same rationale as the in-loop submit).
        try:
            limiter = getattr(self.model, "limiter", None)
            if limiter is not None and hasattr(limiter, "early_release"):
                limiter.early_release()
        except Exception:
            pass
        try:
            output = self.env.execute(submit_cmd)
            text = output.get("output", "") if isinstance(output, dict) else str(output)
            lines = text.lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() in [
                "MINI_SWE_AGENT_FINAL_OUTPUT",
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            ]:
                return "Submitted", "".join(lines[1:])
            return "Submitted", text
        except Exception as submit_err:
            return f"{type(e).__name__}SubmitFailed", f"{e!r} | submit: {submit_err!r}"
        finally:
            if _saved_timeout is not None:
                self.env.config.timeout = _saved_timeout

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        _reason = None
        if 0 < self.config.step_limit <= self.model.n_calls:
            _reason = "step_limit"
        elif 0 < self.config.cost_limit <= self.model.cost:
            _reason = "cost_limit"
        elif self._deadline and time.perf_counter() >= self._deadline:
            _reason = "time_limit"
        if _reason:
            self._limit_reason = _reason
            raise LimitsExceeded()
        try:
            t0 = time.perf_counter()
            response = self.model.query(self.messages, _deadline=(self._deadline or None))
            self.t_llm += time.perf_counter() - t0
        except Exception as e:
            # Detect "context window exceeded" by class name + message text.
            # Avoids a hard import dep on litellm here.
            cls = type(e).__name__
            msg = str(e)
            if (
                cls in {"ContextWindowExceededError"}
                or "ContextWindowExceededError" in msg
                or "context_length" in msg
                or "context length" in msg
                or "maximum input length" in msg
            ):
                raise ContextWindowExceeded(msg) from e
            # Per-traj wall-clock cap hit mid-turn — scheduler admission wait or the
            # deadline-bounded LLM request (incl. DeadlineReached / litellm Timeout).
            # Discard this partial generation and salvage instead of retrying/crashing.
            if self._deadline and time.perf_counter() >= self._deadline:
                self._limit_reason = "time_limit"
                raise LimitsExceeded() from e
            raise
        # Observability — eviction reporting must NEVER break the trajectory.
        # Pulled out of the model.query try-block (which is intended only for
        # ContextWindowExceededError detection) and bulletproofed below.
        try:
            self._report_cache_eviction()
        except Exception:
            pass
        self.add_message("assistant", **response)
        return response

    def _report_cache_eviction(self) -> None:
        """Compare actual cached_tokens against what should be cached from the
        prior turn. Report evicted=True to the per-endpoint limiter when the
        ratio is below the threshold. Skips turn 1 (cold start has nothing to
        compare against). Silent if the model didn't expose usage or if no
        limiter is wired up — both are OK degraded modes.
        """
        usage = getattr(self.model, "last_usage", None)
        if usage is None:
            return
        limiter = getattr(self.model, "limiter", None)
        if limiter is None or not hasattr(limiter, "report_turn"):
            return
        try:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        except Exception:
            return

        expected_cached = self._prev_total_tokens
        # Update for next call's expected (do this before early-return so the
        # value is fresh even for the cold-start turn).
        self._prev_total_tokens = prompt_tokens + completion_tokens

        if expected_cached <= 0:
            # First turn (cold start): no prior conversation to expect cached.
            # Don't generate a signal — would be a false positive.
            return

        evicted = cached < self._eviction_threshold * expected_cached
        limiter.report_turn(evicted=evicted)

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        t0 = time.perf_counter()
        # Detect the submission command and temporarily bump env.config.timeout
        # so `git add -A && git diff --cached` on a large repo (django ~30k
        # files) isn't truncated by the agent's per-step timeout. Mirrors the
        # env_startup_command bump in run/extra/swebench.py.
        #
        # Resolution order for the bumped timeout:
        #   1. env.config.submit_timeout (if set)
        #   2. env.config.startup_timeout (if set — same value used for
        #      env_startup_command; reasonable default since both phases hit
        #      the same big-repo .git operations)
        #   3. env.config.timeout (no bump)
        _is_submit = (
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action.get("action", "")
            or "MINI_SWE_AGENT_FINAL_OUTPUT" in action.get("action", "")
        )
        _saved_timeout = None
        if _is_submit:
            _saved_timeout = getattr(self.env.config, "timeout", None)
            _submit_to = getattr(self.env.config, "submit_timeout", None)
            if _submit_to is None:
                _submit_to = getattr(self.env.config, "startup_timeout", _saved_timeout)
            if _submit_to is not None and _saved_timeout is not None:
                self.env.config.timeout = _submit_to
            # Release KV budget early. The agent's LLM phase is done; the
            # subsequent `git add -A && git diff --cached` is pure I/O and
            # holds the slot for nothing. early_release() zeros this traj's
            # token contribution and wakes a waiter so the freed slot is
            # picked up immediately. Best-effort: agents without a limiter
            # (no token_scheduler config) just skip.
            try:
                limiter = getattr(self.model, "limiter", None)
                if limiter is not None and hasattr(limiter, "early_release"):
                    limiter.early_release()
            except Exception:
                pass
        # Hard wall-clock cap: bound a non-submit command to the time left so the
        # deadline can interrupt it; if already past, don't run it at all (discard
        # the turn). The submit command is never bounded — it must complete to
        # capture the WIP diff for the salvage.
        if self._deadline and not _is_submit:
            _rem = self._deadline - time.perf_counter()
            if _rem <= 0:
                self._limit_reason = "time_limit"
                raise LimitsExceeded()
            _cur = getattr(self.env.config, "timeout", None)
            if _cur is not None:
                _saved_timeout = _cur
                self.env.config.timeout = min(_cur, _rem)
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            self.t_bash += time.perf_counter() - t0
            if self._deadline and time.perf_counter() >= self._deadline:
                self._limit_reason = "time_limit"   # deadline cut the command -> salvage now
                raise LimitsExceeded()
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            self.t_bash += time.perf_counter() - t0
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        finally:
            if _saved_timeout is not None:   # restore for submit-bump OR deadline-bound cmd
                self.env.config.timeout = _saved_timeout
        self.t_bash += time.perf_counter() - t0
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            diff = "".join(lines[1:])
            # Robust submit (opt-in via MSWEA_ROBUST_SUBMIT=1). Small models
            # sometimes emit a malformed submit command — e.g. a bare `git add`
            # with no `-A`, which stages nothing, so `git diff --cached` is
            # empty even though the model DID edit files. Trusting the model's
            # stdout then records an empty patch and grades a solved task as
            # no_patch — a seed-dependent artifact that confounds cross-run
            # comparison. When enabled, re-extract the diff server-side with the
            # canonical command (same as _submit_salvage); this is idempotent
            # for models that already submit correctly, and rescues the rest.
            if os.environ.get("MSWEA_ROBUST_SUBMIT", "0") == "1":
                _saved = getattr(self.env.config, "timeout", None)
                _to = getattr(self.env.config, "submit_timeout", None) or getattr(
                    self.env.config, "startup_timeout", _saved
                )
                try:
                    if _to is not None and _saved is not None:
                        self.env.config.timeout = _to
                    canon = self.env.execute("git add -A && git diff --cached")
                    ctext = canon.get("output", "") if isinstance(canon, dict) else str(canon)
                    if ctext.strip():
                        diff = ctext
                except Exception:
                    pass
                finally:
                    if _saved is not None:
                        self.env.config.timeout = _saved
            raise Submitted(diff)
