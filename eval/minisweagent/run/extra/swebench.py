#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from jinja2 import Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.grading import grade_instance, is_valid_patch
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "verified_cluster": 'ricdomolm/SWE-bench_Verified-Working-Harbor',
    # 477-instance subset (446 - 6 broken + 33 matplotlib + 4 proxy-rescued).
    # Production grading must set SWE_INJECT_PROXY=1 for the 4 proxy_required
    # instances (pylint-4661, sphinx-10435, sphinx-7985, matplotlib-20488) to
    # resolve. See nanoswe/runs/swe_eval/expand500/v477_ids.json.
    "verified_cluster_477": 'ricdomolm/SWE-bench_Verified-Cluster477',
    # 483-instance subset (477 + 6 rescued via inweriok-corrected patches +
    # F2P fix for django-7530 + SKIPPED-rescue harness patch for pylint-6528/7277).
    # Production grading must use the patched SWE-bench-upstream install
    # (pip install -e from /lustre/home/rolmedo/SWE-bench-upstream) for the
    # SKIPPED rescue to apply, in addition to SWE_INJECT_PROXY=1 for the 4
    # proxy_required instances.
    "verified_cluster_483": 'ricdomolm/SWE-bench_Verified-Cluster483',
    # "smith_harbor": 'ricdomolm/SWE-smith-trajectories-harbor-found-235B',
    "smith_harbor": '/fast/rolmedo/SWE-smith-trajectories-harbor-found-235B',
    'smith_og': '/fast/rolmedo/SWE-smith-trajectories-harbor-found',
    # validated_f2p=True AND leak_suspected=False, 80,586 instances; image_name
    # is a direct .sif path consumed by --environment-class singularity-localimage.
    'smith_v1_2026_05_23_dd': '/fast/rolmedo/swesmith/datasets/v1_2026-05-23_dd',
    # v2: 91,735 instances (strict superset of v1; +11,149 new spanning 57 new repos).
    'smith_v2_2026_05_24_dd': '/fast/rolmedo/swesmith/datasets/v2_2026-05-24_dd',
    'smith_v3_2026_05_25_dd': '/fast/rolmedo/swesmith/datasets/v3_2026-05-25_dd',
}


_OUTPUT_FILE_LOCK = threading.Lock()


# Per-process phase telemetry. Each worker writes one (t_wall, t_llm, t_bash)
# tuple per completed trajectory; every PHASE_EMIT_N completions we log a
# rolling summary so we can see whether the bottleneck is LLM, bash exec, or
# setup/other. Output line goes to stderr via the standard logger.
PHASE_EMIT_N = 25
_PHASE_LOCK = threading.Lock()
_PHASE_STATE = {"n": 0, "t_wall": 0.0, "t_llm": 0.0, "t_bash": 0.0, "n_turns": 0}


def _record_phase(t_wall: float, t_llm: float, t_bash: float, n_turns: int) -> None:
    with _PHASE_LOCK:
        _PHASE_STATE["n"] += 1
        _PHASE_STATE["t_wall"] += t_wall
        _PHASE_STATE["t_llm"] += t_llm
        _PHASE_STATE["t_bash"] += t_bash
        _PHASE_STATE["n_turns"] += n_turns
        if _PHASE_STATE["n"] % PHASE_EMIT_N == 0:
            n = _PHASE_STATE["n"]
            tw = _PHASE_STATE["t_wall"]
            tl = _PHASE_STATE["t_llm"]
            tb = _PHASE_STATE["t_bash"]
            to = max(0.0, tw - tl - tb)
            pct = lambda x: 100.0 * x / tw if tw > 0 else 0.0
            logger.warning(
                f"WORKER_PHASES n={n}  "
                f"avg_wall={tw / n:.1f}s  avg_turns={_PHASE_STATE['n_turns'] / n:.1f}  "
                f"t_llm={pct(tl):.0f}%  t_bash={pct(tb):.0f}%  t_other={pct(to):.0f}%"
            )


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if not image_name.startswith("harbor.is.localnet/"):
        image_name = "harbor.is.localnet/" + image_name
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    # Per-call copy: `config` is shared across all worker threads, and the
    # per-instance `image` assignment below raced with get_environment()'s
    # deepcopy in sibling threads — two tasks crossing an instance boundary
    # could swap images and run the agent in the wrong repo's container.
    env_config = dict(config.get("environment") or {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    if env_config["environment_class"] == "singularity-localimage":
        # image_name in the dataset is already a fully-qualified local .sif path
        env_config["image"] = instance["image_name"]
    else:
        image_name = get_swebench_docker_image_name(instance)
        if env_config["environment_class"] == "docker":
            env_config["image"] = image_name
        elif env_config["environment_class"] in ("singularity", "singularity-persistent", "singularity-kernel"):
            env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        # env.config.timeout governs per-step exec; startup needs more (300s
        # for big repos' .git scrub). Temporarily bump, then restore.
        original_timeout = getattr(env.config, "timeout", 30)
        startup_timeout = getattr(env.config, "startup_timeout", original_timeout)
        env.config.timeout = startup_timeout
        try:
            out = env.execute(startup_command)
        finally:
            env.config.timeout = original_timeout
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def _safe_load_preds(output_path: Path) -> dict:
    """Load preds.json tolerating a missing OR CORRUPT file.

    A non-atomic writer killed mid-write (e.g. by a condor memory-limit hold
    during a multi-hundred-MB write) leaves a truncated file. Crashing on it at
    startup put jobs into a permanent held loop. The --run-root by_instance/
    skip-filter is the authoritative dedup mechanism and the real trajectories
    live in by_instance/<iid>/*.traj.json, so a corrupt aggregate is safe to
    treat as empty (it self-heals on the next atomic write)."""
    try:
        return json.loads(output_path.read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning(f"preds file {output_path} unreadable ({type(e).__name__}: {e}); treating as empty")
        return {}


def _atomic_write_json(output_path: Path, data) -> None:
    """Write JSON via a same-dir temp file + atomic os.replace, so a process
    kill mid-write can never leave a truncated/corrupt file."""
    tmp = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, output_path)


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = _safe_load_preds(output_path) if output_path.exists() else {}
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        _atomic_write_json(output_path, output_data)


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = _safe_load_preds(output_path)
        if instance_id in output_data:
            del output_data[instance_id]
            _atomic_write_json(output_path, output_data)


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    *,
    composite_id: str | None = None,
    traj_filename: str | None = None,
) -> None:
    """Process a single SWEBench instance.

    composite_id / traj_filename: when set, used in place of the legacy
    instance_id and <iid>.traj.json. Enables N-samples-per-instance by giving
    each (iid, sample_idx) tuple its own preds.json key and trajectory file.
    """
    instance_id = instance["instance_id"]
    preds_key = composite_id or instance_id
    traj_name = traj_filename or f"{instance_id}.traj.json"
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", preds_key)
    (instance_dir / traj_name).unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(preds_key)
    progress_manager.update_instance_status(preds_key, "Pulling/starting docker")

    agent = None
    extra_info = None
    _phase_t0 = time.perf_counter()

    env = None
    try:
        # Setup phase — apptainer overlay + env_startup_command. NO LLM call yet,
        # so do NOT count this against the adaptive concurrency limit.
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=preds_key,
            **config.get("agent", {}),
        )
        # Trajectory phase — first LLM call about to happen → KV will be held →
        # acquire the per-endpoint trajectory slot. Released as soon as agent.run()
        # returns (before WIP salvage / save_traj / cleanup, which are bash-only).
        limiter = getattr(model, "limiter", None)
        if limiter is not None:
            progress_manager.update_instance_status(preds_key, "Waiting on LLM concurrency slot")
            with limiter.acquire():
                progress_manager.update_instance_status(preds_key, "Agent running")
                exit_status, result = agent.run(task)
        else:
            exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}

    # If the agent didn't successfully Submit (e.g. LimitsExceeded,
    # ContextWindowExceededError, TimeoutExpired, FormatError, ...), the
    # container is still alive at this point. Try one last `git add -A &&
    # git diff --cached` so any in-progress edits become the submitted patch
    # instead of being thrown away. Many "no patch" failures had real edits
    # in flight; this captures them.
    if exit_status != "Submitted" and env is not None:
        try:
            output = env.execute("git add -A && git diff --cached")
            # NOTE: previously `.strip()` here removed the trailing newline that
            # GNU `patch` needs to terminate the last hunk, causing the harness
            # to reject otherwise-valid salvaged diffs with "patch unexpectedly
            # ends in middle of line". `.lstrip()` is enough — git diff always
            # ends in '\n' (even the `\ No newline at end of file` marker line
            # ends in '\n'). Mirrors the inner-salvage idiom in agents/default.py.
            text = output.get("output", "") if isinstance(output, dict) else ""
            wip_patch = text.lstrip()
            if wip_patch.startswith("diff --git"):
                logger.info(
                    f"{instance_id}: salvaged WIP patch ({len(wip_patch)} chars) after {exit_status}"
                )
                result = wip_patch
                extra_info = (extra_info or {}) | {"salvaged_after": exit_status}
                exit_status = f"{exit_status}+SalvagedWIP"
        except Exception as e:
            logger.warning(f"{instance_id}: WIP salvage failed: {e}")

    # Phase telemetry — record per-trajectory phase split (LLM / bash / other).
    # `other` includes env setup, save_traj, WIP salvage, parsing, message rendering.
    _t_wall = time.perf_counter() - _phase_t0
    _t_llm = getattr(agent, "t_llm", 0.0) if agent is not None else 0.0
    _t_bash = getattr(agent, "t_bash", 0.0) if agent is not None else 0.0
    _n_turns = sum(1 for m in (agent.messages if agent is not None else []) if m.get("role") == "assistant")
    extra_info = (extra_info or {}) | {
        "phase_timing": {"t_wall": _t_wall, "t_llm": _t_llm, "t_bash": _t_bash}
    }
    _record_phase(_t_wall, _t_llm, _t_bash, _n_turns)

    # Integrated grading — spin up a FRESH singularity (separate from the
    # agent's container, which we won't trust because the agent could have
    # modified test files / conftest / deps). The AIMD slot was already
    # released before this point, so grading runs outside the LLM-concurrency
    # window. We skip empties cheaply: no_patch results return in < 1ms.
    # MSWEA_INLINE_GRADE=0 skips inline grading entirely (default on, so eval is
    # unaffected). The online-OPD rollout sets this so the rollout subprocess exits
    # as soon as generation finishes instead of blocking on ~50s/traj pytest grading;
    # grading is done asynchronously off the training loop's critical path. The
    # agent's patch is still persisted (info.submission), so it can be graded later.
    grade_report = None
    if os.environ.get("MSWEA_INLINE_GRADE", "1") == "0":
        grade_report = {"status": "grading_disabled", "resolved": None}
    elif is_valid_patch(result):
        progress_manager.update_instance_status(preds_key, "Grading patch")
        try:
            grade_report = grade_instance(instance, result)
        except Exception as e:
            logger.warning(f"{instance_id}: grading raised: {e!r}")
            grade_report = {"status": "exception", "error": repr(e)[:400], "resolved": False}
    else:
        grade_report = {"status": "no_patch", "resolved": False, "patch_exists": False}
    extra_info = (extra_info or {}) | {"grading": grade_report}

    save_traj(
        agent,
        instance_dir / traj_name,
        exit_status=exit_status,
        result=result,
        extra_info=extra_info,
        instance_id=instance_id,
        print_fct=logger.info,
    )
    update_preds_file(output_dir / "preds.json", preds_key, model.config.model_name, result)
    progress_manager.on_instance_end(preds_key, exit_status)
    # Explicit overlay unlink — don't wait for Python GC of `env`. Idempotent
    # with __del__; safe if either fires twice or not at all.
    if env is not None:
        try:
            env.cleanup()
        except Exception:
            pass


def parse_instance_ids(spec: str) -> set[str] | None:
    """Parse --instance-ids: empty -> None; '@path' or path ending in .json ->
    JSON file (list, or dict with 'instance_ids'/'ids' key, or dict of id->...);
    otherwise comma-separated literal."""
    if not spec:
        return None
    if spec.startswith("@") or spec.endswith(".json"):
        path = spec[1:] if spec.startswith("@") else spec
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            ids = data.get("instance_ids") or data.get("ids") or list(data.keys())
        else:
            ids = data
        return {str(x) for x in ids}
    return {x.strip() for x in spec.split(",") if x.strip()}


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    instance_ids_spec: str = typer.Option("", "--instance-ids", help="Restrict to these instance IDs. Comma-separated, or '@path.json' / 'path.json' (list, {instance_ids: [...]}, {ids: [...]}, or dict-of-ids). Applied before --filter/--slice so they still work on the resulting subset.", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    run_root: str = typer.Option("", "--run-root", help="Shared run root containing by_instance/. If set, drop tasks whose trajectory file already exists at <run_root>/by_instance/<iid>/<traj_filename>. Replaces preds.json-based skip-marker pre-fill.", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    num_samples: int = typer.Option(1, "--num-samples", help="Generate N samples per instance (each a fresh apptainer overlay). N>=2 saves trajectories as <iid>/sample_<N>.traj.json with preds.json keys <iid>__sample_<N>; N=1 keeps legacy <iid>/<iid>.traj.json naming.", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    id_set = parse_instance_ids(instance_ids_spec)
    if id_set is not None:
        before = len(instances)
        present = {inst["instance_id"] for inst in instances}
        missing = id_set - present
        if missing:
            sample = sorted(missing)[:5]
            logger.warning(
                f"--instance-ids: {len(missing)}/{len(id_set)} requested IDs not in dataset "
                f"(e.g., {sample})"
            )
        instances = [inst for inst in instances if inst["instance_id"] in id_set]
        logger.info(f"--instance-ids filter: {before} -> {len(instances)} instances")

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    # Expand to (instance, composite_id, traj_filename) tuples. num_samples=1
    # preserves legacy naming so single-sample callers/scripts keep working.
    tasks: list[tuple[dict, str, str]] = []
    for inst in instances:
        iid = inst["instance_id"]
        if num_samples > 1:
            for s in range(num_samples):
                tasks.append((inst, f"{iid}__sample_{s}", f"sample_{s}.traj.json"))
        else:
            tasks.append((inst, iid, f"{iid}.traj.json"))
    # Skip tasks whose trajectory already exists in the shared by_instance/ tree.
    # This is the source of truth across all jobs/runs that share the same run_root.
    # Pure filesystem stat per task — no SIF/overlay/patch work happens for skipped iids.
    if run_root and not redo_existing:
        run_root_p = Path(run_root)
        before = len(tasks)
        tasks = [
            (inst, cid, fn) for inst, cid, fn in tasks
            if not (run_root_p / "by_instance" / inst["instance_id"] / fn).exists()
        ]
        logger.info(f"Skipping {before - len(tasks)} tasks already in {run_root}/by_instance/")
    if not redo_existing and (output_path / "preds.json").exists():
        # Don't count "existing" preds whose model_patch is actually a captured
        # error message — re-attempt those. We've seen four classes of these:
        #   - RetryError / [Errno 108]: original LiteLLM connection failures
        #   - "Command '[": subprocess error from docker setup (docker load /
        #     run failure under cluster contention, recorded as the patch)
        #   - "litellm.ContextWindowExceededError" / "ContextWindowExceeded":
        #     pre-harness-fix context-overflow errors, recorded as the patch
        retry_errors = [
            "RetryError",
            "[Errno 108]",
            "Command '[",
            "ContextWindowExceededError",
            "litellm.ContextWindowExceededError",
            # dockerd-rootless was unreachable (likely a previous run started
            # the daemon but it died before this worker tried to talk to it,
            # or failed to start fast enough). With per-condor-job /tmp/, this
            # is rare in fresh runs but legacy preds may carry the error.
            "Cannot connect to the Docker daemon",
            # `docker load` retries exhausted (4 attempts with backoff). Often
            # actually downstream of cross-job pkill killing the sibling
            # dockerd; the load itself is fine, but the daemon went away.
            "docker load exhausted all retries",
        ]
        preds = _safe_load_preds(output_path / "preds.json")
        print(f"Found {len(preds)} preds entries")
        preds = {k: v for k, v in preds.items() if not any(v.get("model_patch", "").startswith(error) for error in retry_errors)}
        print(f"After removing RetryError entries, {len(preds)} remain")
        existing_keys = set(preds.keys())
        # Skip by composite key so multi-sample resume drops only the
        # (iid, sample) pairs already done, not the whole instance.
        before = len(tasks)
        tasks = [(inst, cid, fn) for inst, cid, fn in tasks if cid not in existing_keys]
        logger.info(f"Skipping {before - len(tasks)} already-done tasks")
    logger.info(f"Running on {len(tasks)} tasks ({len({t[0]['instance_id'] for t in tasks})} unique instances × up to {num_samples} samples)...")


    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(tasks), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                preds_key = futures[future]
                logger.error(f"Error in future for task {preds_key}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(preds_key, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance, inst, output_path, config, progress_manager,
                    composite_id=cid, traj_filename=fn,
                ): cid
                for inst, cid, fn in tasks
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
