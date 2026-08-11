#!/usr/bin/env python3

import fcntl
import logging
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid

def _worker_id() -> str:
    """Per-worker identifier for isolation in /tmp paths.

    mini-extra-swebench runs workers as THREADS in one process, so os.getpid()
    is the same across all workers. Use threading.get_ident() (unique per
    thread within a process) combined with PID for cross-process distinctness.
    """
    return f"{os.getpid()}t{threading.get_ident()}"


def _robust_rmtree(path) -> None:
    """rmtree that survives fuse-overlayfs's internal work/work dir (mode 000).

    fuse-overlayfs leaves a workdir with no permissions; we own it, so chmod
    every directory traversable (top-down, during the walk) before removing.
    Fork-free — safe to call from hundreds of worker threads.
    """
    path = str(path)
    if not os.path.exists(path):
        return
    for root, dnames, _ in os.walk(path):
        for dn in dnames:
            try:
                os.chmod(os.path.join(root, dn), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import socket

@dataclass
class SingularityEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    startup_timeout: int = 300
    """Timeout for env_startup_command (which can do heavy work like .git rescrub).
    swebench.py temporarily swaps `timeout` to this value for that one call."""
    submit_timeout: int = 300
    """Timeout for the agent's submission command (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT).
    `git add -A && git diff --cached` on a big-repo (django ~30k files) can exceed the
    per-step `timeout`; truncating it would yield an ungradable patch. mini-swe-agent's
    agents.default.execute_action detects the submit command and swaps `timeout` to this
    value for that one bash exec."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity/apptainer executable."""
    unsquashfs_executable: str = os.getenv("MSWEA_UNSQUASHFS", "unsquashfs")
    """Tool used to extract a SIF's squashfs rootfs into a directory sandbox."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    image_sif_dir: str = '/fast/rolmedo/swesmith/singularity_images/'
    """Directory containing pre-saved Singularity .sif images to use before pulling."""
    image_tar_dir: str = '/fast/rolmedo/swesmith/docker_tarballs/'
    """Directory containing pre-saved docker image tarballs to use before pulling."""
    fakeroot: bool = False
    """Run apptainer exec with --fakeroot (UID=0 inside, file metadata appears as root)."""
    mem_limit_gb: int | None = None
    """Per-instance virtual memory cap (RLIMIT_AS) enforced via prlimit. None = no cap."""
    cpu_thread_cap: int = 1
    """Cap per-command math-library thread pools (OMP / OpenBLAS / MKL / NumExpr / vecLib)
    to this many threads (0 = no cap). One repo's multithreaded test/build (numpy/scipy
    OpenMP) otherwise grabs ALL cores — measured ~24 cores for a single `python` — which
    saturates the host under high rollout concurrency and starves the GPUs. Capping to 1
    keeps cpu_idle ~80% at 576 concurrent rollouts and is the recommended default for
    parallel SWE-bench rollouts. Overridden by any of these vars set in `env`."""
    prefix_patch_dir: str = os.getenv("MSWEA_SANDBOX_PREFIX_PATCH_DIR", "")
    """rephrase483: directory of per-instance prefix patches (<instance_id>.patch)
    applied to /testbed host-side at sandbox extraction (once per node per image,
    amortized across every rollout sharing the sandbox). Empty = off. Fail-loud:
    if set but no patch matches this image, sandbox build raises — the renamed
    arm must never silently run an unrenamed tree."""
    sandbox_git_reinit: bool = os.getenv("MSWEA_SANDBOX_GIT_REINIT", "") == "1"
    """rephrase483: re-init /testbed/.git as a single fresh root commit at
    extraction. Unlike the in-container ref-drop scrub (which cannot hide an
    applied prefix patch — `git log -p`/`git diff` would print the old<->new
    rename map), this makes the renamed tree the unqualified baseline. Set it
    in BOTH arms (original too) so history availability stays symmetric."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        host_name = socket.gethostname()
        os.environ.setdefault("no_proxy", f"172.22.0.0/16,127.0.0.0/8,{host_name}")
        os.environ.setdefault("NO_PROXY", f"172.22.0.0/16,127.0.0.0/8,{host_name}")
        os.environ.setdefault("http_proxy", "http://172.22.0.103:8080")
        os.environ.setdefault("https_proxy", "http://172.22.0.103:8080")
        os.environ.setdefault("HTTP_PROXY", "http://172.22.0.103:8080")
        os.environ.setdefault("HTTPS_PROXY", "http://172.22.0.103:8080")
        for key in ["no_proxy", "NO_PROXY"]:
            if key not in self.config.forward_env:
                self.config.forward_env.append(key)
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            if key not in self.config.forward_env:
                self.config.forward_env.append(key)
        # Forward TZ-related envs and default to UTC inside the container to make tests deterministic.
        # for key in ["TZ", "LANG", "LC_ALL", "LANGUAGE"]:
        #     if key not in self.config.forward_env:
        #         self.config.forward_env.append(key)
        # self.config.env.setdefault("TZ", "UTC0")
        self.sandbox_dir = self._build_sandbox()

    # ------------------------------------------------------------------
    # Sandbox construction
    # ------------------------------------------------------------------
    def _build_sandbox(self) -> Path:
        """Extract the image rootfs to a shared, read-only *directory* sandbox
        (once per image, flock-guarded) and give this instance a private
        writable directory overlay.

        Why a directory sandbox instead of `--overlay <ovl> <image.sif>`:
        reading a `.sif` in unprivileged (non-setuid) mode mounts it through
        `squashfuse_ll`, which decompresses squashfs on *every file read* in
        userspace. Under hundreds of concurrent rollouts that FUSE
        decompression is the dominant cost — measured ~6-8x slower per exec on
        this cluster, and it degrades further with concurrency. An unsquashfs'd
        directory is read straight from the kernel page cache with zero FUSE.

        Isolation / parallelism guarantees (verified on this cluster):
          * the extracted tree is shared read-only as the overlay *lowerdir*
            across every concurrent rollout of the same image, so rollouts
            never see or corrupt each other's writes (overlayfs keeps the
            lower pristine — a deleted file only creates a whiteout in the
            private upper);
          * `--contain` keeps the host filesystem out of the container, so the
            agent cannot reach /home, /fast, /lustre, or the host /tmp where
            the shared sandbox and sibling overlays live — the worst a
            destructive agent can do is trash its own ephemeral overlay.

        Returns the sandbox directory; execute() composes
        `--overlay <self.overlay_path> <sandbox_dir>`.
        """
        base_tmp = Path(os.getenv("MSWEA_TMPDIR", "/tmp"))
        base_tmp.mkdir(parents=True, exist_ok=True)
        sif_path = self._resolve_sif_path()
        sandbox_dir = self._extract_sandbox(sif_path, base_tmp)
        self.overlay_path = self._create_overlay(base_tmp)
        self.logger.info(f"Using sandbox {sandbox_dir} with overlay {self.overlay_path}")
        return sandbox_dir

    @staticmethod
    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).lstrip("_")

    def _resolve_sif_path(self) -> Path:
        """Locate the .sif for this instance's image by searching image_sif_dir.

        Subclasses (e.g. SingularityLocalImageEnvironment) override this when the
        image is already a fully-qualified .sif path.
        """
        sif_dir = Path(getattr(self.config, "image_sif_dir", "") or "")
        if not sif_dir.exists():
            raise RuntimeError(
                f"image_sif_dir {str(sif_dir)!r} does not exist; cannot locate a SIF for "
                f"image {self.config.image!r}"
            )
        full = self.config.image
        no_registry = full.split("/", 1)[-1]
        candidates = [
            sif_dir / f"{self._sanitize(full)}.sif",
            sif_dir / f"{self._sanitize(no_registry)}.sif",
        ]
        sif_path = next((p for p in candidates if p.exists()), None)
        if sif_path is None:
            raise RuntimeError(
                f"No SIF found for image {self.config.image!r} in {sif_dir} "
                f"(tried {[p.name for p in candidates]})"
            )
        return sif_path

    def _no_leak_env(self, base_tmp: Path) -> dict[str, str]:
        """Keep apptainer's cache/tmp off the quota'd host $HOME (~/.apptainer)."""
        return {
            **os.environ,
            "APPTAINER_CACHEDIR": str(base_tmp), "SINGULARITY_CACHEDIR": str(base_tmp),
            "APPTAINER_TMPDIR": str(base_tmp), "SINGULARITY_TMPDIR": str(base_tmp),
            "TMPDIR": str(base_tmp),
        }

    def _squashfs_offset(self, sif_path: Path, env: dict) -> int:
        """Byte offset of the squashfs rootfs partition inside the SIF.

        `unsquashfs -o <offset>` extracts straight from the SIF (no temp copy);
        the offset is the start of the `FS (Squashfs/...)` partition reported by
        `apptainer sif list`.
        """
        res = subprocess.run(
            [self.config.executable, "sif", "list", str(sif_path)],
            capture_output=True, text=True, env=env,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "quashfs" in line:  # matches "Squashfs" in the TYPE column
                    m = re.search(r"(\d+)\s*-\s*\d+", line)
                    if m:
                        return int(m.group(1))
        raise RuntimeError(
            f"could not determine squashfs offset for {sif_path} "
            f"(sif list rc={res.returncode}); stdout={res.stdout[-400:]!r} stderr={res.stderr[-400:]!r}"
        )

    def _extract_sandbox(self, sif_path: Path, base_tmp: Path) -> Path:
        """Extract sif_path's rootfs into a shared read-only directory sandbox.

        Cached and keyed by the SIF identity (name + size) so every worker and
        rollout of the same image shares ONE extraction. An flock serializes
        extraction on this node (a single ~8s unsquashfs per image); concurrent
        workers block on the lock, then reuse the published sandbox. Publish is
        atomic (os.replace) so a half-written tree is never observed.
        """
        prefix_patch = self._resolve_prefix_patch()
        variant = ""
        if prefix_patch is not None:
            import hashlib
            variant += "-p" + hashlib.sha1(prefix_patch.read_bytes()).hexdigest()[:10]
        if self.config.sandbox_git_reinit:
            variant += "-reinit"
        cache_name = self._sanitize(f"{sif_path.stem}-{sif_path.stat().st_size}{variant}")
        sandbox_dir = base_tmp / f"sandbox-{cache_name}"
        if (sandbox_dir / "testbed").is_dir():
            return sandbox_dir

        env = self._no_leak_env(base_tmp)
        lock_path = base_tmp / f"sandbox-{cache_name}.lock"
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            # Re-check under the lock: another worker may have published it while
            # we were blocked.
            if (sandbox_dir / "testbed").is_dir():
                return sandbox_dir
            if sandbox_dir.exists():  # stale/partial from a crashed extraction
                shutil.rmtree(sandbox_dir, ignore_errors=True)

            offset = self._squashfs_offset(sif_path, env)
            build_dir = base_tmp / f".sandbox-build-{uuid.uuid4().hex[:8]}"
            shutil.rmtree(build_dir, ignore_errors=True)
            self.logger.info(f"Extracting {sif_path.name} -> {sandbox_dir} (squashfs offset {offset})")
            res = subprocess.run(
                [self.config.unsquashfs_executable, "-no-progress", "-f",
                 "-d", str(build_dir), "-o", str(offset), str(sif_path)],
                capture_output=True, text=True, env=env,
            )
            # Validate by content, not just rc: a non-root unsquashfs returns
            # non-zero when it skips device nodes / can't set ownership, which is
            # harmless (apptainer --contain provides /dev). /testbed is the real
            # contract — the repo under test lives there.
            if not (build_dir / "testbed").is_dir():
                tail = (res.stderr or res.stdout)[-800:]
                shutil.rmtree(build_dir, ignore_errors=True)
                raise RuntimeError(
                    f"unsquashfs produced an invalid sandbox for {sif_path} "
                    f"(rc={res.returncode}, no /testbed): {tail}"
                )
            if res.returncode != 0:
                self.logger.warning(
                    f"unsquashfs rc={res.returncode} for {sif_path.name} (likely skipped "
                    f"device nodes as non-root); sandbox has /testbed, proceeding"
                )
            try:
                self._customize_sandbox(build_dir, prefix_patch)
            except Exception:
                shutil.rmtree(build_dir, ignore_errors=True)
                raise
            os.replace(build_dir, sandbox_dir)  # atomic publish onto the same fs
            return sandbox_dir

    def _resolve_prefix_patch(self) -> Path | None:
        """rephrase483: locate this instance's prefix patch (renamed-arm rename.patch).

        The instance_id is recovered from the image name (both the raw
        `sweb.eval.x86_64.<iid>:latest` key and the harbor sif naming with
        `__` -> `_1776_` are handled). Fail-loud when the dir is configured but
        no patch exists — a silently-unrenamed tree would corrupt the arm.
        """
        d = getattr(self.config, "prefix_patch_dir", "") or ""
        if not d:
            return None
        m = re.search(r"sweb\.eval\.x86_64\.(.+?)(?:_latest|:latest|$)", self.config.image)
        iid = m.group(1).replace("_1776_", "__") if m else None
        for cand in filter(None, [iid, self._sanitize(self.config.image)]):
            p = Path(d) / f"{cand}.patch"
            if p.exists():
                return p
        raise RuntimeError(
            f"prefix_patch_dir={d!r} is set but no prefix patch found for image "
            f"{self.config.image!r} (tried {iid!r} and sanitized image name)"
        )

    def _customize_sandbox(self, build_dir: Path, prefix_patch: Path | None) -> None:
        """rephrase483: host-side /testbed customization, inside the extraction
        flock and before the atomic publish — every rollout and grade sharing
        this sandbox sees the result.

        Order matters: patch first, then git re-init, so the fresh root commit
        IS the renamed baseline (agent submissions become agent-only diffs and
        no git metadata can reveal the old<->new rename map)."""
        if prefix_patch is None and not self.config.sandbox_git_reinit:
            return
        testbed = build_dir / "testbed"

        def _git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(testbed), *args], capture_output=True, text=True
            )

        if prefix_patch is not None:
            res = _git("apply", "--whitespace=nowarn", str(prefix_patch))
            if res.returncode != 0:
                raise RuntimeError(
                    f"prefix patch {prefix_patch} failed to apply: {res.stderr[-600:]}"
                )
            self.logger.info(f"Applied prefix patch {prefix_patch.name} to sandbox /testbed")
        if self.config.sandbox_git_reinit:
            shutil.rmtree(testbed / ".git", ignore_errors=True)
            steps = [
                ("init", "-q"),
                ("add", "-A"),
                ("-c", "user.email=eval@nanoswe", "-c", "user.name=nanoswe-eval",
                 "commit", "-qm", "baseline"),
            ]
        elif prefix_patch is not None:
            # Grading-style child commit (no agent in the loop): keeps `git
            # status` clean and lets `git checkout HEAD -- <tests>` restore
            # patched test files. NOT for agent-facing runs — `git log -p`
            # would print the rename map; use sandbox_git_reinit there.
            steps = [
                ("add", "-A"),
                ("-c", "user.email=eval@nanoswe", "-c", "user.name=nanoswe-eval",
                 "commit", "-qm", "prefix-patch baseline"),
            ]
        else:
            steps = []
        for args in steps:
            res = _git(*args)
            if res.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)} failed in sandbox: {res.stderr[-600:]}")
        if steps:
            self.logger.info("Sandbox /testbed git baseline updated (customize hook)")

    def _create_overlay(self, base_tmp: Path) -> Path:
        """Per-instance writable directory overlay (upper/ + work/).

        A directory overlay rather than an ext3 `overlay create` image: no extra
        apptainer subprocess, no 2 GB sparse image, and no fuse2fs daemon — it
        composes directly with the directory sandbox lowerdir. Each rollout gets
        its own overlay dir under a per-worker root, so concurrent rollouts on
        the same image never share a writable layer.
        """
        overlay_root = base_tmp / f"overlays-wkr{_worker_id()}"
        overlay_root.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_root / f"overlay-{uuid.uuid4().hex[:8]}"
        (overlay_path / "upper").mkdir(parents=True, exist_ok=True)
        (overlay_path / "work").mkdir(parents=True, exist_ok=True)
        return overlay_path

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd: list[str] = []
        if self.config.mem_limit_gb is not None:
            cmd.extend(["prlimit", f"--as={int(self.config.mem_limit_gb) * 1024 * 1024 * 1024}", "--"])
        cmd.extend([self.config.executable, "exec"])

        # Do not inherit directories and env vars from host. --contain is what
        # keeps the host filesystem (/home, /fast, /lustre, host /tmp) out of the
        # container — the agent cannot reach the shared sandbox or sibling
        # rollouts' overlays.
        cmd.extend(["--contain", "--cleanenv", "--no-home"])
        if getattr(self.config, "fakeroot", False):
            cmd.append("--fakeroot")

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        # Per-command CPU thread cap (recommended default for parallel rollouts):
        # keep one container's numpy/scipy/OpenMP test or build from grabbing every
        # core and saturating the host. Emitted BEFORE config.env so an explicit
        # value in `env` wins.
        cap = getattr(self.config, "cpu_thread_cap", 0)
        if cap and cap > 0:
            for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                if _var not in self.config.env:
                    cmd.extend(["--env", f"{_var}={cap}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--no-nv"])

        # self.sandbox_dir is an unsquashfs'd directory sandbox (read-only,
        # shared across concurrent rollouts of this image). --overlay gives THIS
        # instance its private writable layer; the sandbox stays pristine.
        shell_preamble = '[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed 2>/dev/null;'
        full_command = f"{shell_preamble} {command}"
        cmd.extend(["--overlay", str(self.overlay_path), str(self.sandbox_dir),
                    "bash", "-lc", full_command])
        base_tmp = Path(os.getenv("MSWEA_TMPDIR", "/tmp"))
        env = os.environ | {
            "APPTAINER_CACHEDIR": str(base_tmp),
            "SINGULARITY_CACHEDIR": str(base_tmp),
            "APPTAINER_TMPDIR": str(base_tmp),
            "SINGULARITY_TMPDIR": str(base_tmp),
            "TMPDIR": str(base_tmp),
        }
        # Launch in its OWN session/process-group so that a timeout can reap the
        # WHOLE group. apptainer's FUSE helper for the writable overlay
        # (fuse-overlayfs) is a group member; subprocess.run(timeout=) SIGKILLs
        # only the apptainer pid, orphaning the helper. Under a long run orphans
        # pile up and starve the FUSE layer until per-command mounts blow past
        # apptainer's ~10s mount deadline (rollout produces no patches). killpg
        # fixes it. (The directory sandbox already removes squashfuse_ll/fuse2fs
        # from the hot path, so there are far fewer FUSE helpers to leak.)
        proc = subprocess.Popen(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=self.config.timeout)
            return {"output": out, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                out, _ = proc.communicate(timeout=10)
            except Exception:
                out = ""
            # Preserve the contract agents.default.execute_action relies on: it
            # catches subprocess.TimeoutExpired and calls e.output.decode(...), so
            # output must be bytes (or None), not str.
            raise subprocess.TimeoutExpired(
                cmd, self.config.timeout,
                output=(out.encode("utf-8") if out else None),
            )

    def cleanup(self):
        # Remove only THIS instance's private overlay dir. The shared sandbox
        # (self.sandbox_dir) is read-only and reused by other rollouts of the
        # same image — never delete it here; the job clears /tmp at teardown.
        overlay = getattr(self, "overlay_path", None)
        if overlay is not None:
            _robust_rmtree(overlay)
            # Also drop the now-empty per-worker overlay root. _worker_id() mixes
            # in threading.get_ident(), which the OS recycles as the worker thread
            # pool churns, so a long run otherwise accumulates tens of thousands of
            # empty `overlays-wkr{pid}t{tid}` dirs in base_tmp. 24k+ sibling entries
            # in one dir slows every mkdir/lookup the next overlay mount does and
            # spams directory listings. rmdir removes the root ONLY if no sibling
            # overlay (a concurrent thread reusing the same id) is still live;
            # ENOTEMPTY/ENOENT are both fine and ignored.
            try:
                os.rmdir(os.path.dirname(str(overlay)))
            except OSError:
                pass

    def __del__(self):
        """Cleanup sandbox when object is destroyed. Guarded: at interpreter
        shutdown module globals (shutil/_robust_rmtree) may already be None, which
        would raise 'NoneType object is not callable' and spam tracebacks."""
        try:
            self.cleanup()
        except Exception:
            pass


@dataclass
class SingularityLocalImageEnvironmentConfig(SingularityEnvironmentConfig):
    # image is interpreted as a direct path to a .sif file on disk
    image_sif_dir: str = ""
    image_tar_dir: str = ""
    # Default to --fakeroot so:
    #   - host UID (rolmedo) doesn't leak into container processes / ls -la output
    #     (matches the upstream training-data convention of root-owned /testbed)
    #   - extra defense-in-depth: root inside maps to an unprivileged subuid
    #     outside, so a leaked host write would fail (--contain already blocks
    #     host paths regardless).
    fakeroot: bool = True


class SingularityLocalImageEnvironment(SingularityEnvironment):
    """SingularityEnvironment variant: `config.image` is a direct .sif path.

    Only the SIF lookup differs; the directory-sandbox extraction, overlay
    creation, execute() and cleanup() are inherited unchanged.
    """

    def __init__(self, *, config_class: type = SingularityLocalImageEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _resolve_sif_path(self) -> Path:
        sif_path = Path(self.config.image)
        if sif_path.suffix != ".sif" or not sif_path.exists():
            raise RuntimeError(
                f"SingularityLocalImageEnvironment: image must be an existing .sif path, "
                f"got {self.config.image!r}"
            )
        return sif_path


class SingularityPersistentEnvironment(SingularityEnvironment):
    """One long-lived `apptainer exec ... bash` per trajectory; commands are piped
    in over stdin and delimited with a unique sentinel.

    DO NOT USE AT SCALE — superseded by KernelOverlayEnvironment. Measured: when
    hundreds of these sessions START at once, their fuse-overlayfs writable-overlay
    mounts deadlock (only ~2 of 547 starters ever mount; even 32 resident sessions
    → 0 mounts in 25s). Amortizing apptainer's per-exec cost over a session does NOT
    help because the cost being amortized is the very FUSE mount that deadlocks.
    Kept only as a record of the explored approach; use `singularity-kernel`.

    Why: the per-`apptainer exec` setup (userns + fuse-overlayfs writable-overlay
    mount + teardown) costs ~6.6s at 546-way concurrency and DOMINATES the rollout
    step (~72% of wall) — measured. apptainer's base setup is irreducible per exec
    (even with no overlay it is ~40s for 512 concurrent), so the only way to kill
    it is to pay it ONCE per trajectory and keep the session alive. Each subsequent
    command is just a write to the live bash (~ms), so the dominant phase amortizes
    ~N-fold over a trajectory's commands.

    Drop-in for SingularityEnvironment: same dir-sandbox build + overlay, same
    execute() contract ({"output", "returncode"}; raises subprocess.TimeoutExpired
    on timeout with output as bytes). Statelessness the agent relies on is
    preserved — each command runs in a `( cd <cwd>; bash -lc '<preamble> <cmd>' )`
    subshell so cwd/env don't leak between commands — while filesystem edits DO
    persist (same overlay), which is the desired behaviour.
    """

    _PREAMBLE = ('[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && '
                 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed 2>/dev/null;')

    def __init__(self, *, config_class: type = SingularityEnvironmentConfig,
                 logger: logging.Logger | None = None, **kwargs):
        self._proc: subprocess.Popen | None = None
        super().__init__(config_class=config_class, logger=logger, **kwargs)

    # -- session lifecycle ------------------------------------------------
    def _session_argv(self) -> list[str]:
        cmd: list[str] = []
        if self.config.mem_limit_gb is not None:
            cmd += ["prlimit", f"--as={int(self.config.mem_limit_gb) * 1024 * 1024 * 1024}", "--"]
        cmd += [self.config.executable, "exec", "--contain", "--cleanenv", "--no-home"]
        if getattr(self.config, "fakeroot", False):
            cmd.append("--fakeroot")
        cmd += ["--pwd", self.config.cwd or "/testbed"]
        for key in self.config.forward_env:
            if (v := os.getenv(key)) is not None:
                cmd += ["--env", f"{key}={v}"]
        cap = getattr(self.config, "cpu_thread_cap", 0)
        if cap and cap > 0:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                if var not in self.config.env:
                    cmd += ["--env", f"{var}={cap}"]
        for k, v in self.config.env.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += ["--no-nv", "--overlay", str(self.overlay_path), str(self.sandbox_dir), "bash"]
        return cmd

    def _ensure_session(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        base_tmp = Path(os.getenv("MSWEA_TMPDIR", "/tmp"))
        env = os.environ | {k: str(base_tmp) for k in (
            "APPTAINER_CACHEDIR", "SINGULARITY_CACHEDIR", "APPTAINER_TMPDIR",
            "SINGULARITY_TMPDIR", "TMPDIR")}
        self._proc = subprocess.Popen(
            self._session_argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env, start_new_session=True, bufsize=0)
        self._send("export PS1='' PS2=''; set +o history 2>/dev/null; true\n")
        # Readiness probe: confirm the relay bash is alive (this also waits out the
        # one-time apptainer/overlay setup). Uses startup_timeout (apptainer mount
        # can be slow under concurrent trajectory starts).
        sent = f"__MSWEA_READY_{uuid.uuid4().hex}"
        self._send(f"printf '\\n{sent}:0:\\n'\n")
        _buf, ok = self._read_until(f"\n{sent}:".encode(),
                                    time.monotonic() + max(self.config.startup_timeout, 60))
        if not ok:
            self._kill_session()
            raise RuntimeError("persistent apptainer session failed to start (no readiness sentinel)")

    def _send(self, s: str):
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(s.encode("utf-8"))
        self._proc.stdin.flush()

    def _read_until(self, sentinel: bytes, deadline: float) -> tuple[bytes, bool]:
        """Read stdout until `sentinel` appears or `deadline` passes. Returns
        (bytes_so_far, found). Reads as data arrives so a large output never
        deadlocks on the pipe buffer."""
        buf = b""
        fd = self._proc.stdout.fileno()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return buf, False
            r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not r:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return buf, False
            if not chunk:               # EOF: session died
                return buf, False
            buf += chunk
            if sentinel in buf:
                return buf, True

    def _kill_session(self):
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        for s in (p.stdin, p.stdout):
            try: s and s.close()
            except Exception: pass

    # -- execute ----------------------------------------------------------
    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        self._ensure_session()
        T = int(self.config.timeout)
        work_dir = cwd or self.config.cwd or "/testbed"
        sent = f"__MSWEA_DONE_{uuid.uuid4().hex}"
        inner = f"{self._PREAMBLE} {command}"
        # In-container `timeout` bounds the command (exit 124 on timeout). Subshell
        # isolates cwd/env. Sentinel + exit code printed after.
        payload = (f"( cd {shlex.quote(work_dir)} 2>/dev/null; "
                   f"timeout {T}s bash -lc {shlex.quote(inner)} ); "
                   f"printf '\\n{sent}:%s:\\n' \"$?\"\n")
        self._send(payload)
        marker = f"\n{sent}:".encode()
        # Backstop deadline > in-container timeout, to allow timeout(1) to kill the
        # command and the sentinel to flush. If it still doesn't arrive the session
        # is wedged → restart it and surface a timeout.
        buf, ok = self._read_until(marker, time.monotonic() + T + 30)
        if not ok:
            self._kill_session()
            out = buf.decode("utf-8", "replace")
            raise subprocess.TimeoutExpired(command, T, output=(out.encode("utf-8") if out else None))
        idx = buf.index(marker)
        out = buf[:idx].decode("utf-8", "replace")
        rc_field = buf[idx + len(marker):].split(b":", 1)[0].strip()
        try:
            rc = int(rc_field)
        except ValueError:
            rc = 0
        if rc == 124:  # timeout(1) killed the command
            raise subprocess.TimeoutExpired(command, T, output=(out.encode("utf-8") if out else None))
        return {"output": out, "returncode": rc}

    def cleanup(self):
        self._kill_session()
        super().cleanup()


class KernelOverlayEnvironment(SingularityEnvironment):
    """Per-command container via `unshare` + KERNEL-native overlayfs + chroot —
    NO apptainer, NO fuse-overlayfs.

    Why: the per-`apptainer exec` setup (userns + fuse-overlayfs writable-overlay
    mount + teardown) costs ~6.6s at 546-way concurrency and DOMINATES the rollout
    step (~72% of wall, measured); fuse-overlayfs additionally deadlocks when many
    overlay mounts start at once. Unprivileged KERNEL overlayfs (in a user
    namespace) mounts with ZERO FUSE — measured 512 simultaneous full runtimes in
    1.25s, 512/512 ok, no stall (vs apptainer 7.34s @512, vs fuse-overlayfs which
    completed 0/32 resident in 25s). Per command we: enter a user+mount namespace,
    kernel-overlay-mount the extracted dir-sandbox (read-only lower) + this
    rollout's overlay (upper/work), rbind the host /proc + /dev + /sys, and chroot
    in — all inside a new PID namespace (--pid --fork). (/proc is rbind'd, not freshly
    mounted — a fresh procfs mount is blocked unprivileged on this cluster by the
    lxcfs overmounts on /proc/{cpuinfo,meminfo,...}, the same EPERM that defeats bwrap.
    The PID namespace, not the proc mount, is what isolates host processes: the agent
    can SEE host PIDs in the rbind'd /proc but cannot signal them, because kill/pkill
    resolve PIDs in the new namespace only — so a stray `pkill python` can't take down
    host training jobs. Verified: a host-PID kill returns ESRCH; the host survives.)

    Same contract as SingularityEnvironment ({"output","returncode"}; raises
    subprocess.TimeoutExpired with bytes output, reaped via killpg). Statelessness
    the agent relies on is preserved (each command chroots fresh at `cwd`); the
    overlay upper persists this rollout's filesystem edits across commands, and the
    shared read-only sandbox lower keeps concurrent rollouts isolated. When the
    process group is killed (timeout) the mount namespace dies with it, so there
    are NO leaked mounts/daemons (unlike apptainer's FUSE helpers).
    """

    _PREAMBLE = ('[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && '
                 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed 2>/dev/null;')

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        work_dir = cwd or self.config.cwd or "/testbed"
        ov = self.overlay_path
        merged = ov / "m"
        merged.mkdir(parents=True, exist_ok=True)
        q = shlex.quote
        inner = f"cd {q(work_dir)} 2>/dev/null; {self._PREAMBLE} {command}"
        script = (
            f"mount -t overlay overlay -o lowerdir={q(str(self.sandbox_dir))},"
            f"upperdir={q(str(ov / 'upper'))},workdir={q(str(ov / 'work'))} {q(str(merged))} || exit 91\n"
            f"mount --rbind /proc {q(str(merged / 'proc'))} 2>/dev/null || true\n"
            f"mount --rbind /dev {q(str(merged / 'dev'))} 2>/dev/null || true\n"
            f"mount --rbind /sys {q(str(merged / 'sys'))} 2>/dev/null || true\n"
            f"exec chroot {q(str(merged))} /bin/bash -lc {q(inner)}\n"
        )
        cmd = []
        if self.config.mem_limit_gb is not None:
            cmd += ["prlimit", f"--as={int(self.config.mem_limit_gb) * 1024 * 1024 * 1024}", "--"]
        # --pid --fork: a new PID namespace so the agent cannot see — and therefore
        # cannot signal — host processes. kill/pkill inside resolve PIDs in THIS
        # namespace only, so a stray `pkill -f python` (common in test/debug commands)
        # cannot reach the OPD loop / vLLM / other rolmedo training jobs (verified: a
        # host-PID kill returns ESRCH, the host process survives). --fork is required
        # (the new namespace's init must be a child of unshare). Bonus: when this PID 1
        # exits the kernel reaps the whole namespace, so no subprocess can leak.
        cmd += ["unshare", "--user", "--pid", "--fork", "--map-root-user", "--mount", "bash", "-c", script]

        env = dict(os.environ)
        cap = getattr(self.config, "cpu_thread_cap", 0)
        if cap and cap > 0:
            # HARD-set (not setdefault): we inherit the FULL host env (no --cleanenv like
            # apptainer), and this cluster's login env exports OMP_NUM_THREADS=<ncpu>, so
            # setdefault would silently leave the cap at the host value -> a single numpy/
            # OpenMP command re-grabs all cores (the exact core-hogging the cap exists to
            # stop). config.env (applied just below) still wins for any var it specifies.
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                if var not in self.config.env:
                    env[var] = str(cap)
        for k, v in self.config.env.items():
            env[k] = str(v)

        proc = subprocess.Popen(
            cmd, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=self.config.timeout)
            return {"output": out, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                out, _ = proc.communicate(timeout=10)
            except Exception:
                out = ""
            raise subprocess.TimeoutExpired(
                cmd, self.config.timeout, output=(out.encode("utf-8") if out else None))
