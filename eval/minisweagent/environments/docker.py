import logging
import os
import re
import shlex
import socket
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 600
    """Timeout in seconds for pulling images."""
    image_tar_dir: str = '/fast/rolmedo/swesmith/docker_tarballs/'
    """Directory containing pre-saved image tarballs to load before pulling."""


class DockerEnvironment:
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        # Ensure Docker/Proxy environment is sane in proxied clusters
        # - Use local unix socket for Docker client
        # - Bypass proxies for the current host name (both lowercase and uppercase variants)
        host_name = socket.gethostname()
        if not os.getenv("DOCKER_HOST"):
            os.environ["DOCKER_HOST"] = "unix:///tmp/docker.sock"
        os.environ.setdefault("no_proxy", host_name)
        os.environ.setdefault("NO_PROXY", host_name)
        for key in ["DOCKER_HOST", "no_proxy", "NO_PROXY"]:
            if key not in self.config.forward_env:
                self.config.forward_env.append(key)
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        self._ensure_image_available()
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            # Belt-and-suspenders: never pull at run-time. Tarball load is
            # the only accepted source of images. If the image isn't local,
            # this errors out cleanly instead of silently fetching from a
            # registry that may have a different version.
            #
            # Opt-out: MSWEA_ALLOW_REGISTRY_PULL=1 → use "missing" pull policy,
            # so dockerd pulls on demand when the tarball pre-stage is empty.
            "--pull",
            "missing" if os.environ.get("MSWEA_ALLOW_REGISTRY_PULL") == "1" else "never",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def _ensure_image_available(self) -> None:
        """Ensure the requested image exists locally, loading from a tarball if configured.

        If the image is not present and a tarball directory is configured, try to load
        a tarball whose filename matches common sanitizations of the image name.
        """
        inspect = subprocess.run(
            [self.config.executable, "image", "inspect", self.config.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if inspect.returncode == 0:
            return

        tar_dir = self.config.image_tar_dir
        if not tar_dir:
            return
        if isinstance(tar_dir, str):
            tar_dir = Path(tar_dir)
        if not tar_dir.exists():
            return

        def sanitize(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

        full = self.config.image
        no_registry = full.split("/", 1)[-1]
        candidates = [
            tar_dir / f"{sanitize(full)}.tar",
            tar_dir / f"{sanitize(full)}.tar.gz",
            tar_dir / f"{sanitize(no_registry)}.tar",
            tar_dir / f"{sanitize(no_registry)}.tar.gz",
        ]

        import time
        for tarball in candidates:
            if tarball.exists():
                self.logger.info(f"Loading image from tarball: {tarball}")
                # Retry with exponential backoff. Under cluster contention the
                # docker daemon transiently refuses concurrent loads even with
                # the flock — retrying after a brief wait recovers most of
                # these without re-pulling the 5GB tarball over Lustre.
                #
                # Policy: tarball load is the ONLY accepted source. If all
                # retries are exhausted, raise — DO NOT fall back to pulling
                # from a registry. Pulling at eval time is slow, network-noisy,
                # and can introduce image-version drift.
                last_error: BaseException | None = None
                last_output: str = ""
                for attempt, backoff in enumerate([0, 5, 15, 45], start=1):
                    if backoff:
                        time.sleep(backoff)
                    try:
                        subprocess.run(
                            [
                                "flock", "/tmp/mswea_docker_load.lock",
                                self.config.executable, "load", "-i", str(tarball),
                            ],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=self.config.pull_timeout,
                        )
                        self.logger.info(f"Loaded image from tarball: {tarball}")
                        # Free disk: each tarball is single-use within a job
                        # (one image per instance), and the loaded image lives
                        # in dockerd's overlay store. Deleting the tarball
                        # right away keeps /tmp from filling up under high
                        # per-job density (we've seen 14 × 5 GB pre-stage +
                        # extracted layers exhaust 200 GB request_disk).
                        try:
                            tarball.unlink()
                        except OSError:
                            pass
                        return
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                        last_error = e
                        out = getattr(e, "stdout", None) or getattr(e, "output", None) or ""
                        if isinstance(out, bytes):
                            out = out.decode("utf-8", "replace")
                        last_output = out
                        self.logger.warning(
                            f"docker load attempt {attempt}/4 failed for {tarball}: "
                            f"stderr_tail={out[-800:]!r}"
                        )
                        continue
                # All retries exhausted: hard-fail rather than fall back to a
                # registry pull. Caller treats this as a setup failure for the
                # instance.
                raise RuntimeError(
                    f"docker load exhausted all retries for {tarball}: {last_error!r}. "
                    f"last_stderr_tail={last_output[-800:]!r}. "
                    f"Refusing to fall back to registry pull."
                )
        # No matching tarball found in the configured tar_dir.
        # Policy: refuse fallback unless MSWEA_ALLOW_REGISTRY_PULL=1 — when set,
        # we silently return so dockerd pulls the image on the first `docker run`.
        # Original rationale for the hard-fail was Lustre contention + version
        # drift; the fallback is opt-in for cases where the pre-stage step
        # legitimately has no tarballs (e.g. after a tarball garbage-collect).
        if os.environ.get("MSWEA_ALLOW_REGISTRY_PULL") == "1":
            self.logger.warning(
                f"No tarball found for image {self.config.image} under {tar_dir}; "
                f"MSWEA_ALLOW_REGISTRY_PULL=1 → letting dockerd pull on first run."
            )
            return
        raise RuntimeError(
            f"No tarball found for image {self.config.image} under {tar_dir}. "
            f"Refusing to fall back to registry pull. "
            f"(Set MSWEA_ALLOW_REGISTRY_PULL=1 to opt into pull-on-demand.)"
        )

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop and remove the Docker container, then `docker rmi` the image so
        rootless docker doesn't keep accumulating extracted layers in /tmp.
        Each SWE-bench instance uses a unique harbor image tag, so removing
        the image is safe and recovers ~5 GB of /tmp per instance.

        Synchronous (with timeouts) so the storage actually frees before the
        worker moves to the next instance — async fire-and-forget left the
        rmi pending and did not actually reclaim /tmp."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            image = getattr(self.config, "image", None)
            try:
                subprocess.run(
                    [self.config.executable, "rm", "-f", self.container_id],
                    timeout=30,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            if image:
                try:
                    subprocess.run(
                        [self.config.executable, "rmi", "-f", image],
                        timeout=30,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
