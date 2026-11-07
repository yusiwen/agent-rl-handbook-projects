"""The sandbox: where a program is run, and what a run is allowed to tell you.

The sandbox is the reward source for this project, so the interesting decisions are about
what a *failure* means rather than about isolation:

* **A timeout is not a wrong answer.** A program that is stopped at 20 seconds teaches
  nothing about the answer; if the record folds it into "incorrect", the model learns from a
  verdict the task never gave. The observation keeps ``timed_out`` separate, and the episode
  counts it separately.
* **A crashed program is not a wrong answer either.** Exit code, stderr and the timeout flag
  travel together; the episode reports ``sandbox_errors`` next to ``solved``.
* **The table is a file, not a string in the prompt.** Programs read ``table.csv`` from the
  working directory, which is what makes the same task runnable by a third party.

Two implementations, on purpose:

``DockerSandbox`` is the real one - ``--network=none``, a memory cap, a CPU cap, a pid cap and
a wall-clock timeout. ``LocalSandbox`` runs the same program in a plain subprocess on the host.
The local one exists for tests and for machines without a container runtime (the author's
laptop has none, by design); it is **not** a safety boundary and must never be the executor for
a model you do not already trust.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    """The result of one program run, in the form the record and the model both need."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    executor: str = "docker"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def failed(self) -> bool:
        return self.timed_out or self.exit_code != 0


class DockerSandbox:
    """Runs a program in a throwaway container with no network and a hard time limit."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout_s: float = 20.0,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        docker: str = "docker",
    ) -> None:
        self.image = image
        self.timeout_s = timeout_s
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker = docker

    def available(self) -> bool:
        try:
            done = subprocess.run(
                [self.docker, "image", "inspect", self.image],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0

    def run(self, code: str, files: dict[str, str] | None = None) -> Observation:
        workdir = Path(tempfile.mkdtemp(prefix="p4-sandbox-"))
        name = f"p4-{uuid.uuid4().hex[:12]}"
        try:
            (workdir / "prog.py").write_text(code, encoding="utf-8")
            for filename, content in (files or {}).items():
                target = workdir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            command = [
                self.docker, "run", "--rm", "--name", name,
                "--network=none",
                f"--memory={self.memory}", f"--memory-swap={self.memory}",
                f"--cpus={self.cpus}", f"--pids-limit={self.pids_limit}",
                "-v", f"{workdir}:/w", "-w", "/w",
                self.image, "python", "/w/prog.py",
            ]
            started = time.monotonic()
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired as expired:
                self._kill(name)
                return Observation(
                    exit_code=None,
                    stdout=_text(expired.stdout),
                    stderr=_text(expired.stderr),
                    timed_out=True,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    executor=f"docker:{self.image}",
                )
            return Observation(
                exit_code=done.returncode,
                stdout=done.stdout[-4000:],
                stderr=done.stderr[-4000:],
                timed_out=False,
                duration_ms=int((time.monotonic() - started) * 1000),
                executor=f"docker:{self.image}",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _kill(self, name: str) -> None:
        subprocess.run([self.docker, "kill", name], capture_output=True, timeout=30)


class LocalSandbox:
    """Runs the program in a subprocess on this machine.  Tests and laptops only."""

    def __init__(self, timeout_s: float = 20.0, python: str | None = None) -> None:
        self.timeout_s = timeout_s
        self.python = python or sys.executable

    def available(self) -> bool:
        return True

    def run(self, code: str, files: dict[str, str] | None = None) -> Observation:
        workdir = Path(tempfile.mkdtemp(prefix="p4-local-"))
        try:
            (workdir / "prog.py").write_text(code, encoding="utf-8")
            for filename, content in (files or {}).items():
                target = workdir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir)}
            started = time.monotonic()
            try:
                done = subprocess.run(
                    [self.python, "prog.py"],
                    cwd=workdir, capture_output=True, text=True,
                    timeout=self.timeout_s, env=env,
                )
            except subprocess.TimeoutExpired as expired:
                return Observation(
                    exit_code=None,
                    stdout=_text(expired.stdout),
                    stderr=_text(expired.stderr),
                    timed_out=True,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    executor="local-python",
                )
            return Observation(
                exit_code=done.returncode,
                stdout=done.stdout[-4000:],
                stderr=done.stderr[-4000:],
                timed_out=False,
                duration_ms=int((time.monotonic() - started) * 1000),
                executor="local-python",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[-4000:]
    return value[-4000:]


def python_available_in(sandbox: DockerSandbox) -> dict[str, object]:
    """A one-line health check a run record can cite: does the image exist, and does it run?"""
    probe = sandbox.run("import sys; print(sys.version.split()[0])")
    return {
        "image": sandbox.image,
        "present": probe.exit_code == 0,
        "python": probe.stdout.strip(),
        "stderr": probe.stderr.strip()[:200],
        "executor": probe.executor,
    }


if __name__ == "__main__":  # pragma: no cover - manual probe
    print(json.dumps(python_available_in(DockerSandbox()), indent=2))
