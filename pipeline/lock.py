"""
Single-instance run lock
========================
Keeps a scheduled (cron) run and an on-demand manual run from executing at the
same time.  Concurrent runs of the same entry point are unsafe: they race on the
shared admdb accumulators (`{ds}_history_stats`, `{ds}_trends_stats`) and on the
`{ds}_updates` watermark, so two overlapping runs can double-count a slice or
lose one entirely.

Uses `flock(2)` on a lock file.  Chosen over a PID file because the kernel
releases the lock when the process dies **however** it dies — a PID file left by
a `kill -9` would block every subsequent run until removed by hand.

Scope is one host: flock does not coordinate across machines.  That matches the
deployment (cron and manual runs on the Zabbix server).  If runs are ever
launched from several hosts against one admdb, this needs to become a
PostgreSQL advisory lock instead.

Note that flock is associated with the *open file description*, not the process,
so two separate `open()` calls conflict even inside one process — which is what
makes this testable without spawning subprocesses.
"""
from __future__ import annotations
from contextlib import contextmanager
import errno
import fcntl
import logging
import os
import time

logger = logging.getLogger(__name__)

# EX_TEMPFAIL from sysexits.h: "the command was not run, try again later".
# Distinct from 1 so a wrapper script can tell "already running" from "failed".
EXIT_ALREADY_RUNNING = 75

_HOLDER_BYTES = 256


class AlreadyRunning(RuntimeError):
    """Raised when another instance holds the lock and waiting timed out."""

    def __init__(self, name: str, path: str, holder: str = ""):
        self.name = name
        self.path = path
        self.holder = holder
        detail = f" ({holder})" if holder else ""
        super().__init__(
            f"another '{name}' run is already in progress{detail}; lock file: {path}"
        )


@contextmanager
def single_instance(
    name: str,
    lock_dir: str,
    wait_secs: int = 0,
    poll_secs: float = 1.0,
):
    """Hold an exclusive lock named `name` for the duration of the block.

    Parameters
    ----------
    name      : lock identity, e.g. "detect" -> <lock_dir>/detect.lock
    lock_dir  : directory for lock files; created if absent
    wait_secs : how long to wait for a competing run to finish.  0 (default)
                fails immediately — the right behaviour for cron, which should
                skip rather than pile up.
    poll_secs : retry interval while waiting

    Raises
    ------
    AlreadyRunning : the lock is held and `wait_secs` elapsed.

    The lock file is deliberately **not** deleted on release: unlinking it would
    race with a waiter that has already opened the same path and is blocking on
    an inode we would then detach, letting a third process create a fresh file
    and acquire the "same" lock concurrently.
    """
    os.makedirs(lock_dir, exist_ok=True)
    path = os.path.join(lock_dir, f"{name}.lock")

    # O_CREAT|O_RDWR rather than open(path, "w"): "w" truncates on open, which
    # would wipe the holder record of a run that currently owns the lock.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + max(wait_secs, 0)
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise AlreadyRunning(name, path, _read_holder(fd)) from None
                if not waited:
                    waited = True
                    logger.info(
                        "%s lock held by %s; waiting up to %ds",
                        name, _read_holder(fd) or "another run", wait_secs,
                    )
                time.sleep(poll_secs)

        _write_holder(fd)
        logger.info("acquired '%s' run lock (pid=%d)", name, os.getpid())
        try:
            yield path
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            logger.info("released '%s' run lock", name)
    finally:
        os.close(fd)


def _write_holder(fd: int) -> None:
    """Record who holds the lock, for a useful message in the blocked process."""
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"{os.getpid()}\n{time.time():.0f}\n".encode(), 0)


def _read_holder(fd: int) -> str:
    """Best-effort description of the current holder. Never raises."""
    try:
        raw = os.pread(fd, _HOLDER_BYTES, 0).decode("utf-8", "replace")
    except OSError:
        return ""
    lines = raw.strip().split("\n")
    if not lines or not lines[0].strip():
        return ""
    pid = lines[0].strip()
    if len(lines) > 1:
        try:
            age = int(time.time() - float(lines[1].strip()))
            return f"pid {pid}, running for {age}s"
        except ValueError:
            pass
    return f"pid {pid}"
