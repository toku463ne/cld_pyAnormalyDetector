"""Unit tests for the single-instance run lock.

flock is tied to the open file description rather than the process, so two
separate single_instance() calls conflict even inside one interpreter — no
subprocess needed to prove mutual exclusion.  One test does fork, to confirm
the kernel releases the lock when a holder is SIGKILLed (the failure mode a PID
file would get wrong).
"""
import os
import signal
import time

import pytest

from pipeline.lock import (
    EXIT_ALREADY_RUNNING,
    AlreadyRunning,
    single_instance,
)


def test_second_run_is_refused(tmp_path):
    with single_instance("detect", str(tmp_path)):
        with pytest.raises(AlreadyRunning):
            with single_instance("detect", str(tmp_path)):
                pytest.fail("second run must not acquire the lock")


def test_lock_is_released_on_normal_exit(tmp_path):
    with single_instance("detect", str(tmp_path)):
        pass
    with single_instance("detect", str(tmp_path)):
        pass          # re-acquirable


def test_lock_is_released_when_body_raises(tmp_path):
    with pytest.raises(ValueError):
        with single_instance("detect", str(tmp_path)):
            raise ValueError("boom")
    with single_instance("detect", str(tmp_path)):
        pass          # a crashed run must not wedge the lock


def test_different_names_do_not_block_each_other(tmp_path):
    # anomdec-detect and anomdec-detect-fast must be able to run concurrently.
    with single_instance("detect", str(tmp_path)):
        with single_instance("detect-fast", str(tmp_path)):
            pass


def test_lock_dir_is_created(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    with single_instance("detect", str(nested)):
        assert (nested / "detect.lock").exists()


def test_error_names_the_holder(tmp_path):
    with single_instance("detect", str(tmp_path)):
        with pytest.raises(AlreadyRunning) as excinfo:
            with single_instance("detect", str(tmp_path)):
                pass
    msg = str(excinfo.value)
    assert f"pid {os.getpid()}" in msg          # who is holding it
    assert "detect.lock" in msg                  # and where the lock lives


def test_wait_times_out_and_reports(tmp_path):
    with single_instance("detect", str(tmp_path)):
        started = time.monotonic()
        with pytest.raises(AlreadyRunning):
            with single_instance("detect", str(tmp_path), wait_secs=1, poll_secs=0.1):
                pass
        assert time.monotonic() - started >= 1.0     # actually waited


def test_wait_acquires_once_the_holder_finishes(tmp_path):
    # Child holds the lock briefly; parent waits and must succeed.
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:                                  # child
        os.close(read_fd)
        try:
            with single_instance("detect", str(tmp_path)):
                os.write(write_fd, b"x")          # signal "lock held"
                time.sleep(1.0)
        finally:
            os._exit(0)

    os.close(write_fd)
    assert os.read(read_fd, 1) == b"x"            # child holds it now
    os.close(read_fd)
    try:
        with single_instance("detect", str(tmp_path), wait_secs=10, poll_secs=0.1):
            pass                                  # acquired after child exited
    finally:
        os.waitpid(pid, 0)


def test_lock_survives_a_killed_holder(tmp_path):
    # The reason this is flock and not a PID file: SIGKILL must not wedge it.
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:                                  # child
        os.close(read_fd)
        with single_instance("detect", str(tmp_path)):
            os.write(write_fd, b"x")
            time.sleep(30)                        # killed before this returns
        os._exit(0)

    os.close(write_fd)
    assert os.read(read_fd, 1) == b"x"
    os.close(read_fd)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)

    with single_instance("detect", str(tmp_path), wait_secs=5, poll_secs=0.1):
        pass                                      # kernel released it


def test_exit_code_is_tempfail():
    # 75 = EX_TEMPFAIL; distinct from 1 so wrappers can tell the cases apart.
    assert EXIT_ALREADY_RUNNING == 75
