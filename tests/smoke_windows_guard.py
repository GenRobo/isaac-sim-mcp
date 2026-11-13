"""Exercise real Windows jobs using ONLY this script's harmless Python children.

No Isaac, Kit, renderer, PowerShell, GPU query or arbitrary command option exists.
An outer kill-on-close job contains each test worker and all descendants, including
when an assertion fails. Each harmless child also exits on its own after 15 seconds.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaac_launch_guard as guard


SCRIPT = Path(__file__).resolve()
BASELINE = (guard.GPU("GPU-smoke", 24576, 1024, 23552),)
SCENARIOS = ("deadline", "vram", "monitor-failure", "root-exit", "guard-kill")


def command(*arguments):
    return subprocess.list2cmdline([sys.executable, str(SCRIPT), *map(str, arguments)])


def wait_until(predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for harmless process-tree state")
        time.sleep(0.05)


def child(mode, state, scenario):
    (state / f"{mode}.pid").write_text(str(os.getpid()), encoding="ascii")
    if mode == "root":
        # These processes inherit the inner job automatically, even though Popen
        # does not know about that job. The root may exit ahead of its child.
        subprocess.Popen([sys.executable, str(SCRIPT), "--child", "leaf", "--state", str(state), "--scenario", scenario])
        wait_until(lambda: (state / "leaf.pid").exists())
        if scenario == "root-exit":
            return 0
    time.sleep(15)
    return 0


def worker(state, scenario):
    api = guard.WindowsAPI()
    runtime = 3.0 if scenario == "deadline" else 10.0
    policy = guard.Policy(runtime=runtime)

    def monitor(timeout):
        if (state / "leaf.pid").exists():
            if scenario == "vram":
                return (guard.GPU("GPU-smoke", 24576, 13000, 11576),)
            if scenario == "monitor-failure":
                raise guard.GuardError("injected monitoring failure")
        return BASELINE

    with guard.LaunchLock(api), guard.WindowsJob(api) as job:
        job.spawn(sys.executable, command("--child", "root", "--state", state, "--scenario", scenario), state)
        (state / "ready").write_text("ready", encoding="ascii")
        outcome = guard.supervise(job, monitor, policy, BASELINE)
    (state / "outcome.json").write_text(json.dumps({
        "code": outcome.code, "reason": outcome.reason, "peak": outcome.peak,
    }), encoding="utf-8")
    return 0


def smoke(scenarios):
    api = guard.WindowsAPI()
    terminate_process = api.dll.TerminateProcess
    terminate_process.argtypes = [wintypes.HANDLE, wintypes.UINT]
    terminate_process.restype = wintypes.BOOL
    expected = {"deadline": "runtime deadline", "vram": "VRAM threshold",
                "monitor-failure": "injected monitoring failure", "root-exit": "launcher exited"}
    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix="isaac-guard-smoke-") as temp:
            state = Path(temp)
            # The outer job is solely a test safety net. Production ownership is
            # the separate inner job created by the worker's real WindowsJob.
            with guard.WindowsJob(api) as safety:
                safety.spawn(sys.executable, command("--worker", "--state", state, "--scenario", scenario), state)
                wait_until(lambda: (state / "ready").exists() and (state / "leaf.pid").exists())
                if scenario == "guard-kill":
                    if safety.active_count() < 3:
                        raise AssertionError("expected worker, root and grandchild before guard termination")
                    try:
                        with guard.LaunchLock(api):
                            raise AssertionError("second guard acquired the interprocess lock")
                    except guard.GuardError as exc:
                        if "another guarded launch" not in str(exc):
                            raise
                    api.require(terminate_process(safety.process, 99), "terminate harmless guard worker")
                wait_until(lambda: safety.root_exit_code() is not None, 12)
                # Check before the safety job closes: only the inner job could
                # have performed this cleanup. No PID-based killing is used.
                wait_until(lambda: safety.active_count() == 0, 5)
                if scenario != "guard-kill":
                    if safety.root_exit_code() != 0:
                        raise AssertionError("harmless worker failed before reporting")
                    outcome = json.loads((state / "outcome.json").read_text(encoding="utf-8"))
                    if expected[scenario] not in outcome["reason"]:
                        raise AssertionError(f"unexpected stop: {outcome}")
                    if scenario == "deadline" and outcome["code"] != 124:
                        raise AssertionError(f"deadline did not propagate: {outcome}")
                    if scenario == "vram" and outcome["peak"] != 13000:
                        raise AssertionError(f"threshold peak not retained: {outcome}")
            print(f"PASS {scenario}: owned Python root/grandchild gone; no GPU work", flush=True)
    # All workers exited, including the force-killed owner: the mutex must be free.
    with guard.LaunchLock(api):
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child", choices=("root", "leaf"), help=argparse.SUPPRESS)
    parser.add_argument("--state", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if os.name != "nt":
        parser.error("actual ownership smoke requires Windows 10/11")
    if options.worker or options.child:
        if options.state is None or options.scenario == "all":
            parser.error("internal worker arguments missing")
        return worker(options.state, options.scenario) if options.worker else child(options.child, options.state, options.scenario)
    return smoke(SCENARIOS if options.scenario == "all" else (options.scenario,))


if __name__ == "__main__":
    sys.exit(main())
