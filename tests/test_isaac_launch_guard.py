"""CPU-only tests: never invoke Isaac, nvidia-smi, PowerShell or real child processes."""
import argparse
import json
import os
import unittest
from unittest.mock import patch

import isaac_launch_guard as guard


BASELINE = (guard.GPU("GPU-test", 24576, 1024, 23552),)


class PolicyTests(unittest.TestCase):
    def test_positive_finite_controls(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "no", "1e999"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                guard.positive_number(value)
        self.assertEqual(guard.positive_number("0.1"), 0.1)
        with self.assertRaises(guard.GuardError):
            guard.Policy(margin=12288)

    def test_bad_environment_is_not_an_unlimited_mode(self):
        with patch.dict(os.environ, {"ISAAC_GUARD_RUNTIME_SECONDS": "nan"}, clear=True):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                guard.parse_arguments([])

    def test_csv_rejects_unknown_or_inconsistent_monitoring(self):
        for text in ("", "N/A", "GPU-a, 24000, N/A, 20000", "GPU-a, 24000, nan, 20000",
                     "GPU-a, 24000, -1, 20000", "GPU-a, 24000, 20000, 20000",
                     "GPU-a, 0, 0, 0", "GPU-a, 24000, 1, 23999\nGPU-a, 24000, 1, 23999"):
            with self.subTest(text=text), self.assertRaises(guard.GuardError):
                guard.parse_gpu_csv(text)
        self.assertEqual(guard.parse_gpu_csv("GPU-a, 24576, 1024, 23552\n")[0].used, 1024)

    def test_capacity_limit_reserve_and_headroom(self):
        with self.assertRaisesRegex(guard.GuardError, "explicitly lower"):
            guard.check_memory((guard.GPU("GPU-small", 8192, 0, 8192),), guard.Policy())
        for reading in (guard.GPU("GPU-test", 24576, 12288, 12288),
                        guard.GPU("GPU-test", 24576, 100, 500)):
            with self.assertRaisesRegex(guard.GuardError, "VRAM threshold"):
                guard.check_memory((reading,), guard.Policy())
        with self.assertRaisesRegex(guard.GuardError, "headroom"):
            guard.check_memory((guard.GPU("GPU-test", 24576, 12000, 12576),), guard.Policy(), preflight=True)
        with self.assertRaises(guard.GuardError):
            guard.check_memory(BASELINE + (guard.GPU("GPU-small", 8192, 0, 8192),), guard.Policy())

    def test_monitor_rejects_changed_device_set(self):
        with patch.object(guard.shutil, "which", return_value="fake-nvidia-smi.exe"):
            monitor = guard.NvidiaMonitor()
        with patch.object(guard, "run_query", side_effect=["GPU-a, 24576, 100, 24476", "GPU-b, 24576, 100, 24476"]):
            monitor(1)
            with self.assertRaisesRegex(guard.GuardError, "identities"):
                monitor(1)



class ProcessTests(unittest.TestCase):
    @staticmethod
    def snapshot(name, path=None, command=None):
        return json.dumps([{"ProcessId": 123, "Name": name, "ExecutablePath": path, "CommandLine": command}])

    def test_existing_kit_or_isaac_is_rejected_without_killing(self):
        for name, path, command in (
            ("KiT.ExE", None, None),
            ("other.exe", r"C:\Program Files\Isaac\KIT.EXE", None),
            ("python.exe", r"C:\Isaac\kit\python\python.exe", "python script.py"),
            ("cmd.exe", None, 'cmd /c "C:\\Isaac Sim\\isaac-sim.newton.bat"'),
            ("python3.exe", None, "python3.exe -m isaacsim"),
            ("isaacsim.exe", None, None),
        ):
            with self.subTest(name=name, command=command), self.assertRaisesRegex(guard.GuardError, "existing"):
                guard.check_process_snapshot(self.snapshot(name, path, command))

    def test_unknown_process_state_fails_closed(self):
        for text in ("", "[]", "null", "{}", '[{"Name":"kit.exe"}]', self.snapshot("cmd.exe"), self.snapshot("pythonw.exe")):
            with self.subTest(text=text), self.assertRaises(guard.GuardError):
                guard.check_process_snapshot(text)



class CommandTests(unittest.TestCase):
    def test_shell_metacharacters_fail_instead_of_becoming_commands(self):
        for value in ('x&echo bad', '%PATH%', '!x!', 'x"y', 'x\ny', 'x|y', 'x^y', '(x)', 'x>y', 'x<y', 'x\x00y'):
            with self.subTest(value=value), self.assertRaises(guard.GuardError):
                guard.cmd_argument(value)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeJob:
    started = 0.0

    def __init__(self, code=None):
        self.code = code

    def root_exit_code(self):
        return self.code


class SupervisionTests(unittest.TestCase):
    def run_supervision(self, monitor, runtime=2.5, job=None):
        clock = Clock()
        return guard.supervise(job or FakeJob(), monitor, guard.Policy(runtime=runtime), BASELINE,
                               clock=clock, sleep=clock.sleep)

    def test_deadline_and_poll_timeout_budget(self):
        clock = Clock()
        runtime = 2.5
        def monitor(timeout):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, runtime - clock())
            return BASELINE
        result = guard.supervise(FakeJob(), monitor, guard.Policy(runtime=runtime), BASELINE,
                                 clock=clock, sleep=clock.sleep)
        self.assertEqual(result.code, 124)
        self.assertEqual(result.elapsed, runtime)

    def test_vram_stop_records_peak(self):
        result = self.run_supervision(lambda _: (guard.GPU("GPU-test", 24576, 13000, 11576),))
        self.assertEqual(result.peak, 13000)
        self.assertEqual(result.code, 1)
        self.assertIn("VRAM threshold", result.reason)

    def test_monitor_failure_stops_and_ctrl_c_stops(self):
        def fail(_):
            raise guard.GuardError("lost monitor")
        result = self.run_supervision(fail)
        self.assertIn("lost monitor", result.reason)
        self.assertEqual(result.code, 1)
        def interrupt(_):
            raise KeyboardInterrupt
        self.assertEqual(self.run_supervision(interrupt).code, 130)

    def test_early_launcher_exit_is_a_shutdown_reason(self):
        result = self.run_supervision(lambda _: self.fail("must not query after root exit"), job=FakeJob(0))
        self.assertEqual(result.code, 0)
        self.assertIn("entire owned tree", result.reason)


if __name__ == "__main__":
    unittest.main()
