"""Conservative Windows-only Isaac supervisor; no Kit imports or GPU initialization."""
from __future__ import annotations

import argparse
import csv
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


DEFAULT_ROOT = r"D:\github\IsaacSim\_build\windows-x86_64\release"
ENTRYPOINTS = {"physx": "isaac-sim.bat", "newton": "isaac-sim.newton.bat"}
POLL_SECONDS = 1.0
COMMAND_TIMEOUT = 3.0
LOCK_NAME = r"Global\IsaacSimMcpLaunchGuard-v1"


class GuardError(RuntimeError):
    pass


def positive_number(value: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


@dataclass(frozen=True)
class Policy:
    runtime: float = 600.0
    limit: float = 12288.0
    margin: float = 1024.0

    def __post_init__(self):
        for value in (self.runtime, self.limit, self.margin):
            if not math.isfinite(value) or value <= 0:
                raise GuardError("runtime, VRAM limit and margin must be positive and finite")
        if self.margin >= self.limit:
            raise GuardError("VRAM margin must be smaller than the VRAM limit")


@dataclass(frozen=True)
class GPU:
    uuid: str
    total: float
    used: float
    free: float


def parse_gpu_csv(text: str) -> tuple[GPU, ...]:
    devices = []
    seen = set()
    try:
        for row in csv.reader(text.splitlines()):
            if len(row) != 4:
                raise ValueError("expected uuid,total,used,free")
            uuid, *values = (item.strip() for item in row)
            total, used, free = map(float, values)
            if not uuid.startswith("GPU-") or uuid in seen:
                raise ValueError("missing or duplicate GPU identity")
            if not all(math.isfinite(n) and n >= 0 for n in (total, used, free)):
                raise ValueError("unknown or invalid memory reading")
            if total <= 0 or used > total or free > total or used + free > total + 2:
                raise ValueError("inconsistent memory reading")
            devices.append(GPU(uuid, total, used, free))
            seen.add(uuid)
    except (ValueError, csv.Error) as exc:
        raise GuardError(f"unparseable NVIDIA memory monitoring: {exc}") from exc
    if not devices:
        raise GuardError("NVIDIA memory monitoring returned no GPUs")
    return tuple(devices)


def check_memory(devices: tuple[GPU, ...], policy: Policy, *, preflight=False) -> None:
    for gpu in devices:
        if policy.limit + policy.margin > gpu.total:
            raise GuardError(
                f"{gpu.uuid}: limit {policy.limit:g} + margin {policy.margin:g} MiB "
                f"exceeds {gpu.total:g} MiB; explicitly lower the VRAM limit"
            )
        if gpu.used >= policy.limit or gpu.free < policy.margin:
            raise GuardError(f"VRAM threshold: {gpu.uuid} used={gpu.used:g}, free={gpu.free:g} MiB")
        if preflight and policy.limit - gpu.used < policy.margin:
            raise GuardError(f"insufficient preflight VRAM headroom below limit on {gpu.uuid}")


def run_query(command: list[str], timeout: float) -> str:
    if timeout <= 0 or not math.isfinite(timeout):
        raise GuardError("monitoring deadline expired")
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="strict",
            timeout=timeout, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise GuardError(f"monitoring command failed ({type(exc).__name__})") from exc


class NvidiaMonitor:
    def __init__(self):
        executable = shutil.which("nvidia-smi.exe")
        if not executable:
            system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            candidate = system / "System32" / "nvidia-smi.exe"
            if candidate.is_file():
                executable = str(candidate)
        if not executable:
            raise GuardError("nvidia-smi.exe is unavailable; refusing to launch")
        self.command = [executable, "--query-gpu=uuid,memory.total,memory.used,memory.free",
                        "--format=csv,noheader,nounits"]
        self.identities = None

    def __call__(self, timeout: float) -> tuple[GPU, ...]:
        devices = parse_gpu_csv(run_query(self.command, timeout))
        identities = {gpu.uuid: gpu.total for gpu in devices}
        if self.identities is not None and identities != self.identities:
            raise GuardError("GPU identities/capacities changed during supervision")
        self.identities = identities
        return devices


# Query all processes, not per-process VRAM (unavailable under WDDM). Protected OS
# processes may omit paths; executable names still come from the complete CIM snapshot.
PROCESS_QUERY = (
    "$ErrorActionPreference='Stop'; "
    "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false); "
    "@(Get-CimInstance Win32_Process -ErrorAction Stop | "
    "Select-Object ProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress"
)
HOST_NAMES = {"cmd.exe", "powershell.exe", "pwsh.exe", "python.exe", "pythonw.exe", "py.exe"}
ISAAC_COMMAND = re.compile(
    r"(?i)(?:^|[\\/\s\"'])"
    r"(?:kit(?:\.exe)?|isaac[-_.]?sim(?:[.\w-]*)\.(?:bat|exe|kit)|"
    r"isaacsim[.\w-]*\.kit)(?=$|[\s\"'])"
)
ISAAC_MODULE = re.compile(r"(?i)(?:^|\s)-m\s+isaacsim(?:\s|$)")


def check_process_snapshot(text: str) -> None:
    try:
        processes = json.loads(text.lstrip("\ufeff"))
        if isinstance(processes, dict):
            processes = [processes]
        if not isinstance(processes, list) or not processes:
            raise ValueError("empty process snapshot")
        seen = set()
        for process in processes:
            pid = process["ProcessId"]
            name = process["Name"]
            if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0 or pid in seen:
                raise ValueError("invalid process identity")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("missing executable name")
            seen.add(pid)
            path = process.get("ExecutablePath")
            command = process.get("CommandLine")
            if path is not None and not isinstance(path, str):
                raise ValueError("invalid executable path")
            if command is not None and not isinstance(command, str):
                raise ValueError("invalid command line")
            base = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
            normalized_path = (path or "").replace("\\", "/").casefold()
            if (base == "kit.exe" or base.startswith(("isaac-sim", "isaac_sim", "isaacsim", "omni.kit"))
                    or "/kit/" in normalized_path or (path and ISAAC_COMMAND.search(path))):
                raise GuardError(f"existing Isaac/Kit process: {name} (PID {pid}); nothing was killed")
            if base in HOST_NAMES or (base.startswith("python") and base.endswith(".exe")):
                if not command:
                    raise GuardError(f"unknown scripting-host process state: {name} (PID {pid})")
                if ISAAC_COMMAND.search(command) or ISAAC_MODULE.search(command):
                    raise GuardError(f"existing Isaac/Kit launcher: {name} (PID {pid}); nothing was killed")
    except (ValueError, TypeError, KeyError) as exc:
        raise GuardError(f"unknown process state: {exc}") from exc


def preflight_processes() -> None:
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = system / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not powershell.is_file():
        raise GuardError("Windows process inspection unavailable")
    check_process_snapshot(run_query(
        [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", PROCESS_QUERY],
        COMMAND_TIMEOUT,
    ))


# The installed batch files expand %* and CALL Kit. Reject shell metacharacters
# rather than pretending C-runtime quoting is sufficient for nested cmd parsing.
def cmd_argument(value: str) -> str:
    if any(ch in value for ch in '\x00\r\n"%!^&|<>()'):
        raise GuardError("batch paths/arguments cannot contain quotes or cmd metacharacters: % ! ^ & | < > ( )")
    # Preserve bare flags: the vendor FOR loop compares --no-ros-env before
    # passing %* to Kit and does not strip quotes from each token.
    if value and not any(ch.isspace() for ch in value):
        return value
    trailing = len(value) - len(value.rstrip("\\"))
    return '"' + value + "\\" * trailing + '"'


def isaac_command(root: Path, extension: Path, backend: str, arguments: list[str]) -> tuple[str, str]:
    entrypoint = root / ENTRYPOINTS[backend]
    if not entrypoint.is_file():
        raise GuardError(f"Isaac entrypoint not found: {entrypoint}")
    if not (extension / "isaac.sim.mcp_extension").is_dir():
        raise GuardError(f"MCP extension not found under {extension}")
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    command_processor = system / "System32" / "cmd.exe"
    if not command_processor.is_file():
        raise GuardError("Windows command processor unavailable")
    tokens = [str(entrypoint), "--ext-folder", str(extension), "--enable", "isaac.sim.mcp_extension", *arguments]
    command = f'{cmd_argument(str(command_processor))} /d /v:off /s /c "' + " ".join(map(cmd_argument, tokens)) + '"'
    return str(command_processor), command


# ctypes definitions are importable on non-Windows for deterministic policy tests.
ULONG_PTR = ctypes.c_size_t
SIZE_T = ctypes.c_size_t
HANDLE = wintypes.HANDLE


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", SIZE_T),
                ("MaximumWorkingSetSize", SIZE_T), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ULONG_PTR), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]


class BASIC_ACCOUNTING(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong), ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", HANDLE), ("hStdOutput", HANDLE), ("hStdError", HANDLE)]


class STARTUPINFOEX(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFO), ("lpAttributeList", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE), ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class WindowsAPI:
    def __init__(self):
        if os.name != "nt":
            raise GuardError("this guard requires Windows 10/11; Linux launcher is not guarded")
        self.dll = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateMutexW": (HANDLE, [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]),
            "CreateJobObjectW": (HANDLE, [ctypes.c_void_p, wintypes.LPCWSTR]),
            "SetInformationJobObject": (wintypes.BOOL, [HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]),
            "QueryInformationJobObject": (wintypes.BOOL, [HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]),
            "InitializeProcThreadAttributeList": (wintypes.BOOL, [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(SIZE_T)]),
            "UpdateProcThreadAttribute": (wintypes.BOOL, [ctypes.c_void_p, wintypes.DWORD, SIZE_T, ctypes.c_void_p, SIZE_T, ctypes.c_void_p, ctypes.c_void_p]),
            "DeleteProcThreadAttributeList": (None, [ctypes.c_void_p]),
            "CreateProcessW": (wintypes.BOOL, [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(PROCESS_INFORMATION)]),
            "WaitForSingleObject": (wintypes.DWORD, [HANDLE, wintypes.DWORD]),
            "GetExitCodeProcess": (wintypes.BOOL, [HANDLE, ctypes.POINTER(wintypes.DWORD)]),
            "TerminateJobObject": (wintypes.BOOL, [HANDLE, wintypes.UINT]),
            "CloseHandle": (wintypes.BOOL, [HANDLE]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.dll, name)
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)

    @staticmethod
    def require(result, operation):
        if not result:
            raise GuardError(f"{operation} failed (Windows error {ctypes.get_last_error()}); refusing unsafe fallback")
        return result


class LaunchLock:
    def __init__(self, api: WindowsAPI):
        self.api = api
        self.handle = None

    def __enter__(self):
        ctypes.set_last_error(0)
        handle = self.api.CreateMutexW(None, True, LOCK_NAME)
        error = ctypes.get_last_error()
        self.api.require(handle, "global launch lock")
        if error == 183:  # ERROR_ALREADY_EXISTS, including another session/user
            self.api.CloseHandle(handle)
            raise GuardError("another guarded launch is active; nothing was killed")
        self.handle = handle
        return self

    def __exit__(self, *_):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class WindowsJob:
    """The sole non-inheritable job handle belongs to the guard, never its children."""
    def __init__(self, api: WindowsAPI):
        self.api = api
        self.handle = None
        self.process = None
        self.pid = None
        self.started = None

    def __enter__(self):
        self.handle = self.api.require(self.api.CreateJobObjectW(None, None), "create ownership job")
        try:
            limits = EXTENDED_LIMIT()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self.api.require(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)), "configure kill-on-close")
        except BaseException:
            self.api.CloseHandle(self.handle)
            self.handle = None
            raise
        return self

    def spawn(self, executable: str, command: str, cwd: Path):
        # JOB_LIST assigns at CreateProcess time. There is NO unowned running or
        # suspended-child window if the guard is killed during process creation.
        size = SIZE_T()
        self.api.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        if not size.value:
            raise GuardError("atomic job assignment unavailable")
        buffer = ctypes.create_string_buffer(size.value)
        self.api.require(self.api.InitializeProcThreadAttributeList(buffer, 1, 0, ctypes.byref(size)), "initialize process attributes")
        try:
            job_handle = HANDLE(self.handle)
            self.api.require(self.api.UpdateProcThreadAttribute(
                buffer, 0, 0x0002000D, ctypes.byref(job_handle), ctypes.sizeof(job_handle), None, None,
            ), "atomic job-list assignment")
            startup = STARTUPINFOEX()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.lpAttributeList = ctypes.cast(buffer, ctypes.c_void_p)
            info = PROCESS_INFORMATION()
            self.started = time.monotonic()  # Includes CreateProcess itself in the deadline.
            self.api.require(self.api.CreateProcessW(
                executable, ctypes.create_unicode_buffer(command), None, None, False,
                0x00080000, None, str(cwd), ctypes.byref(startup), ctypes.byref(info),
            ), "create atomically owned process")
            self.process = info.hProcess
            self.pid = info.dwProcessId
            self.api.CloseHandle(info.hThread)
        finally:
            self.api.DeleteProcThreadAttributeList(buffer)

    def root_exit_code(self):
        status = self.api.WaitForSingleObject(self.process, 0)
        if status == 258:  # WAIT_TIMEOUT
            return None
        if status != 0:
            raise GuardError("unable to query owned root process")
        code = wintypes.DWORD()
        self.api.require(self.api.GetExitCodeProcess(self.process, ctypes.byref(code)), "read root exit code")
        return code.value

    def active_count(self):
        info = BASIC_ACCOUNTING()
        self.api.require(self.api.QueryInformationJobObject(
            self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None,
        ), "query owned process tree")
        return info.ActiveProcesses

    def __exit__(self, *_):
        try:
            if self.handle:
                self.api.require(self.api.TerminateJobObject(self.handle, 1), "terminate owned process tree")
                deadline = time.monotonic() + 5.0
                while self.active_count():
                    if time.monotonic() >= deadline:
                        raise GuardError("owned tree shutdown not yet confirmed; closing kill-on-close job")
                    time.sleep(0.05)
        finally:
            if self.process:
                self.api.CloseHandle(self.process)
                self.process = None
            if self.handle:
                self.api.CloseHandle(self.handle)
                self.handle = None


@dataclass(frozen=True)
class Outcome:
    reason: str
    code: int
    peak: float
    elapsed: float


def supervise(job, monitor, policy: Policy, baseline: tuple[GPU, ...], *, clock=time.monotonic, sleep=time.sleep) -> Outcome:
    peak = max(gpu.used for gpu in baseline)
    reason, code = "guard exit", 1
    deadline = job.started + policy.runtime
    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                reason, code = "runtime deadline", 124
                break
            root_code = job.root_exit_code()
            if root_code is not None:
                reason, code = f"launcher exited ({root_code}); closing entire owned tree", root_code
                break
            devices = monitor(min(COMMAND_TIMEOUT, remaining))
            peak = max(peak, *(gpu.used for gpu in devices))
            if clock() >= deadline:
                reason, code = "runtime deadline", 124
                break
            check_memory(devices, policy)
            sleep(min(POLL_SECONDS, max(0.0, deadline - clock())))
    except KeyboardInterrupt:
        reason, code = "Ctrl+C", 130
    except Exception as exc:
        reason, code = f"guard/monitoring failure: {exc}", 1
    return Outcome(reason, code, peak, max(0.0, clock() - job.started))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", default=os.environ.get("ISAAC_SIM_ROOT", DEFAULT_ROOT))
    parser.add_argument("--backend", choices=ENTRYPOINTS, default=os.environ.get("ISAAC_GUARD_BACKEND", "physx"))
    parser.add_argument("--runtime-seconds", type=positive_number, default=os.environ.get("ISAAC_GUARD_RUNTIME_SECONDS", "600"))
    parser.add_argument("--vram-limit-mib", type=positive_number, default=os.environ.get("ISAAC_GUARD_VRAM_LIMIT_MIB", "12288"))
    parser.add_argument("--vram-margin-mib", type=positive_number, default=os.environ.get("ISAAC_GUARD_VRAM_MARGIN_MIB", "1024"))
    parser.add_argument("isaac_arguments", nargs=argparse.REMAINDER, help="Isaac arguments after --")
    options = parser.parse_args(argv)
    if options.backend not in ENTRYPOINTS:
        parser.error("ISAAC_GUARD_BACKEND must be physx or newton")
    if options.isaac_arguments[:1] == ["--"]:
        options.isaac_arguments.pop(0)
    return options


def main(argv=None) -> int:
    # A GUI interpreter has no attached output and must never reach launch.
    if sys.stdout is None or sys.stderr is None:
        return 1
    outcome = None
    try:
        if sys.version_info < (3, 10):
            raise GuardError("Python 3.10+ is required")
        options = parse_arguments(argv)
        policy = Policy(options.runtime_seconds, options.vram_limit_mib, options.vram_margin_mib)
        api = WindowsAPI()
        root = Path(options.root).resolve()
        executable, command = isaac_command(root, Path(__file__).resolve().parent, options.backend, options.isaac_arguments)
        with LaunchLock(api):
            monitor = NvidiaMonitor()
            baseline = monitor(COMMAND_TIMEOUT)
            check_memory(baseline, policy, preflight=True)
            preflight_processes()
            # Refresh VRAM after the potentially slower process snapshot.
            baseline = monitor(COMMAND_TIMEOUT)
            check_memory(baseline, policy, preflight=True)
            print(f"[guard] {options.backend}; runtime={policy.runtime:g}s; per-GPU total-used cap="
                  f"{policy.limit:g} MiB; reserve={policy.margin:g} MiB; owned-tree shutdown enabled", flush=True)
            with WindowsJob(api) as job:
                job.spawn(executable, command, root)
                outcome = supervise(job, monitor, policy, baseline)
        print(f"[guard] stopped: {outcome.reason}; elapsed={outcome.elapsed:.1f}s; "
              f"peak observed GPU usage={outcome.peak:g} MiB (all processes, busiest GPU)", flush=True)
        return outcome.code
    except KeyboardInterrupt:
        print("[guard] stopped: Ctrl+C; owned job closed", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        detail = f"; peak observed={outcome.peak:g} MiB; reason={outcome.reason}" if outcome else ""
        print(f"[guard] refused/stopped: {exc}{detail}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
