# Isaac Sim MCP Extension and Server

The MCP Server and its extension leverage the Model Context Protocol (MCP) framework to enable natural language control of NVIDIA Isaac Sim, transforming conversational AI inputs into precise simulation manipulation.

## Features

- Natural language control of Isaac Sim
- Robot inspection and joint control
- Simulation playback control (play/stop/pause/step)
- Physics scene creation and management
- Log capture for debugging
- Dynamic robot positioning and movement
- Execute Python scripts directly in Isaac Sim

## Requirements

- **NVIDIA Isaac Sim 6.0** (tested with 6.0.0-rc.59); for Isaac Sim 4.2–5.1 use commit `0bb9c07` or earlier
- Python 3.10+
- Cursor AI editor for MCP integration

---

## Launch Scripts

The Windows entry point always runs an attached, bounded supervisor. The Linux script is unchanged and **does not provide these safety controls**.

| Platform | Script | Usage |
|----------|--------|-------|
| **Windows 10/11** | `launch_isaac_mcp.bat` | Run from PowerShell; guarded PhysX or Newton launch |
| **Linux** | `launch_isaac_mcp.sh` | Run with `./launch_isaac_mcp.sh`; not guarded |

### Configuration

**Windows:** set `ISAAC_SIM_ROOT` in the calling shell; it is never overwritten. The existing source-build default is `D:\github\IsaacSim\_build\windows-x86_64\release`. The installed 6.0 entrypoints are `isaac-sim.bat` (PhysX) and `isaac-sim.newton.bat` (Newton); the selected file must exist. No other launch executable can be selected through the guard.

```powershell
$env:ISAAC_SIM_ROOT = "D:\isaacsim6"
$env:ISAAC_GUARD_BACKEND = "physx"
.\launch_isaac_mcp.bat

# Alternative future run, after the previous guarded run has stopped:
$env:ISAAC_GUARD_BACKEND = "newton"
.\launch_isaac_mcp.bat
```

Python selection is `ISAAC_GUARD_PYTHON`, then `PYTHON`, then `%ProgramData%\miniconda3\python.exe` if installed, then `py.exe -3`, then `python.exe` on PATH. Overrides must name a Python 3.10+ executable, **without additional arguments or enclosing quotes in the value**. An unavailable override fails rather than silently choosing another interpreter. The launcher does not use `start` or detach; `pythonw`/`pyw` overrides and interpreters without stdout/stderr are refused.

#### Windows guard controls

Batch arguments remain **Isaac application arguments**, including their original order. The guard adds `--ext-folder` and `--enable isaac.sim.mcp_extension`. Configure safety separately through these environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ISAAC_GUARD_BACKEND` | `physx` | `physx` or `newton`; does not modify physics frequency |
| `ISAAC_GUARD_RUNTIME_SECONDS` | `600` | Wall-clock deadline starting immediately before process creation |
| `ISAAC_GUARD_VRAM_LIMIT_MIB` | `12288` | Stop at or above this **total used VRAM on any GPU**, including unrelated applications |
| `ISAAC_GUARD_VRAM_MARGIN_MIB` | `1024` | Minimum free reserve on every GPU and minimum preflight headroom below the used-VRAM limit |

Every numeric control must be positive and finite; there is no unlimited or bypass option. The limit plus reserve must fit **every** NVIDIA GPU reported by `nvidia-smi`; otherwise preflight refuses. For an 8 GiB card, explicitly choosing a 6144 MiB limit with the default 1024 MiB reserve passes the capacity check, but is not a promise that an Isaac workload fits:

```powershell
$env:ISAAC_GUARD_RUNTIME_SECONDS = "120"
$env:ISAAC_GUARD_VRAM_LIMIT_MIB = "6144"
$env:ISAAC_GUARD_VRAM_MARGIN_MIB = "1024"
.\launch_isaac_mcp.bat --no-ros-env
```

Alternatively invoke the same supervisor directly; only options **before** `--` are safety controls:

```powershell
python .\isaac_launch_guard.py --root "$env:ISAAC_SIM_ROOT" --backend newton --runtime-seconds 120 --vram-limit-mib 6144 -- --no-ros-env
```

Paths and Isaac arguments containing literal double quotes, newlines, NUL, or cmd metacharacters `% ! ^ & | < > ( )` are refused instead of risking shell injection through the installed batch files. Ordinary spaces and trailing backslashes are supported. This restriction applies to the repository path and installation path too. Shell syntax is not an application argument; do not append additional shell commands to the batch invocation.

#### What the Windows guard owns and stops

- A machine-wide named mutex serializes cooperating guarded launches. A bounded PowerShell/CIM snapshot rejects existing Isaac/Kit executable names, known batch/module invocations, Kit-directory executables, and unreadable scripting-host process state. Nothing pre-existing is killed.
- `nvidia-smi` must report parseable GPU UUIDs and total/used/free memory. Missing monitoring, unsuitable capacity, insufficient headroom, or changed GPU identities fail closed. This does **not** use per-process memory values, which can be `N/A` under WDDM.
- A non-inheritable Windows Job Object uses kill-on-close. `PROC_THREAD_ATTRIBUTE_JOB_LIST` assigns ownership **atomically at process creation**, before any child can run; there is no launch-then-assign race. Unsupported job setup refuses rather than using a weaker fallback. Descendants cannot break away from this job.
- Deadline, VRAM limit/reserve breach, monitoring/ownership-query failure, Ctrl+C, normal guard exit, and early batch/root exit terminate the **entire owned tree**, not merely `cmd.exe`. Closing or force-killing the guard also closes its sole job handle, so surviving Kit children are terminated by Windows. Shutdown is forced, not a request to save the scene.
- Polling is approximately once per second. Monitoring subprocesses have a three-second maximum timeout, shortened to the remaining runtime after launch. Cleanup waits up to five seconds for job accounting to reach zero and always closes the kill-on-close handle. The final concise summary reports the reason and peak **observed** total usage of the busiest GPU.

This is a conservative containment policy, **not proof against sudden allocations, GPU OOM, driver/hardware failure, or a machine freeze**. A sample can miss a fast spike; shutdown latency includes polling, query time and OS scheduling. Save work before launching, keep an operator present, and do not use direct vendor batch launches as a routine workaround for refusal. A manually launched, renamed, or embedded Isaac process can evade/race preflight; the mutex only coordinates this guard. No renderer or physics-frequency settings are injected. Linux is outside this guard's scope.

#### CPU-only verification (does not launch Isaac)

From this repository in PowerShell, using the installed standard Python (no `pxr`, Kit, or NVIDIA monitoring required):

```powershell
& "$env:ProgramData\miniconda3\python.exe" -m unittest discover -s tests -p "test_isaac_launch_guard.py"
& "$env:ProgramData\miniconda3\python.exe" .\tests\smoke_windows_guard.py
```

The first command runs deterministic policy, parsing, argument-boundary and injected-supervision tests. The Windows-only smoke harness creates harmless Python roots/grandchildren and injects memory readings; it exercises actual atomic job ownership, timeout cleanup, VRAM-triggered cleanup, monitoring failure, early launcher exit, forced guard termination, and mutex exclusion/release. An outer safety job encloses each worker, and harmless children have their own 15-second lifetime. It never queries the GPU or launches Isaac. Use `--scenario deadline`, `vram`, `monitor-failure`, `root-exit`, or `guard-kill` to isolate a smoke case. There is no arbitrary-command testing mode in the product launcher.

Verified on the preparation Windows workstation: 12 regression tests and all five real job-ownership
smoke scenarios passed. The production batch path was also exercised with harmless replacement
launchers for both backend selections: actual GPU/process preflight, paths/arguments containing spaces,
a two-second deadline, and invalid-duration refusal passed. No Isaac/Kit executable or GPU simulation
was launched during this verification.


**Linux configuration (unguarded):**
```bash
export ISAAC_SIM_ROOT=~/.local/share/ov/pkg/isaac-sim-5.1.0
./launch_isaac_mcp.sh
```

---

## Quick Start (Windows)

### 1. Clone the Repository

```powershell
cd D:\github
git clone https://github.com/GenRobo/isaac-sim-mcp.git
cd isaac-sim-mcp
```

### 2. Launch Isaac Sim with MCP Extension

Set the installation and backend as described above, then use the guarded entry point:

```powershell
$env:ISAAC_SIM_ROOT = "D:\isaacsim6"
$env:ISAAC_GUARD_BACKEND = "physx"  # or "newton"
.\launch_isaac_mcp.bat
```

The default run ends after 600 seconds, or earlier on a safety stop. Leave the console attached. A refusal is not permission to launch the vendor batch file manually; resolve the reported process, monitoring, or capacity condition first.

**Verify** - You should see in the Isaac Sim console:
```
Isaac Sim MCP server started on localhost:8766
```

### 3. Setup MCP Server for Cursor

Create a virtual environment and install dependencies:

```powershell
cd D:\github\isaac-sim-mcp
python -m venv .venv
.venv\Scripts\activate
pip install "mcp[cli]"
```

### 4. Configure Cursor

Add to your Cursor MCP settings file at `%USERPROFILE%\.cursor\mcp.json`:

```json
{
    "mcpServers": {
        "isaac-sim": {
            "command": "D:\\github\\isaac-sim-mcp\\.venv\\Scripts\\python.exe",
            "args": ["D:\\github\\isaac-sim-mcp\\isaac_mcp\\server.py"]
        }
    }
}
```

**Restart Cursor** and verify the `isaac-sim` MCP server appears in the MCP panel (click the MCP icon in the sidebar).

---

## Quick Start (Linux)

### 1. Clone the Repository

```bash
cd ~/Documents
git clone https://github.com/GenRobo/isaac-sim-mcp.git
cd isaac-sim-mcp
```

### 2. Launch Isaac Sim with MCP Extension

**Option A: Use the launch script (recommended)**

```bash
# Set your Isaac Sim path (or edit launch_isaac_mcp.sh)
export ISAAC_SIM_ROOT=~/.local/share/ov/pkg/isaac-sim-5.1.0

./launch_isaac_mcp.sh
```

**Option B: Manual launch**

```bash
# For Isaac Sim 5.x:
cd ~/.local/share/ov/pkg/isaac-sim-5.1.0
./isaac-sim.sh --ext-folder ~/Documents/isaac-sim-mcp --enable isaac.sim.mcp_extension
```

**Verify** - You should see:
```
Isaac Sim MCP server started on localhost:8766
```

### 3. Setup MCP Server for Cursor

```bash
cd ~/Documents/isaac-sim-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install "mcp[cli]"
```

### 4. Configure Cursor

Add to `~/.cursor/mcp.json`:

```json
{
    "mcpServers": {
        "isaac-sim": {
            "command": "/home/YOUR_USERNAME/Documents/isaac-sim-mcp/.venv/bin/python",
            "args": ["/home/YOUR_USERNAME/Documents/isaac-sim-mcp/isaac_mcp/server.py"]
        }
    }
}
```

Replace `YOUR_USERNAME` with your actual username, or use absolute paths.

---

## MCP Tools Reference

### Connection & Scene Info

| Tool | Description |
|------|-------------|
| `get_scene_info` | Ping server and get scene info. **Always call first!** |

### Simulation Control

| Tool | Description |
|------|-------------|
| `simulation_control` | Control playback: `play`, `stop`, `pause`, `step`, `status` |
| `capture_logs` | Capture Isaac Sim logs for debugging |

### Robot Inspection

| Tool | Description |
|------|-------------|
| `list_robots` | List all articulations (robots) in the scene |
| `get_articulation_info` | Get detailed joint/body info for a robot |
| `get_joint_states` | Get current joint positions and velocities |
| `set_joint_positions` | Set joint positions for a robot |

### Scene Creation

| Tool | Description |
|------|-------------|
| `create_physics_scene` | Create physics scene with gravity and ground |
| `create_robot` | Spawn a robot (franka, jetbot, carter, g1, go1) |
| `transform` | Move/scale a USD prim |

### Scripting

| Tool | Description |
|------|-------------|
| `execute_script` | Execute Python code in Isaac Sim |
| `execute_script_file` | Execute a .py file from disk |
| `omni_kit_command` | Execute an Omniverse Kit command |

### 3D Generation (Optional - requires API keys)

| Tool | Description |
|------|-------------|
| `generate_3d_from_text_or_image` | Generate 3D model using Beaver3D |
| `search_3d_usd_by_text` | Search USD asset library |

---

## Example Usage

### Basic Scene Setup

```
# In Cursor chat (Agent mode):
Check the Isaac Sim connection using get_scene_info

Create a physics scene with ground plane

Add a Franka robot at position [0, 0, 0]

Play the simulation
```

### Robot Debugging

```
# List all robots in the scene
List the robots in my scene

# Get joint info for a specific robot
Get the articulation info for /World/MyRobot

# Check joint states while simulation runs
Get the joint states for /World/MyRobot
```

### Capture Physics Errors

```
# Start log capture, run sim, check for errors
Start capturing logs
Play the simulation for a few seconds  
Stop the simulation
Get the captured logs filtered by errors
```

### Execute Custom Scripts

```
# Run a fix script from isaac-tools
Execute the script file D:/github/isaac-tools/scripts/fix_vision60_arm.py
```

---

## Architecture

```
┌─────────────────┐     JSON-RPC     ┌──────────────────────┐
│   Cursor IDE    │◄────────────────►│   MCP Server         │
│   (MCP Client)  │                  │   (server.py)        │
└─────────────────┘                  └──────────┬───────────┘
                                                │
                                                │ Socket :8766
                                                ▼
                                     ┌──────────────────────┐
                                     │  Isaac Sim Extension │
                                     │  (extension.py)      │
                                     │                      │
                                     │  - Command queue     │
                                     │  - USD manipulation  │
                                     │  - Physics control   │
                                     └──────────────────────┘
```

---

## Troubleshooting

### MCP Server Not Responding

1. Check Isaac Sim console for `"Isaac Sim MCP server started on localhost:8766"`
2. In Cursor, click the MCP icon and restart the isaac-sim server
3. Check firewall isn't blocking localhost:8766

### Commands Timing Out

The extension uses queue-based command processing (one command per frame). Long-running scripts may take time. Check Isaac Sim console for `[MCP]` log messages.

### Robot Not Falling / Physics Issues

Use the `capture_logs` tool to check for physics errors:
```
capture_logs(action="get", filter_level="error")
```

Common issues:
- Missing physics scene → use `create_physics_scene`
- Triangle mesh collisions on dynamic bodies → use convexHull
- World-anchored joints → check for joints with empty body targets

### Extension Not Loading

Make sure the extension folder path is correct:
```
--ext-folder "D:\github\isaac-sim-mcp"  # NOT the isaac.sim.mcp_extension subfolder
--enable isaac.sim.mcp_extension
```

---

## Related Projects

- **[isaac-tools](https://github.com/GenRobo/isaac-tools)** - Robot debugging scripts and USD inspection tools

---

## License

MIT License - see LICENSE file for details.
