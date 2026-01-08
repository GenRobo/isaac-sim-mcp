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

- **NVIDIA Isaac Sim 5.x** (tested with 5.1) or Isaac Sim 4.2.0+
- Python 3.10+
- Cursor AI editor for MCP integration

---

## Launch Scripts

Pre-configured scripts are provided to launch Isaac Sim with the MCP extension:

| Platform | Script | Usage |
|----------|--------|-------|
| **Windows** | `launch_isaac_mcp.bat` | Double-click or run from PowerShell |
| **Linux** | `launch_isaac_mcp.sh` | Run with `./launch_isaac_mcp.sh` |

### Configuration

Before first use, edit the script to set your Isaac Sim installation path:

**Windows** (`launch_isaac_mcp.bat`):
```batch
set ISAAC_SIM_ROOT=D:\github\IsaacSim\_build\windows-x86_64\release
```

**Linux** (`launch_isaac_mcp.sh`):
```bash
export ISAAC_SIM_ROOT=~/.local/share/ov/pkg/isaac-sim-5.1.0
./launch_isaac_mcp.sh
```

Common Isaac Sim locations:
- **Windows (source build):** `D:\github\IsaacSim\_build\windows-x86_64\release`
- **Windows (Omniverse):** `C:\Users\USERNAME\AppData\Local\ov\pkg\isaac-sim-5.1.0`
- **Linux (Omniverse):** `~/.local/share/ov/pkg/isaac-sim-5.1.0`

---

## Quick Start (Windows)

### 1. Clone the Repository

```powershell
cd D:\github
git clone https://github.com/GenRobo/isaac-sim-mcp.git
cd isaac-sim-mcp
```

### 2. Launch Isaac Sim with MCP Extension

**Option A: Use the launch script (recommended)**

Edit `launch_isaac_mcp.bat` to set your Isaac Sim path, then run:

```powershell
.\launch_isaac_mcp.bat
```

**Option B: Manual launch**

```powershell
# For Isaac Sim 5.x built from source:
cd D:\github\IsaacSim\_build\windows-x86_64\release
.\isaac-sim.bat --ext-folder D:\github\isaac-sim-mcp --enable isaac.sim.mcp_extension

# For Isaac Sim installed via Omniverse Launcher:
cd C:\Users\%USERNAME%\AppData\Local\ov\pkg\isaac-sim-5.1.0
.\isaac-sim.bat --ext-folder D:\github\isaac-sim-mcp --enable isaac.sim.mcp_extension
```

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
