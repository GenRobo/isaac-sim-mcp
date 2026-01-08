# Isaac Sim MCP Extension and Server

The MCP Server and its extension leverage the Model Context Protocol (MCP) framework to enable natural language control of NVIDIA Isaac Sim, transforming conversational AI inputs into precise simulation manipulation.

## Features

- Natural language control of Isaac Sim
- Robot inspection and joint control
- Simulation playback control (play/stop/pause/step)
- Physics scene creation and management
- Log capture for debugging
- Dynamic robot positioning and movement
- Custom lighting and scene creation

## Requirements

- NVIDIA Isaac Sim 4.2.0+ or **Isaac Sim 5.x** (tested with 5.1)
- Python 3.10+
- Cursor AI editor for MCP integration

## Quick Start (Windows)

### 1. Clone the Repository

```powershell
cd D:\github
git clone https://github.com/omni-mcp/isaac-sim-mcp
```

### 2. Launch Isaac Sim with MCP Extension

Use the provided launch script:

```powershell
# Edit launch_isaac_mcp.bat to set your Isaac Sim path first
D:\github\isaac-sim-mcp\launch_isaac_mcp.bat
```

Or manually:

```powershell
cd D:\github\IsaacSim\_build\windows-x86_64\release
.\isaac-sim.bat --ext-folder D:\github\isaac-sim-mcp --enable isaac.sim.mcp_extension
```

Verify the extension starts - you should see:
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

Add to your Cursor MCP settings (`%USERPROFILE%\.cursor\mcp.json`):

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

Restart Cursor and verify the `isaac-sim` MCP server appears in the MCP panel.

---

## Quick Start (Linux)

### 1. Clone and Setup

```bash
cd ~/Documents
git clone https://github.com/omni-mcp/isaac-sim-mcp
```

### 2. Launch Isaac Sim

```bash
# Set environment variables for 3D generation (optional)
export BEAVER3D_MODEL=<your beaver3d model name>
export ARK_API_KEY=<Your Beaver3D API Key>
export NVIDIA_API_KEY="<your nvidia api key>"

# Launch Isaac Sim with extension
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0
./isaac-sim.sh --ext-folder ~/Documents/isaac-sim-mcp/ --enable isaac.sim.mcp_extension
```

### 3. Configure Cursor MCP

```json
{
    "mcpServers": {
        "isaac-sim": {
            "command": "uv",
            "args": ["run", "~/Documents/isaac-sim-mcp/isaac_mcp/server.py"]
        }
    }
}
```

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

### 3D Generation (Optional)

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

### Capture Errors

```
# Start log capture, run sim, check for errors
Start capturing logs
Play the simulation for a few seconds
Stop the simulation
Get the captured logs filtered by errors
```

---

## Architecture

```
┌─────────────────┐     Socket      ┌──────────────────────┐
│   Cursor IDE    │◄──────────────►│   MCP Server         │
│   (MCP Client)  │   (localhost)   │   (server.py)        │
└─────────────────┘                 └──────────┬───────────┘
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

1. Check Isaac Sim console for "MCP server started on localhost:8766"
2. Restart Cursor's MCP connection
3. Check firewall isn't blocking localhost:8766

### Commands Timing Out

The extension uses queue-based command processing (one command per frame). Long-running scripts may take time. Check Isaac Sim console for progress.

### Robot Not Falling / Physics Issues

Use the `capture_logs` tool to check for physics errors:
```
capture_logs(action="get", filter_level="error")
```

Common issues:
- Missing physics scene (use `create_physics_scene`)
- Triangle mesh collisions on dynamic bodies
- World-anchored joints

---

## Development

### MCP Inspector (Debug Mode)

```bash
uv run mcp dev ~/Documents/isaac-sim-mcp/isaac_mcp/server.py
# Visit http://localhost:5173
```

### Extension Development

The extension auto-reloads when you modify `extension.py`. Check Isaac Sim's console for errors.

---

## Related Projects

- **[isaac-tools](https://github.com/your-repo/isaac-tools)** - Robot debugging scripts and USD inspection tools

---

## License

MIT License - see LICENSE file for details.
