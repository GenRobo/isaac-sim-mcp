#!/bin/bash

# Isaac Sim MCP Launcher
# Starts Isaac Sim with the MCP extension enabled

# --- Configuration ---
# Set ISAAC_SIM_ROOT environment variable, or it will use these defaults:

# For Isaac Sim 5.x installed via Omniverse Launcher:
DEFAULT_ISAAC_SIM="$HOME/.local/share/ov/pkg/isaac-sim-5.1.0"

# For Isaac Sim 4.x:
# DEFAULT_ISAAC_SIM="$HOME/.local/share/ov/pkg/isaac-sim-4.2.0"

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$DEFAULT_ISAAC_SIM}"

# Path to this MCP extension folder (auto-detected from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_EXT_FOLDER="$SCRIPT_DIR"
# ---------------------

echo "========================================"
echo " Isaac Sim MCP Launcher"
echo "========================================"
echo ""
echo "Isaac Sim:     $ISAAC_SIM_ROOT"
echo "MCP Extension: $MCP_EXT_FOLDER"
echo ""

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM_ROOT/isaac-sim.sh" ]; then
    echo "ERROR: Isaac Sim not found at: $ISAAC_SIM_ROOT"
    echo ""
    echo "Please set ISAAC_SIM_ROOT environment variable or update this script."
    echo ""
    echo "Common locations:"
    echo "  - Omniverse Launcher: ~/.local/share/ov/pkg/isaac-sim-5.1.0"
    echo "  - Built from source:  /path/to/IsaacSim/_build/linux-x86_64/release"
    echo ""
    echo "Example:"
    echo "  export ISAAC_SIM_ROOT=/path/to/isaac-sim"
    echo "  ./launch_isaac_mcp.sh"
    exit 1
fi

# Check if MCP extension exists
if [ ! -d "$MCP_EXT_FOLDER/isaac.sim.mcp_extension" ]; then
    echo "ERROR: MCP extension not found at: $MCP_EXT_FOLDER/isaac.sim.mcp_extension"
    exit 1
fi

# Launch Isaac Sim with MCP extension
echo "Starting Isaac Sim with MCP extension..."
echo ""
echo "Command: isaac-sim.sh --ext-folder \"$MCP_EXT_FOLDER\" --enable isaac.sim.mcp_extension"
echo ""
echo "Once Isaac Sim loads, look for:"
echo "  \"Isaac Sim MCP server started on localhost:8766\""
echo ""

cd "$ISAAC_SIM_ROOT"
./isaac-sim.sh --ext-folder "$MCP_EXT_FOLDER" --enable isaac.sim.mcp_extension "$@"
