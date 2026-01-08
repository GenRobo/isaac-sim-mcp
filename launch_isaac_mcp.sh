#!/bin/bash

# Isaac Sim MCP Launcher
# Starts Isaac Sim with the MCP extension enabled

# --- Configuration ---
# Set this to your Isaac Sim installation directory
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/.local/share/ov/pkg/isaac-sim-5.1.0}"

# Path to this MCP extension folder (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_EXT_FOLDER="$SCRIPT_DIR"
# ---------------------

echo "========================================"
echo " Isaac Sim MCP Launcher"
echo "========================================"
echo ""
echo "Isaac Sim: $ISAAC_SIM_ROOT"
echo "MCP Extension: $MCP_EXT_FOLDER"
echo ""

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM_ROOT/isaac-sim.sh" ]; then
    echo "ERROR: Isaac Sim not found at: $ISAAC_SIM_ROOT"
    echo "Please set ISAAC_SIM_ROOT environment variable or update this script."
    exit 1
fi

# Launch Isaac Sim with MCP extension
echo "Starting Isaac Sim with MCP extension..."
cd "$ISAAC_SIM_ROOT"
./isaac-sim.sh --ext-folder "$MCP_EXT_FOLDER" --enable isaac.sim.mcp_extension
