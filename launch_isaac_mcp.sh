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

# Detect launch mode: Omniverse Launcher (isaac-sim.sh) vs pip-installed (isaacsim entrypoint)
LAUNCHER_SH="$ISAAC_SIM_ROOT/isaac-sim.sh"
PIP_ISAACSIM="$(command -v isaacsim 2>/dev/null || true)"
# ---------------------

echo "========================================"
echo " Isaac Sim MCP Launcher"
echo "========================================"
echo ""
echo "MCP Extension: $MCP_EXT_FOLDER"

if [ -f "$LAUNCHER_SH" ]; then
    MODE="launcher"
    echo "Isaac Sim:     $ISAAC_SIM_ROOT (Omniverse Launcher install)"
elif [ -n "$PIP_ISAACSIM" ]; then
    MODE="pip"
    echo "Isaac Sim:     $PIP_ISAACSIM (pip-installed in current Python env)"
else
    echo ""
    echo "ERROR: Could not find Isaac Sim."
    echo "  - Looked for Launcher install at: $LAUNCHER_SH"
    echo "  - Looked for pip entrypoint 'isaacsim' on PATH (activate the conda env that has it)"
    echo ""
    echo "Either set ISAAC_SIM_ROOT to an Omniverse Launcher install, or activate"
    echo "a Python env where 'pip show isaacsim' succeeds."
    exit 1
fi
echo ""

# Check if MCP extension exists
if [ ! -d "$MCP_EXT_FOLDER/isaac.sim.mcp_extension" ]; then
    echo "ERROR: MCP extension not found at: $MCP_EXT_FOLDER/isaac.sim.mcp_extension"
    exit 1
fi

echo "Starting Isaac Sim with MCP extension..."
echo "Once Isaac Sim loads, look for:"
echo "  \"Isaac Sim MCP server started on localhost:8766\""
echo ""

if [ "$MODE" = "launcher" ]; then
    cd "$ISAAC_SIM_ROOT" || exit 1
    ./isaac-sim.sh --ext-folder "$MCP_EXT_FOLDER" --enable isaac.sim.mcp_extension "$@"
else
    exec "$PIP_ISAACSIM" --ext-folder "$MCP_EXT_FOLDER" --enable isaac.sim.mcp_extension "$@"
fi
