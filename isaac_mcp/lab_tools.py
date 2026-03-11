"""Generic MCP tools for project-owned Isaac Lab runtimes."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def register(mcp, get_lab_connection, logger) -> None:
    """Register generic Isaac Lab runtime tools on the MCP server."""

    def _send_lab_command(command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        lab_runtime = get_lab_connection()
        return lab_runtime.send_command(command_type, params or {})

    @mcp.tool()
    def lab_runtime_ping(ctx, project_root: Optional[str] = None) -> str:
        """Check that a project-owned Isaac Lab runtime bridge is alive."""
        try:
            result = _send_lab_command("ping", {"project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error pinging Isaac Lab runtime bridge: {str(e)}")
            return f"Error pinging Isaac Lab runtime bridge: {str(e)}"

    @mcp.tool()
    def lab_runtime_call(
        ctx,
        command: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call a project-defined command on the active Isaac Lab runtime bridge."""
        try:
            result = _send_lab_command(command, params or {})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error calling Isaac Lab runtime command '{command}': {str(e)}")
            return f"Error calling Isaac Lab runtime command '{command}': {str(e)}"
