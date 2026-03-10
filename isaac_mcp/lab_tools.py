"""Typed Isaac Lab MCP tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def register(mcp, get_isaac_connection, logger) -> None:
    """Register Isaac Lab tools on the existing MCP server."""

    def _send_lab_command(command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        isaac = get_isaac_connection()
        return isaac.send_command(command_type, params or {})

    @mcp.tool()
    def lab_ping(ctx, project_root: Optional[str] = None) -> str:
        """Check that the Isaac Lab command surface is alive."""
        try:
            result = _send_lab_command("lab_ping", {"project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error pinging Isaac Lab command surface: {str(e)}")
            return f"Error pinging Isaac Lab command surface: {str(e)}"

    @mcp.tool()
    def lab_get_runtime_info(ctx, project_root: Optional[str] = None) -> str:
        """Get Isaac Lab runtime and project-runtime status."""
        try:
            result = _send_lab_command("lab_get_runtime_info", {"project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error getting Isaac Lab runtime info: {str(e)}")
            return f"Error getting Isaac Lab runtime info: {str(e)}"

    @mcp.tool()
    def lab_create_scene(ctx, project_root: str, scene_name: str = "fanuc_validation", device: str = "cuda:0") -> str:
        """Create a project-owned Isaac Lab scene by name."""
        try:
            result = _send_lab_command(
                "lab_create_scene",
                {"project_root": project_root, "scene_name": scene_name, "device": device},
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error creating Isaac Lab scene: {str(e)}")
            return f"Error creating Isaac Lab scene: {str(e)}"

    @mcp.tool()
    def lab_reset_scene(ctx, project_root: Optional[str] = None) -> str:
        """Reset the active Isaac Lab scene."""
        try:
            result = _send_lab_command("lab_reset_scene", {"project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error resetting Isaac Lab scene: {str(e)}")
            return f"Error resetting Isaac Lab scene: {str(e)}"

    @mcp.tool()
    def lab_step_scene(ctx, num_steps: int = 1, project_root: Optional[str] = None) -> str:
        """Step the active Isaac Lab scene."""
        try:
            result = _send_lab_command("lab_step_scene", {"num_steps": num_steps, "project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error stepping Isaac Lab scene: {str(e)}")
            return f"Error stepping Isaac Lab scene: {str(e)}"

    @mcp.tool()
    def lab_list_articulations(ctx, project_root: Optional[str] = None) -> str:
        """List project-owned articulations in the active Isaac Lab scene."""
        try:
            result = _send_lab_command("lab_list_articulations", {"project_root": project_root})
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error listing Isaac Lab articulations: {str(e)}")
            return f"Error listing Isaac Lab articulations: {str(e)}"

    @mcp.tool()
    def lab_get_robot_state(ctx, robot_name: str = "fanuc", project_root: Optional[str] = None) -> str:
        """Get a JSON-friendly robot state from the active Isaac Lab scene."""
        try:
            result = _send_lab_command(
                "lab_get_robot_state",
                {"robot_name": robot_name, "project_root": project_root},
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error getting Isaac Lab robot state: {str(e)}")
            return f"Error getting Isaac Lab robot state: {str(e)}"

    @mcp.tool()
    def lab_apply_robot_action(
        ctx,
        robot_name: str = "fanuc",
        action_mode: str = "position_target",
        axis_targets: Optional[Dict[str, float]] = None,
        joint_targets: Optional[Dict[str, float]] = None,
        step_after: bool = False,
        num_steps: int = 1,
        project_root: Optional[str] = None,
    ) -> str:
        """Apply a typed robot action in the active Isaac Lab scene."""
        try:
            result = _send_lab_command(
                "lab_apply_robot_action",
                {
                    "robot_name": robot_name,
                    "action_mode": action_mode,
                    "axis_targets": axis_targets,
                    "joint_targets": joint_targets,
                    "step_after": step_after,
                    "num_steps": num_steps,
                    "project_root": project_root,
                },
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error applying Isaac Lab robot action: {str(e)}")
            return f"Error applying Isaac Lab robot action: {str(e)}"
