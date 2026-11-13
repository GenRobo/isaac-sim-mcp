"""FastMCP registrations for the asset tools (stage inspection, NVIDIA asset validation, RTX render, Newton drop
test). They need the extension in external-driver mode: a headless host that services the command queue between
its own app.update() calls (see the README); in the default update-event mode the scripts fail at `update()`."""

from __future__ import annotations

import json
from typing import List, Optional

from asset_tools import script


def register(mcp, get_isaac_connection, logger) -> None:
    def run(name, arguments, timeout):
        isaac = get_isaac_connection()
        result = isaac.send_command("execute_script", {"code": script(name, arguments)}, timeout=timeout)
        return json.dumps(result.get("result"), indent=2)

    @mcp.tool()
    def asset_stage_open(ctx, path: str, timeout_seconds: int = 600) -> str:
        """Open a USD stage as the session's current stage and report units, default prim, prim count, rigid bodies and world bounds."""
        try:
            return run("stage_open", {"path": path}, timeout_seconds)
        except Exception as e:
            logger.error(f"asset_stage_open: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def asset_validate(ctx, path: str, timeout_seconds: int = 600) -> str:
        """Run NVIDIA's asset validator (isaacsim.asset.validation rules) on a stage and return its issues."""
        try:
            return run("validate", {"path": path}, timeout_seconds)
        except Exception as e:
            logger.error(f"asset_validate: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def asset_render(ctx, path: str, output_dir: str, frames: int = 1, width: int = 960, height: int = 720,
                     camera_position: Optional[List[float]] = None, look_at: Optional[List[float]] = None,
                     timeout_seconds: int = 900) -> str:
        """Render RTX frames of a stage (dome + distant light, explicit clipping range) into output_dir; a cold shader cache takes minutes."""
        try:
            arguments = {"path": path, "outputDir": output_dir, "frames": frames, "width": width, "height": height}
            if camera_position:
                arguments["cameraPosition"] = camera_position
            if look_at:
                arguments["lookAt"] = look_at
            return run("render", arguments, timeout_seconds)
        except Exception as e:
            logger.error(f"asset_render: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def asset_simulate(ctx, path: str, seconds: float = 3.0, drop_height: float = 0.0, physics_hz: Optional[int] = None,
                       timeout_seconds: int = 900) -> str:
        """Drop a metres/Z-up rigid-body asset onto a ground plane (Isaac's physics rate unless physics_hz is given) and report rest, penetration or divergence."""
        try:
            arguments = {"path": path, "seconds": seconds, "dropHeight": drop_height}
            if physics_hz:
                arguments["physicsHz"] = physics_hz
            return run("simulate", arguments, timeout_seconds)
        except Exception as e:
            logger.error(f"asset_simulate: {e}")
            return f"Error: {e}"
