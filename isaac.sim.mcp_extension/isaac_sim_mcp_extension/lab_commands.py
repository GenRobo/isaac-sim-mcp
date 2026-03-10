"""Isaac Lab-specific command handlers for the MCP extension."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _resolve_project_root(extension, project_root: str | None = None) -> Path:
    root = project_root or getattr(extension, "_lab_project_root", None)
    if root is None:
        raise ValueError("project_root is required before using Isaac Lab project commands")
    root_path = Path(root).resolve()
    extension._lab_project_root = root_path.as_posix()
    return root_path


def _load_runtime_api(extension, project_root: str | None = None):
    project_root_path = _resolve_project_root(extension, project_root)
    source_root = project_root_path / "source" / "lowes_rdc_sim"
    if not source_root.exists():
        raise FileNotFoundError(f"Project source root not found: {source_root}")
    source_root_str = source_root.as_posix()
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

    from lowes_rdc_sim.runtime import api as runtime_api

    return runtime_api


def get_lab_handlers(extension) -> dict[str, Any]:
    """Return typed Lab command handlers bound to the live extension instance."""

    def lab_ping(project_root: str | None = None):
        if project_root is not None:
            _resolve_project_root(extension, project_root)
        return {
            "status": "success",
            "pong": True,
            "project_root": getattr(extension, "_lab_project_root", None),
        }

    def lab_get_runtime_info(project_root: str | None = None):
        runtime_info = {
            "runtime": "isaac_lab",
            "project_root": getattr(extension, "_lab_project_root", None),
        }
        if project_root is not None or getattr(extension, "_lab_project_root", None) is not None:
            api = _load_runtime_api(extension, project_root)
            runtime_info.update(api.get_runtime_info())
        return {"status": "success", "runtime_info": runtime_info}

    def lab_create_scene(project_root: str, scene_name: str = "fanuc_validation", device: str = "cuda:0"):
        api = _load_runtime_api(extension, project_root)
        result = api.create_scene(scene_name=scene_name, device=device)
        return {"status": "success", "scene": result}

    def lab_reset_scene(project_root: str | None = None):
        api = _load_runtime_api(extension, project_root)
        result = api.reset_scene()
        return {"status": "success", "scene": result}

    def lab_step_scene(num_steps: int = 1, project_root: str | None = None):
        api = _load_runtime_api(extension, project_root)
        result = api.step_scene(num_steps=num_steps)
        return {"status": "success", "step": result}

    def lab_list_articulations(project_root: str | None = None):
        api = _load_runtime_api(extension, project_root)
        result = api.list_articulations()
        return {"status": "success", "articulations": result}

    def lab_get_robot_state(robot_name: str = "fanuc", project_root: str | None = None):
        api = _load_runtime_api(extension, project_root)
        result = api.get_robot_state(robot_name=robot_name)
        return {"status": "success", "robot_state": result}

    def lab_apply_robot_action(
        robot_name: str = "fanuc",
        action_mode: str = "position_target",
        axis_targets: dict[str, float] | None = None,
        joint_targets: dict[str, float] | None = None,
        step_after: bool = False,
        num_steps: int = 1,
        project_root: str | None = None,
    ):
        api = _load_runtime_api(extension, project_root)
        action_result = api.apply_robot_action(
            robot_name=robot_name,
            action_mode=action_mode,
            axis_targets=axis_targets,
            joint_targets=joint_targets,
        )
        response: dict[str, Any] = {"status": "success", "action": action_result}
        if step_after:
            response["step"] = api.step_scene(num_steps=num_steps)
        return response

    return {
        "lab_ping": lab_ping,
        "lab_get_runtime_info": lab_get_runtime_info,
        "lab_create_scene": lab_create_scene,
        "lab_reset_scene": lab_reset_scene,
        "lab_step_scene": lab_step_scene,
        "lab_list_articulations": lab_list_articulations,
        "lab_get_robot_state": lab_get_robot_state,
        "lab_apply_robot_action": lab_apply_robot_action,
    }
