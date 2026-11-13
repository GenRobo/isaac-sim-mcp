# Changelog

All notable changes to the isaac-sim-mcp project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Asset tools (`asset_stage_open`, `asset_validate`, `asset_render`, `asset_simulate`) as FastMCP tools and as reusable `execute_script` templates in `isaac_mcp/asset_tools.py`: stage inspection, NVIDIA's asset validator, RTX renders with an explicit camera clipping range, and a drop test that builds its own stage and references the asset last (editing a freshly opened stage while Kit's loaders run is a USD threading violation). The drop test uses a plane floor (a large box floor makes Newton 1.6.1 accept a nonlocal contact witness, newton-physics/newton#5036, and SolverMuJoCo diverged at first contact for every off-origin body on Isaac Sim 6.2), takes an optional `physics_hz`, and reports `diverged` for a non-finite pose.
- Extension external-driver mode: `MCPExtension.instance`, `drive_externally()` and the public `process_pending_commands()` let a headless `SimulationApp` host service the command queue between its own `app.update()` calls; scripts then get an `update()` helper to advance frames (refused in the default update-event mode, where it would re-enter Kit's update).
- Request deadlines: a command carrying `deadline` (Unix seconds) that the extension only reaches afterwards is refused, never run late; the FastMCP client sends its timeout as the deadline.
- A script's `update()` raises once the command's deadline has passed, so frame loops (stage load, play, render) stop when their caller has given up; `SystemExit`/`KeyboardInterrupt` from a script are reported like other errors instead of ending Kit; error replies keep the traceback and captured output on the wire; results are made JSON-safe element by element (non-finite floats become null). Tests with a stubbed Kit in `tests/test_extension_stubbed.py`.
- `isaac_sim_mcp_extension/protocol.py`: Kit-free protocol helpers (bounded output, JSON-safe results, framing, deadlines, request buffering) with tests in `tests/test_protocol_and_asset_tools.py`.

### Changed
- `execute_script` captures stdout and stderr per execution (bounded to 64 KiB each, UTF-8 kept) instead of swapping the process-wide `sys.stdout` and ASCII-mangling the output; `result` is bounded to 256 KiB of JSON; errors carry a bounded traceback and the captured output.
- Responses are newline-terminated JSON (whitespace to older clients); the listen backlog is 8; an oversized request buffer is dropped through the protocol helper.
- A stdlib-only Windows launch supervisor with a 600-second default deadline, a 12288 MiB total-used-VRAM cap per GPU, configurable positive finite limits/reserve, bounded monitoring queries, and fail-closed process/GPU preflight.
- Atomic Windows Job Object ownership, kill-on-close tree cleanup, and a machine-wide single-launch mutex. Timeout, VRAM pressure, monitoring failure, Ctrl+C, early launcher exit, and guard termination stop only the owned process tree.
- CPU-only regression tests and a harmless Windows Python-process smoke harness for real job ownership, tree termination, and lock behavior; neither launches Isaac nor queries the GPU.

### Changed
- `launch_isaac_mcp.bat` now always invokes the attached supervisor, preserves `ISAAC_SIM_ROOT` and application arguments/MCP flags, and selects standard PhysX (`isaac-sim.bat`) or Newton (`isaac-sim.newton.bat`) through separate guard configuration. Python selection supports executable overrides, ProgramData Miniconda, and normal `py`/`python` resolution.
- Windows launch documentation removes the routine unguarded manual path and explains reserve/capacity checks, argument restrictions, forced shutdown, and limitations against abrupt OOM or hardware failure. The Linux launcher remains unchanged and unguarded.
- The Kit extension no longer depends on `omni.physx`, so it also loads in the Newton-physics app (`isaac-sim.newton.bat`)
- Ported the Kit extension to Isaac Sim 6.0: removed `omni.isaac.*` imports/dependencies in favor of `isaacsim.core.experimental.*` and `isaacsim.storage.native`
- `launch_isaac_mcp.bat` honors an `ISAAC_SIM_ROOT` already set by the caller
- `create_physics_scene` builds prims with `pxr` UsdGeom/UsdPhysics in meters, Z-up: gravity is applied, objects get collision, rigid body, mass and display color, and the floor is a collision plane

### Fixed
- `create_robot` uses the Isaac Sim 6.0 asset paths for Franka, Jetbot and Carter
- `create_physics_scene` MCP tool no longer fails output validation (declared `dict`, returned `str`)
- `get_scene_info` returns a clear error instead of an empty `AssertionError` when no stage is open yet
- `launch_isaac_mcp.bat` calls `isaac-sim.bat` by full path so it works when `NoDefaultCurrentDirectoryInExePath` is set


## [0.3.0] - 2024-04-22

### Added
- USD asset search integration with `search_3d_usd_by_text` tool
- Ability to search and load pre-existing 3D models from USD libraries
- Support for custom positioning and scaling of USD models
- Direct model transformation capabilities with the improved `transform` tool
- Enhanced scene management with multi-object placement

### Improved
- Scene object manipulation with precise positioning controls
- Asset loading performance and reliability
- Error handling for model search and placement
- Integration with existing physics scene management

### Technical Details
- Advanced USD model retrieval system
- Optimized asset loading pipeline
- Position and scale customization for USD models
- Better compatibility with Isaac Sim's native USD handling

## [0.2.1] - 2024-04-15

### Added
- Beaver3D integration for 3D model generation from text prompts and images
- Asynchronous model loading with asyncio support
- Task caching system to prevent duplicate model generation
- New MCP tools:
  - `generate_3d_from_text_or_image` for AI-powered 3D asset creation
  - `transform` for manipulating generated 3D models in the scene
- Texture and material binding for generated 3D models

### Improved
- Asynchronous command execution with `run_coroutine`
- Error handling and reporting for 3D generation tasks
- Performance optimizations for model loading

### Technical Details
- Integration with Beaver3D API for 3D generation
- Task monitoring with callback support
- Position and scale customization for generated models

## [0.1.0] - 2025-04-02

### Added
- Initial implementation of Isaac Sim MCP Extension
- Natural language control interface for Isaac Sim through MCP framework
- Core robot manipulation capabilities:
  - Dynamic placement and positioning of robots (Franka, G1, Go1, Jetbot)
  - Robot movement controls with position updates
  - Multi-robot grid creation (3x3 arrangement support)
- Advanced simulation features:
  - Quadruped robot walking simulation with waypoint navigation
  - Physics-based interactions between robots and environment
  - Custom lighting controls for better scene visualization
- Environment enrichment:
  - Various obstacle types: boxes, spheres, cylinders, cones
  - Wall creation for maze-like environments
  - Dynamic obstacle placement with customizable properties
- Development tools:
  - MCP server integration with Cursor AI
  - Debug interface accessible via local web server
  - Connection status verification with `get_scene_info`
- Documentation:
  - Installation instructions
  - Example prompts for common simulation scenarios
  - Configuration guidelines

### Technical Details
- Extension server running on localhost:8766
- Compatible with NVIDIA Isaac Sim 4.2.0
- Support for Python 3.9+
- MIT License for open development 