"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Extension module for Isaac Sim MCP - Refactored for stability."""

import carb
import omni.usd
import omni.kit.app
import threading
import time
import socket
import json
import traceback
import queue
import gc

from pxr import Usd, UsdGeom, Sdf, Gf

import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World

# Import Beaver3d and USDLoader
from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader
from isaac_sim_mcp_extension.usd import USDSearch3d


class MCPExtension(omni.ext.IExt):
    """MCP Extension with queue-based command processing for stability."""
    
    def __init__(self) -> None:
        """Initialize the extension."""
        super().__init__()
        self.ext_id = None
        self.running = False
        self.host = None
        self.port = None
        self.socket = None
        self.server_thread = None
        self._usd_context = None
        self._physx_interface = None
        self._timeline = None
        self._window = None
        self._status_label = None
        self._server_thread = None
        self._models = None
        self._settings = carb.settings.get_settings()
        self._image_url_cache = {}
        self._text_prompt_cache = {}
        self._captured_logs = []
        self._log_capture_enabled = False
        self._max_log_buffer = 500
        
        # NEW: Queue-based command processing
        self._command_queue = queue.Queue()
        self._update_sub = None
        self._processing_command = False
        

    def on_startup(self, ext_id: str):
        """Initialize extension and start server."""
        print(f"[MCP] Starting extension: {ext_id}")
        self.port = self._settings.get("/exts/isaac.sim.mcp/server.port") or 8766
        self.host = self._settings.get("/exts/isaac.sim.mcp/server.host") or "localhost"
        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        
        # Subscribe to update event for command processing
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="mcp_command_processor")
        )
        
        self._start()
    
    def on_shutdown(self):
        """Clean up on shutdown."""
        print(f"[MCP] Shutting down extension: {self.ext_id}")
        
        # Unsubscribe from update events
        if self._update_sub:
            self._update_sub = None
        
        self._models = {}
        gc.collect()
        self._stop()
    
    def _on_update(self, event):
        """Process commands during the update cycle - runs on main thread."""
        if self._processing_command:
            return  # Already processing, skip
        
        try:
            # Process ONE command per frame to avoid blocking
            if not self._command_queue.empty():
                self._processing_command = True
                try:
                    client, command = self._command_queue.get_nowait()
                    self._process_command(client, command)
                except queue.Empty:
                    pass
                finally:
                    self._processing_command = False
        except Exception as e:
            print(f"[MCP] Error in update loop: {e}")
            self._processing_command = False
    
    def _process_command(self, client, command):
        """Process a single command and send response."""
        try:
            response = self.execute_command(command)
            
            # Safely serialize response
            try:
                response_json = json.dumps(response)
            except (TypeError, ValueError) as json_err:
                print(f"[MCP] JSON serialization failed: {json_err}")
                response_json = json.dumps({
                    "status": response.get("status", "success") if isinstance(response, dict) else "success",
                    "message": str(response)
                })
            
            # Log truncated response
            log_response = response_json[:500] + "..." if len(response_json) > 500 else response_json
            print(f"[MCP] Response: {log_response}")
            
            # Send response
            try:
                client.sendall(response_json.encode('utf-8'))
            except Exception as send_err:
                print(f"[MCP] Failed to send response: {send_err}")
                
        except Exception as e:
            print(f"[MCP] Error processing command: {e}")
            traceback.print_exc()
            try:
                error_response = {"status": "error", "message": str(e)}
                client.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
    
    def _start(self):
        """Start the socket server."""
        if self.running:
            print("[MCP] Server already running")
            return
            
        self.running = True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            print(f"[MCP] Server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"[MCP] Failed to start server: {e}")
            self._stop()
            
    def _stop(self):
        """Stop the socket server."""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        
        print("[MCP] Server stopped")

    def _server_loop(self):
        """Main server loop - accepts connections."""
        print("[MCP] Server thread started")
        self.socket.settimeout(1.0)

        while self.running:
            try:
                try:
                    client, address = self.socket.accept()
                    print(f"[MCP] Client connected: {address}")
                    
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[MCP] Error accepting connection: {e}")
                    time.sleep(0.5)
            except Exception as e:
                if self.running:
                    print(f"[MCP] Error in server loop: {e}")
                time.sleep(0.5)
        
        print("[MCP] Server thread stopped")
    
    def _handle_client(self, client):
        """Handle a connected client - receives commands and queues them."""
        print("[MCP] Client handler started")
        client.settimeout(30.0)  # 30 second timeout for receiving data
        buffer = b''
        
        try:
            while self.running:
                try:
                    data = client.recv(16384)
                    if not data:
                        print("[MCP] Client disconnected")
                        break
                    
                    buffer += data
                    
                    try:
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''
                        
                        # Queue the command for processing in the main thread
                        self._command_queue.put((client, command))
                        
                        # Wait for response (the update loop will process it)
                        # We don't need to do anything here - response is sent by _process_command
                        
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        if len(buffer) > 1024 * 1024:  # 1MB limit
                            print("[MCP] Buffer overflow, clearing")
                            buffer = b''
                            
                except socket.timeout:
                    # Check if still running
                    continue
                except Exception as e:
                    print(f"[MCP] Error receiving data: {e}")
                    break
                    
        except Exception as e:
            print(f"[MCP] Error in client handler: {e}")
        finally:
            try:
                client.close()
            except:
                pass
            print("[MCP] Client handler stopped")

    def execute_command(self, command):
        """Execute a command - called from main thread via update loop."""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            if cmd_type in ["create_object", "modify_object", "delete_object"]:
                self._usd_context = omni.usd.get_context()
            
            return self._execute_command_internal(command)
                
        except Exception as e:
            print(f"[MCP] Error executing command: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution."""
        cmd_type = command.get("type")
        params = command.get("params", {})

        handlers = {
            "execute_script": self.execute_script,
            "get_scene_info": self.get_scene_info,
            "omini_kit_command": self.omini_kit_command,
            "create_physics_scene": self.create_physics_scene,
            "create_robot": self.create_robot,
            "generate_3d_from_text_or_image": self.generate_3d_from_text_or_image,
            "transform": self.transform,
            "search_3d_usd_by_text": self.search_3d_usd_by_text,
            "simulation_control": self.simulation_control,
            "capture_logs": self.capture_logs,
        }
        
        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"[MCP] Executing: {cmd_type}")
                result = handler(**params)
                print(f"[MCP] Completed: {cmd_type}")
                
                if result and result.get("status") == "success":   
                    return {"status": "success", "result": result}
                else:
                    return {"status": "error", "message": result.get("message", "Unknown error")}
            except Exception as e:
                print(f"[MCP] Error in handler {cmd_type}: {e}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}

    def execute_script(self, code: str):
        """Execute a Python script within the Isaac Sim context."""
        import io
        import sys
        
        def make_json_safe(obj):
            """Convert object to JSON-serializable format."""
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            return str(obj)
        
        def make_ascii_safe(text):
            """Convert text to ASCII-safe string."""
            if not isinstance(text, str):
                return str(text)
            replacements = {
                '\u2713': '[OK]', '\u2714': '[OK]',
                '\u2717': '[X]', '\u2718': '[X]',
                '\u2192': '->', '\u2190': '<-',
                '\u2022': '*', '\u25cf': '*',
            }
            for unicode_char, ascii_char in replacements.items():
                text = text.replace(unicode_char, ascii_char)
            return text.encode('ascii', errors='replace').decode('ascii')
        
        old_stdout = None
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            local_ns = {
                "omni": omni,
                "carb": carb,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "Sdf": Sdf,
                "Gf": Gf,
            }
            
            def safe_get_prim(path):
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return None
                prim = stage.GetPrimAtPath(path)
                return prim if prim and prim.IsValid() else None
            
            def safe_traverse(root_path):
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                root = stage.GetPrimAtPath(root_path)
                if not root or not root.IsValid():
                    return
                for prim in Usd.PrimRange(root):
                    if prim.IsValid():
                        yield prim
            
            local_ns["safe_get_prim"] = safe_get_prim
            local_ns["safe_traverse"] = safe_traverse
            
            exec(code, local_ns)
            
            output = make_ascii_safe(captured_output.getvalue())
            sys.stdout = old_stdout
            
            result = local_ns.get("result", None)
            result = make_json_safe(result)
            
            return {
                "status": "success",
                "message": "Script executed successfully",
                "result": result,
                "output": output
            }
        except Exception as e:
            if old_stdout is not None:
                sys.stdout = old_stdout
            error_msg = make_ascii_safe(str(e))
            carb.log_error(f"[MCP] Script error: {error_msg}")
            tb = make_ascii_safe(traceback.format_exc())
            return {
                "status": "error",
                "message": error_msg,
                "traceback": tb
            }
        
    def get_scene_info(self):
        """Get current scene information."""
        self._stage = omni.usd.get_context().get_stage()
        assert self._stage is not None
        assets_root_path = get_assets_root_path()
        return {"status": "success", "message": "pong", "assets_root_path": assets_root_path}
    
    def simulation_control(self, action: str = "status", frames: int = 60):
        """Control simulation playback."""
        try:
            timeline = omni.timeline.get_timeline_interface()
            
            result = {"status": "success", "action": action}
            
            if action == "play":
                timeline.play()
                result["message"] = "Simulation started"
            elif action == "stop":
                timeline.stop()
                result["message"] = "Simulation stopped"
            elif action == "pause":
                timeline.pause()
                result["message"] = "Simulation paused"
            elif action == "step":
                frames = min(frames, 60)
                current_time = timeline.get_current_time()
                fps = timeline.get_time_codes_per_seconds()
                new_time = current_time + (frames / fps)
                timeline.set_current_time(new_time)
                result["message"] = f"Advanced timeline by {frames} frames to {new_time:.2f}s"
            elif action == "status":
                result["message"] = "Current simulation status"
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
            
            result["is_playing"] = timeline.is_playing()
            result["is_stopped"] = timeline.is_stopped()
            result["current_time"] = timeline.get_current_time()
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def capture_logs(self, action: str = "get", clear: bool = False, filter_level: str = "all"):
        """Capture and retrieve Isaac Sim console logs."""
        import os
        
        try:
            log_file = carb.settings.get_settings().get("/log/file")
            
            if action == "start":
                self._log_start_line = 0
                if log_file and os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        self._log_start_line = sum(1 for _ in f)
                self._log_capture_enabled = True
                return {"status": "success", "message": "Log capture started", "log_file": log_file}
                
            elif action == "stop":
                self._log_capture_enabled = False
                return {"status": "success", "message": "Log capture stopped"}
                
            elif action == "get":
                logs = []
                
                if log_file and os.path.exists(log_file):
                    start_line = getattr(self, '_log_start_line', 0)
                    
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        all_lines = f.readlines()
                    
                    recent_lines = all_lines[start_line:] if start_line > 0 else all_lines[-500:]
                    
                    for line in recent_lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        level = "info"
                        if "[Error]" in line:
                            level = "error"
                        elif "[Warning]" in line:
                            level = "warning"
                        
                        if filter_level == "error" and level != "error":
                            continue
                        elif filter_level == "warning" and level not in ["error", "warning"]:
                            continue
                        
                        logs.append({"level": level, "message": line[:300]})
                
                if clear and log_file and os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        self._log_start_line = sum(1 for _ in f)
                
                return {
                    "status": "success",
                    "message": f"Retrieved {len(logs)} logs",
                    "logs": logs[-100:],
                    "total_in_range": len(logs)
                }
            
            elif action == "clear":
                if log_file and os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        self._log_start_line = sum(1 for _ in f)
                return {"status": "success", "message": "Log position reset"}
                
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
    def omini_kit_command(self, command: str, prim_type: str) -> Dict[str, Any]:
        """Execute an Omni Kit command."""
        omni.kit.commands.execute(command, prim_type=prim_type)
        return {"status": "success", "message": "command executed"}
    
    def create_robot(self, robot_type: str = "g1", position: List[float] = [0, 0, 0]):
        """Create a robot in the scene."""
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.stage import add_reference_to_stage

        stage = omni.usd.get_context().get_stage()
        assets_root_path = get_assets_root_path()
        
        robot_configs = {
            "franka": ("/Isaac/Robots/Franka/franka_alt_fingers.usd", "/Franka"),
            "jetbot": ("/Isaac/Robots/Jetbot/jetbot.usd", "/Jetbot"),
            "carter": ("/Isaac/Robots/Carter/carter.usd", "/Carter"),
            "g1": ("/Isaac/Robots/Unitree/G1/g1.usd", "/G1"),
            "go1": ("/Isaac/Robots/Unitree/Go1/go1.usd", "/Go1"),
        }
        
        config = robot_configs.get(robot_type.lower(), robot_configs["franka"])
        asset_path = assets_root_path + config[0]
        prim_path = config[1]
        
        add_reference_to_stage(asset_path, prim_path)
        robot_prim = XFormPrim(prim_path=prim_path)
        robot_prim.set_world_pose(position=np.array(position))
        
        return {"status": "success", "message": f"{robot_type} robot created at {prim_path}"}
    
    def create_physics_scene(
            self,
            objects: List[Dict[str, Any]] = [],
            floor: bool = True,
            gravity: List[float] = (0.0, -9.81, 0.0),
            scene_name: str = "None"
        ) -> Dict[str, Any]:
        """Create a physics scene with multiple objects."""
        try:
            gravity = gravity or [0, -9.81, 0]
            scene_name = scene_name or "physics_scene"
            
            stage = omni.usd.get_context().get_stage()
            
            scene_path = "/World/PhysicsScene"
            omni.kit.commands.execute("CreatePrim", prim_path=scene_path, prim_type="PhysicsScene")

            world_path = "/World"
            omni.kit.commands.execute("CreatePrim", prim_path=world_path, prim_type="Xform")
            
            if floor:
                floor_path = "/World/ground"
                omni.kit.commands.execute(
                    "CreatePrim",
                    prim_path=floor_path,
                    prim_type="Plane",
                    attributes={"size": 100.0}
                )
                
            objects_created = 0
            for i, obj in enumerate(objects):
                obj_name = obj.get("name", f"object_{i}")
                obj_type = obj.get("type", "Cube")
                obj_position = obj.get("position", [0, 0, 0])
                obj_rotation = obj.get("rotation", [1, 0, 0, 0])
                obj_scale = obj.get("scale", [1, 1, 1])
                obj_color = obj.get("color", [0.5, 0.5, 0.5, 1.0])
                obj_physics = obj.get("physics_enabled", True)
                obj_mass = obj.get("mass", 1.0)
                obj_kinematic = obj.get("is_kinematic", False)
                
                obj_path = obj.get("path", f"/World/{obj_name}")
                
                if stage.GetPrimAtPath(obj_path):
                    continue
                
                if obj_type in ["Cube", "Sphere", "Cylinder", "Cone", "Plane"]:
                    omni.kit.commands.execute(
                        "CreatePrim",
                        prim_path=obj_path,
                        prim_type=obj_type,
                        attributes={"size": obj.get("size", 100.0)} if obj_type in ["Cube", "Sphere", "Plane"] else {},
                    )
                else:
                    return {"status": "error", "message": f"Invalid object type: {obj_type}"}
                
                omni.kit.commands.execute(
                    "TransformPrimSRT",
                    path=obj_path,
                    new_translation=obj_position,
                    new_rotation_euler=[0, 0, 0],
                    new_scale=obj_scale,
                )
                
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(obj_path))
                if xform and obj_rotation != [1, 0, 0, 0]:
                    quat = Gf.Quatf(obj_rotation[0], obj_rotation[1], obj_rotation[2], obj_rotation[3])
                    xform_op = xform.AddRotateOp()
                    xform_op.Set(quat)
                
                if obj_physics:
                    omni.kit.commands.execute(
                        "CreatePhysics",
                        prim_path=obj_path,
                        physics_type="rigid_body" if not obj_kinematic else "kinematic_body",
                        attributes={"mass": obj_mass, "collision_enabled": True, "kinematic": obj_kinematic}
                    )
                
                if obj_color:
                    material_path = f"{obj_path}/material"
                    omni.kit.commands.execute(
                        "CreatePrim",
                        prim_path=material_path,
                        prim_type="Material",
                        attributes={"diffuseColor": obj_color[:3], "opacity": obj_color[3] if len(obj_color) > 3 else 1.0}
                    )
                    omni.kit.commands.execute("BindMaterial", material_path=material_path, prim_path=obj_path)
                    objects_created += 1
                    
            return {
                "status": "success",
                "message": f"Created physics scene with {objects_created} objects",
                "result": scene_name
            }
                
        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
   
    def generate_3d_from_text_or_image(self, text_prompt=None, image_url=None, position=(0, 0, 50), scale=(10, 10, 10)):
        """Generate a 3D model from text or image."""
        try:
            beaver = Beaver3d()
            
            if not hasattr(self, '_image_url_cache'):
                self._image_url_cache = {}
            if not hasattr(self, '_text_prompt_cache'):
                self._text_prompt_cache = {}
            
            task_id = None
            if image_url and image_url in self._image_url_cache:
                task_id = self._image_url_cache[image_url]
            elif text_prompt and text_prompt in self._text_prompt_cache:
                task_id = self._text_prompt_cache[text_prompt]

            if task_id:
                print(f"[MCP] Using cached model ID: {task_id}")
            elif image_url:
                task_id = beaver.generate_3d_from_image(image_url)
            elif text_prompt:
                task_id = beaver.generate_3d_from_text(text_prompt)
            else:
                return {"status": "error", "message": "Either text_prompt or image_url must be provided"}
            
            def load_model_into_scene(task_id, status, result_path):
                print(f"[MCP] {task_id} is {status}, downloaded to: {result_path}")
                if image_url and image_url not in self._image_url_cache:
                    self._image_url_cache[image_url] = task_id
                elif text_prompt and text_prompt not in self._text_prompt_cache:
                    self._text_prompt_cache[text_prompt] = task_id
                    
                loader = USDLoader()
                prim_path = loader.load_usd_model(task_id=task_id)
                
                try:
                    texture_path, material = loader.load_texture_and_create_material(task_id=task_id)
                    loader.bind_texture_to_model()
                except Exception as e:
                    print(f"[MCP] Texture loading failed: {e}")
                
                loader.transform(position=position, scale=scale)
                return {"status": "success", "task_id": task_id, "prim_path": prim_path}
            
            from omni.kit.async_engine import run_coroutine
            run_coroutine(beaver.monitor_task_status_async(task_id, on_complete_callback=load_model_into_scene))
            
            return {"status": "success", "task_id": task_id, "message": f"3D model generation started: {task_id}"}
            
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def search_3d_usd_by_text(self, text_prompt: str, target_path: str, position=(0, 0, 50), scale=(10, 10, 10)):
        """Search for a USD asset and load it."""
        try:
            if not text_prompt:
                return {"status": "error", "message": "text_prompt must be provided"}
            
            searcher3d = USDSearch3d()
            url = searcher3d.search(text_prompt)

            loader = USDLoader()
            prim_path = loader.load_usd_from_url(url, target_path)
            
            return {
                "status": "success",
                "prim_path": prim_path,
                "message": f"Loaded USD from: {url}"
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def transform(self, prim_path, position=(0, 0, 50), scale=(10, 10, 10)):
        """Transform a USD prim."""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                return {"status": "error", "message": f"Prim not found: {prim_path}"}
            
            loader = USDLoader()
            loader.transform(prim=prim, position=position, scale=scale)
            
            return {
                "status": "success",
                "message": f"Transformed {prim_path}",
                "position": position,
                "scale": scale
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
