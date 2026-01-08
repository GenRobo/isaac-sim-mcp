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

# isaac_sim_mcp_server.py
import time
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsaacMCPServer")

@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Isaac: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(300.0)  # Match the extension's timeout
        
        try:
            while True:
                try:
                    logger.info("Waiting for data from Isaac")
                    #time.sleep(0.5)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(300.0)  # Match the extension's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Isaac"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Isaac")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Isaac response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Isaac: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Isaac: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Isaac: {str(e)}")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Isaac: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    
    try:
        # Just log that we're starting up
        logger.info("IsaacMCP server starting up")
        
        # Try to connect to Isaac on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            isaac = get_isaac_connection()
            logger.info("Successfully connected to Isaac on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Isaac on startup: {str(e)}")
            logger.warning("Make sure the Isaac addon is running before using Isaac resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _isaac_connection
        if _isaac_connection:
            logger.info("Disconnecting from Isaac Sim on shutdown")
            _isaac_connection.disconnect()
            _isaac_connection = None
        logger.info("Isaac SimMCP server shut down")

# Create the MCP server
mcp = FastMCP("IsaacSimMCP")

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None
# _polyhaven_enabled = False  # Add this global variable

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    
    # If we have an existing connection, check if it's still valid
    if _isaac_connection is not None:
        try:
            
            return _isaac_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None
    
    # Create a new connection if needed
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host="localhost", port=8766)
        if not _isaac_connection.connect():
            logger.error("Failed to connect to Isaac")
            _isaac_connection = None
            raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
        logger.info("Created new persistent connection to Isaac")
    
    return _isaac_connection


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_scene_info")
        print("result: ", result)
        
        # Just return the JSON representation of what Isaac sent us
        return json.dumps(result, indent=2)
        # return json.dumps(result)
        # return result
    except Exception as e:
        logger.error(f"Error getting scene info from Isaac: {str(e)}")
        return f"Error getting scene info: {str(e)}"

@mcp.tool()
def list_robots(ctx: Context) -> str:
    """
    List all robots (articulation roots) in the current Isaac Sim scene.
    Returns paths and basic info for each articulation found.
    """
    code = '''
import omni.usd
from pxr import UsdPhysics
import json

stage = omni.usd.get_context().get_stage()
robots = []

for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        robot_info = {
            "path": str(prim.GetPath()),
            "name": prim.GetName(),
            "type": prim.GetTypeName()
        }
        
        # Count joints under this articulation
        joint_count = 0
        rigid_body_count = 0
        for child in stage.Traverse():
            child_path = str(child.GetPath())
            if child_path.startswith(str(prim.GetPath())):
                if child.IsA(UsdPhysics.Joint):
                    joint_count += 1
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body_count += 1
        
        robot_info["joint_count"] = joint_count
        robot_info["rigid_body_count"] = rigid_body_count
        robots.append(robot_info)

result = json.dumps({"robots": robots, "count": len(robots)}, indent=2)
print(result)
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Robots in scene:\n{output}"
    except Exception as e:
        logger.error(f"Error listing robots: {str(e)}")
        return f"Error listing robots: {str(e)}"


@mcp.tool()
def get_articulation_info(ctx: Context, prim_path: str) -> str:
    """
    Get detailed articulation info for a robot: DOFs, joint names, limits, drives.
    
    Parameters:
    - prim_path: USD path to the robot articulation root (e.g., "/World/spot" or "/spot")
    """
    code = f'''
import omni.usd
from pxr import UsdPhysics, Gf
import json

stage = omni.usd.get_context().get_stage()
root_path = "{prim_path}"
root_prim = stage.GetPrimAtPath(root_path)

if not root_prim:
    print(json.dumps({{"error": f"Prim not found: {{root_path}}"}}))
else:
    info = {{
        "path": root_path,
        "has_articulation": root_prim.HasAPI(UsdPhysics.ArticulationRootAPI),
        "joints": [],
        "rigid_bodies": []
    }}
    
    # Get articulation settings if available
    if root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        try:
            from pxr import PhysxSchema
            if root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                physx = PhysxSchema.PhysxArticulationAPI(root_prim)
                info["solver_position_iterations"] = physx.GetSolverPositionIterationCountAttr().Get()
                info["solver_velocity_iterations"] = physx.GetSolverVelocityIterationCountAttr().Get()
                info["enabled_self_collisions"] = physx.GetEnabledSelfCollisionsAttr().Get()
        except:
            pass
    
    # Find all joints
    for prim in stage.Traverse():
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(root_path):
            continue
            
        if prim.IsA(UsdPhysics.Joint):
            joint_info = {{
                "path": prim_path_str,
                "name": prim.GetName(),
                "type": prim.GetTypeName()
            }}
            
            # Get joint properties
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if body0:
                joint_info["body0"] = str(body0[0])
            if body1:
                joint_info["body1"] = str(body1[0])
            
            # Get limits for revolute/prismatic
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rev = UsdPhysics.RevoluteJoint(prim)
                joint_info["lower_limit"] = rev.GetLowerLimitAttr().Get()
                joint_info["upper_limit"] = rev.GetUpperLimitAttr().Get()
                joint_info["axis"] = rev.GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                pris = UsdPhysics.PrismaticJoint(prim)
                joint_info["lower_limit"] = pris.GetLowerLimitAttr().Get()
                joint_info["upper_limit"] = pris.GetUpperLimitAttr().Get()
                joint_info["axis"] = pris.GetAxisAttr().Get()
            
            # Get drive info
            for drive_type in ["angular", "linear"]:
                try:
                    drive = UsdPhysics.DriveAPI(prim, drive_type)
                    if drive:
                        stiffness = drive.GetStiffnessAttr().Get()
                        damping = drive.GetDampingAttr().Get()
                        if stiffness is not None or damping is not None:
                            joint_info["drive_type"] = drive_type
                            joint_info["stiffness"] = stiffness
                            joint_info["damping"] = damping
                            max_force = drive.GetMaxForceAttr().Get()
                            if max_force:
                                joint_info["max_force"] = max_force
                            break
                except:
                    pass
            
            info["joints"].append(joint_info)
        
        # Count rigid bodies
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_info = {{"path": prim_path_str, "name": prim.GetName()}}
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                if mass:
                    body_info["mass"] = mass
            info["rigid_bodies"].append(body_info)
    
    info["joint_count"] = len(info["joints"])
    info["rigid_body_count"] = len(info["rigid_bodies"])
    print(json.dumps(info, indent=2, default=str))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Articulation info for {prim_path}:\n{output}"
    except Exception as e:
        logger.error(f"Error getting articulation info: {str(e)}")
        return f"Error getting articulation info: {str(e)}"


@mcp.tool()
def get_joint_states(ctx: Context, prim_path: str) -> str:
    """
    Get current joint positions and velocities for a robot articulation.
    
    Parameters:
    - prim_path: USD path to the robot articulation root (e.g., "/World/spot" or "/spot")
    """
    code = f'''
import omni.usd
from pxr import UsdPhysics
import json

try:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core import World
    
    # Get or create world
    my_world = World.instance()
    if my_world is None:
        my_world = World(stage_units_in_meters=1.0)
    
    art = Articulation("{prim_path}")
    art.initialize()
    
    joint_names = art.dof_names
    joint_positions = art.get_joint_positions().tolist() if art.get_joint_positions() is not None else []
    joint_velocities = art.get_joint_velocities().tolist() if art.get_joint_velocities() is not None else []
    
    states = {{
        "prim_path": "{prim_path}",
        "num_dofs": art.num_dof,
        "joint_names": list(joint_names) if joint_names else [],
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities
    }}
    
    # Combine into per-joint info
    joints = []
    for i, name in enumerate(joint_names or []):
        joint = {{"name": name}}
        if i < len(joint_positions):
            joint["position"] = joint_positions[i]
        if i < len(joint_velocities):
            joint["velocity"] = joint_velocities[i]
        joints.append(joint)
    states["joints"] = joints
    
    print(json.dumps(states, indent=2))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Joint states for {prim_path}:\n{output}"
    except Exception as e:
        logger.error(f"Error getting joint states: {str(e)}")
        return f"Error getting joint states: {str(e)}"


@mcp.tool()
def set_joint_positions(ctx: Context, prim_path: str, positions: List[float], joint_indices: List[int] = None) -> str:
    """
    Set joint positions for a robot articulation.
    
    Parameters:
    - prim_path: USD path to the robot articulation root
    - positions: List of joint position values
    - joint_indices: Optional list of joint indices to set (if None, sets all joints)
    """
    indices_str = str(joint_indices) if joint_indices else "None"
    positions_str = str(positions)
    
    code = f'''
import json
try:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core import World
    import numpy as np
    
    my_world = World.instance()
    if my_world is None:
        my_world = World(stage_units_in_meters=1.0)
    
    art = Articulation("{prim_path}")
    art.initialize()
    
    positions = np.array({positions_str})
    indices = {indices_str}
    
    if indices is not None:
        art.set_joint_positions(positions, joint_indices=np.array(indices))
    else:
        art.set_joint_positions(positions)
    
    # Get new positions to confirm
    new_positions = art.get_joint_positions().tolist()
    print(json.dumps({{"success": True, "new_positions": new_positions}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint positions result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint positions: {str(e)}")
        return f"Error setting joint positions: {str(e)}"


@mcp.tool("create_physics_scene")
def create_physics_scene(
    objects: List[Dict[str, Any]] = [],
    floor: bool = True,
    gravity: List[float] = [0,  -0.981, 0],
    scene_name: str = "physics_scene"
) -> Dict[str, Any]:
    """Create a physics scene with multiple objects. Before create physics scene, you need to call get_scene_info() first to verify availability of connection.
    
    Args:
        objects: List of objects to create. Each object should have at least 'type' and 'position'. 
        objects  = [
        {"path": "/World/Cube", "type": "Cube", "size": 20, "position": [0, 100, 0]},
        {"path": "/World/Sphere", "type": "Sphere", "radius": 5, "position": [5, 200, 0]},
        {"path": "/World/Cone", "type": "Cone", "height": 8, "radius": 3, "position": [-5, 150, 0]}
         ]
        floor: Whether to create a floor. deafult is True
        gravity: The gravity vector. Default is [0, 0, -981.0] (cm/s^2).
        scene_name: The name of the scene. deafult is "physics_scene"
        
    Returns:
        Dictionary with result information.
    """
    params = {"objects": objects, "floor": floor}
    
    if gravity is not None:
        params["gravity"] = gravity
    if scene_name is not None:
        params["scene_name"] = scene_name
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("create_physics_scene", params)
        return f"create_physics_scene successfully: {result.get('result', '')}, {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error create_physics_scene: {str(e)}")
        return f"Error create_physics_scene: {str(e)}"
    
@mcp.tool("create_robot")
def create_robot(robot_type: str = "g1", position: List[float] = [0, 0, 0]) -> str:
    """Create a robot in the Isaac scene. Directly create robot prim in stage at the right position. For any creation of robot, you need to call create_physics_scene() first. call create_robot() as first attmpt beofre call execute_script().
    
    Args:
        robot_type: The type of robot to create. Available options:
            - "franka": Franka Emika Panda robot
            - "jetbot": NVIDIA JetBot robot
            - "carter": Carter delivery robot
            - "g1": Unitree G1 quadruped robot (default)
            - "go1": Unitree Go1 quadruped robot
        
    Returns:
        String with result information.
    """
    isaac = get_isaac_connection()
    result = isaac.send_command("create_robot", {"robot_type": robot_type, "position": position})
    return f"create_robot successfully: {result.get('result', '')}, {result.get('message', '')}"

@mcp.tool("omni_kit_command")
def omni_kit_command(command: str = "CreatePrim", prim_type: str = "Sphere") -> str:
    """Execute an Omni Kit command.
    
    Args:
        command: The Omni Kit command to execute.
        prim_type: The primitive type for the command.
        
    Returns:
        String with result information.
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("omini_kit_command", {
            "command": command,
            "prim_type": prim_type
        })
        return f"Omni Kit command executed successfully: {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error executing Omni Kit command: {str(e)}")
        return f"Error executing Omni Kit command: {str(e)}"


@mcp.tool()
def execute_script(ctx: Context, code: str) -> str:
    """
    Before execute script pls check prompt from asset_creation_strategy() to ensure the scene is properly initialized.
    Execute arbitrary Python code in Isaac Sim. Before executing any code, first verify if get_scene_info() has been called to ensure the scene is properly initialized. Always print the formatted code into chat to confirm before execution to confirm its correctness. 
    Before execute script pls check if create_physics_scene() has been called to ensure the physics scene is properly initialized.
    When working with robots, always try using the create_robot() function first before resorting to execute_script(). The create_robot() function provides a simpler, more reliable way to add robots to your scene with proper initialization and positioning. Only use execute_script() for robot creation when you need custom configurations or behaviors not supported by create_robot().
    
    For physics simulation, avoid using simulation_context to run simulations in the main thread as this can cause blocking. Instead, use the World class with async methods for initializing physics and running simulations. For example, use my_world = World(physics_dt=1.0/60.0) and my_world.step_async() in a loop, which allows for better performance and responsiveness. If you need to wait for physics to stabilize, consider using my_world.play() followed by multiple step_async() calls.
    To create an simulation of Franka robot, the code should be like this:
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")




    To control the Franka robot, the code should be like this:

from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
art = Articulation("/Franka")
art.initialize(my_world.physics_sim_view)
dof_ptr = art.get_dof_index("panda_joint2")

simulation_context.play()
# NOTE: before interacting with dc directly you need to step physics for one step at least
# simulation_context.step(render=True) which happens inside .play()
for i in range(1000):
    art.set_joint_positions([-1.5], [dof_ptr])
    simulation_context.step(render=True)

simulation_context.stop()


    
    Parameters:
    - code: The Python code to execute, e.g. "omni.kit.commands.execute("CreatePrim", prim_type="Sphere")"
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        print("code: ", code)
        
        result = isaac.send_command("execute_script", {"code": code})
        print("result: ", result)
        return f"Script executed. Result: {result}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"


@mcp.tool()
def execute_script_file(ctx: Context, file_path: str, variables: str = None) -> str:
    """
    Execute a Python script file in Isaac Sim. The script file should be a .py file
    that can be executed in the Isaac Sim Python environment.
    
    This is useful for running pre-written scripts from the isaac-tools repository
    or other locations without having to paste the entire code.
    
    Parameters:
    - file_path: Absolute path to the Python script file to execute
    - variables: Optional JSON string of variables to inject into the script namespace
                 e.g. '{"robot_name": "franka", "position": [0, 0, 0]}'
    
    Example:
        execute_script_file("D:/github/isaac-tools/scripts/inspect_stage.py")
        execute_script_file("D:/github/isaac-tools/scripts/spawn_robot.py", '{"robot_type": "franka"}')
    """
    try:
        # Read the script file
        script_path = Path(file_path)
        if not script_path.exists():
            return f"Error: Script file not found: {file_path}"
        
        if not script_path.suffix == '.py':
            return f"Error: File must be a .py file: {file_path}"
        
        code = script_path.read_text(encoding='utf-8')
        
        # If variables are provided, prepend them to the script
        if variables:
            try:
                import json as json_module
                vars_dict = json_module.loads(variables)
                var_lines = []
                for key, value in vars_dict.items():
                    if isinstance(value, str):
                        var_lines.append(f'{key} = "{value}"')
                    else:
                        var_lines.append(f'{key} = {value}')
                vars_code = '\n'.join(var_lines) + '\n\n'
                code = vars_code + code
            except json.JSONDecodeError as e:
                return f"Error parsing variables JSON: {str(e)}"
        
        # Get the global connection
        isaac = get_isaac_connection()
        logger.info(f"Executing script file: {file_path}")
        
        result = isaac.send_command("execute_script", {"code": code})
        return f"Script file executed: {file_path}\nResult: {result}"
    except Exception as e:
        logger.error(f"Error executing script file: {str(e)}")
        return f"Error executing script file: {str(e)}"

                
@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Isaac Sim"""
    return """
    0. Before anything, always check the scene from get_scene_info(), retrive rool path of assset through return value of assets_root_path.
    1. If the scene is empty, create a physics scene with create_physics_scene()
    2. if execute script due to communication error, then retry 3 times at most

    3. For Franka robot simulation, the code should be like this:
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")


# need to initialize physics getting any articulation..etc
simulation_context = SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()

for i in range(1000):
    simulation_context.step(render=True)

simulation_context.stop()

    4. For Franka robot control, the code should be like this:
    
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path
from pxr import UsdPhysics

def create_physics_scene(stage, scene_path="/World/PhysicsScene"):
    if not stage.GetPrimAtPath(scene_path):
        UsdPhysics.Scene.Define(stage, scene_path)
    
    return stage.GetPrimAtPath(scene_path)

stage = omni.usd.get_context().get_stage()
physics_scene = create_physics_scene(stage)
if not physics_scene:
    raise RuntimeError("Failed to create or find physics scene")
import omni.physics.tensors as physx

def create_simulation_view(stage):
    sim_view = physx.create_simulation_view(stage)
    if not sim_view:
        carb.log_error("Failed to create simulation view")
        return None
    
    return sim_view

sim_view = create_simulation_view(stage)
if not sim_view:
    raise RuntimeError("Failed to create simulation view")

simulation_context = SimulationContext()
assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
art = Articulation("/Franka")
art.initialize()
dof_ptr = art.get_dof_index("panda_joint2")

simulation_context.play()
# NOTE: before interacting with dc directly you need to step physics for one step at least
# simulation_context.step(render=True) which happens inside .play()
for i in range(1000):
    art.set_joint_positions([-1.5], [dof_ptr])
    simulation_context.step(render=True)

simulation_context.stop()

    5. For Jetbot simulation, the code should be like this:
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot

simulation_context = SimulationContext()
simulation_context.initialize_physics()

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0, 0.0, 2.0]),
    )
)


create_prim("/DistantLight", "DistantLight")
# need to initialize physics getting any articulation..etc


my_world.scene.add_default_ground_plane()
my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
my_world.reset()

simulation_context.play()
for i in range(10):
    simulation_context.step(render=True) 

i = 0
reset_needed = False
while i < 2000:
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        if i >= 0 and i < 1000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            print(my_jetbot.get_linear_velocity())
        elif i >= 1000 and i < 1300:
            # rotate
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
            print(my_jetbot.get_angular_velocity())
        elif i >= 1300 and i < 2000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
        elif i == 2000:
            i = 0
        i += 1
simulation_context.stop()

6. For G1 simulation, the code should be like this see g1_ok.py


    """


def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i<=0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None


#@mcp.tool()
def get_beaver3d_status(ctx: Context) -> str:
    """
    TODO: Get the status of Beaver3D.
    """
    return "Beaver3D service is Available"



@mcp.tool("generate_3d_from_text_or_image")
def generate_3d_from_text_or_image(
    ctx: Context,
    text_prompt: str = None,
    image_url: str = None,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Generate a 3D model from text or image, load it into the scene and transform it.
    
    Args:
        text_prompt (str, optional): Text prompt for 3D generation
        image_url (str, optional): URL of image for 3D generation
        position (list, optional): Position to place the model [x, y, z]
        scale (list, optional): Scale of the model [x, y, z]
        
    Returns:
        String with the task_id and prim_path information
    """
    if not (text_prompt or image_url):
        return "Error: Either text_prompt or image_url must be provided"
    
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("generate_3d_from_text_or_image", {
            "text_prompt": text_prompt,
            "image_url": image_url,
            "position": position,
            "scale": scale
        })
        
        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"
    
@mcp.tool("search_3d_usd_by_text")
def search_3d_usd_by_text(
    ctx: Context,
    text_prompt: str = None,
    target_path: str = "/World/my_usd",
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Search for a 3D model using text prompt in USD libraries, then load and position it in the scene.
    
    Args:
        text_prompt (str): Text description to search for matching 3D models
        target_path (str, optional): Path where the USD model will be placed in the scene
        position (list, optional): Position coordinates [x, y, z] for placing the model
        scale (list, optional): Scale factors [x, y, z] to resize the model
        
    Returns:
        String with search results including task_id and prim_path of the loaded model
    """
    if not text_prompt:
        return "Error: Either text_prompt or image_url must be provided"
    
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        params = {"text_prompt": text_prompt, 
                  "target_path": target_path}
            
        result = isaac.send_command("search_3d_usd_by_text", params)
        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"

@mcp.tool("transform")
def transform(
    ctx: Context,
    prim_path: str,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Transform a USD model by applying position and scale.
    
    Args:
        prim_path (str): Path to the USD prim to transform
        position (list, optional): The position to set [x, y, z]
        scale (list, optional): The scale to set [x, y, z]
        
    Returns:
        String with transformation result
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("transform", {
            "prim_path": prim_path,
            "position": position,
            "scale": scale
        })
        
        if result.get("status") == "success":
            return f"Successfully transformed model at {prim_path} to position {position} and scale {scale}"
        else:
            return f"Error transforming model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error transforming model: {str(e)}")
        return f"Error transforming model: {str(e)}"

# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()