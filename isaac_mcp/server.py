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
import threading
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional
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
    command_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    
    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        with self.command_lock:
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
        with self.command_lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception as e:
                    logger.error(f"Error disconnecting from Isaac: {str(e)}")
                finally:
                    self.sock = None

    def receive_full_response(self, sock, buffer_size=16384, timeout=60.0):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a reasonable timeout - 60 seconds for most commands
        sock.settimeout(timeout)
        
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

    def send_command(self, command_type: str, params: Dict[str, Any] = None, timeout: float = 60.0) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        with self.command_lock:
            if not self.sock and not self.connect():
                raise ConnectionError("Not connected to Isaac")
            
            command = {
                "type": command_type,
                "params": params or {},
                # The extension refuses a command it only reaches after this: a timed-out call never runs late.
                "deadline": time.time() + timeout,
            }
            
            try:
                # Serialize command/response traffic on the shared socket.
                logger.info(f"Sending command: {command_type}")
                self.sock.sendall(json.dumps(command).encode('utf-8'))
                logger.info(f"Command sent, waiting for response (timeout={timeout}s)...")
                
                self.sock.settimeout(timeout)
                response_data = self.receive_full_response(self.sock, timeout=timeout)
                logger.info(f"Received {len(response_data)} bytes of data")
                
                response = json.loads(response_data.decode('utf-8'))
                logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
                
                if response.get("status") == "error":
                    logger.error(f"Isaac error: {response.get('message')}")
                    raise Exception(response.get("message", "Unknown error from Isaac"))
                
                return response.get("result", {})
            except socket.timeout:
                logger.error("Socket timeout while waiting for response from Isaac")
                self.disconnect()
                raise Exception("Timeout waiting for Isaac response - try simplifying your request")
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Socket connection error: {str(e)}")
                self.disconnect()
                raise Exception(f"Connection to Isaac lost: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from Isaac: {str(e)}")
                if 'response_data' in locals() and response_data:
                    logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
                self.disconnect()
                raise Exception(f"Invalid response from Isaac: {str(e)}")
            except Exception as e:
                logger.error(f"Error communicating with Isaac: {str(e)}")
                self.disconnect()
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
        global _isaac_connection, _lab_connection
        if _isaac_connection:
            logger.info("Disconnecting from Isaac Sim on shutdown")
            _isaac_connection.disconnect()
            _isaac_connection = None
        if _lab_connection:
            logger.info("Disconnecting from Isaac Lab runtime bridge on shutdown")
            _lab_connection.disconnect()
            _lab_connection = None
        logger.info("Isaac SimMCP server shut down")

# Create the MCP server
mcp = FastMCP("IsaacSimMCP")

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None
_isaac_connection_lock = threading.RLock()
_lab_connection = None
_lab_connection_lock = threading.RLock()
# _polyhaven_enabled = False  # Add this global variable

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection
    with _isaac_connection_lock:
        # Reuse the existing connection and let send_command() handle reconnects
        # under its command lock. Peeking the socket here can race with an active
        # send/receive in another tool call.
        if _isaac_connection is not None and _isaac_connection.sock is not None:
            return _isaac_connection
        
        if _isaac_connection is None:
            _isaac_connection = IsaacConnection(host="localhost", port=8766)
            if not _isaac_connection.connect():
                logger.error("Failed to connect to Isaac")
                _isaac_connection = None
                raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
            logger.info("Created new persistent connection to Isaac")
        
        return _isaac_connection


def get_lab_connection():
    """Get or create a persistent project-owned Isaac Lab runtime connection."""
    global _lab_connection
    with _lab_connection_lock:
        if _lab_connection is not None and _lab_connection.sock is not None:
            return _lab_connection

        if _lab_connection is None:
            _lab_connection = IsaacConnection(host="127.0.0.1", port=8767)
            if not _lab_connection.connect():
                logger.error("Failed to connect to Isaac Lab runtime bridge")
                _lab_connection = None
                raise Exception("Could not connect to Isaac Lab runtime bridge. Make sure the Lab script is running.")
            logger.info("Created new persistent connection to Isaac Lab runtime bridge")

        return _lab_connection


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_scene_info")
        logger.debug("get_scene_info result: %s", result)
        
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
def get_joint_states(
    ctx: Context,
    prim_path: str,
    driven_only: bool = False,
    joint_names: Optional[List[str]] = None,
) -> str:
    """
    Get current joint positions and velocities for a robot articulation.
    
    Parameters:
    - prim_path: USD path to the robot articulation root (e.g., "/World/spot" or "/spot")
    - driven_only: If True, return only joints with a drive target attribute
    - joint_names: Optional list of joint names to include
    """
    joint_names_literal = repr(joint_names)
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
driven_only = {repr(driven_only)}
requested_joint_names = set({joint_names_literal}) if {joint_names_literal} is not None else None

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    joints = []
    joint_root = f"{{search_root}}/joints/"
    for prim in stage.Traverse():
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(joint_root):
            continue

        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        position_attr = prim.GetAttribute("state:angular:physics:position")
        velocity_attr = prim.GetAttribute("state:angular:physics:velocity")
        index_attr = prim.GetAttribute("isaac:physics:index")
        driven = target_attr.IsValid()

        if requested_joint_names is not None and prim.GetName() not in requested_joint_names:
            continue

        has_state = position_attr.IsValid() or velocity_attr.IsValid()
        has_control = driven or index_attr.IsValid()
        if not (has_state or has_control):
            continue
        if driven_only and not driven:
            continue

        index_value = index_attr.Get() if index_attr.IsValid() else None
        joints.append({{
            "name": prim.GetName(),
            "path": prim_path_str,
            "index": index_value,
            "driven": driven,
            "position": position_attr.Get() if position_attr.IsValid() else None,
            "velocity": velocity_attr.Get() if velocity_attr.IsValid() else None,
            "target_position": target_attr.Get() if driven else None,
        }})

    joints.sort(key=lambda joint: (joint["index"] is None, joint["index"] if joint["index"] is not None else 10**9, joint["name"]))

    result = {{
        "prim_path": requested_path,
        "search_root": search_root,
        "driven_only": driven_only,
        "num_dofs": len(joints),
        "joint_names": [joint["name"] for joint in joints],
        "joint_positions": [joint["position"] for joint in joints],
        "joint_velocities": [joint["velocity"] for joint in joints],
        "joints": joints,
    }}
    print(json.dumps(result, indent=2))
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
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
positions = {positions_str}
indices = {indices_str}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    joints = []
    joint_root = f"{{search_root}}/joints/"
    for prim in stage.Traverse():
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(joint_root):
            continue

        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        index_attr = prim.GetAttribute("isaac:physics:index")
        if not target_attr.IsValid():
            continue

        index_value = index_attr.Get() if index_attr.IsValid() else None
        joints.append({{
            "name": prim.GetName(),
            "path": prim_path_str,
            "index": index_value,
            "target_attr": target_attr,
        }})

    joints.sort(key=lambda joint: (joint["index"] is None, joint["index"] if joint["index"] is not None else 10**9, joint["name"]))

    selected = list(range(len(joints))) if indices is None else list(indices)
    if len(selected) != len(positions):
        raise ValueError(f"Expected {{len(selected)}} positions, received {{len(positions)}}")

    updates = []
    for requested_index, target_value in zip(selected, positions):
        if requested_index < 0 or requested_index >= len(joints):
            raise IndexError(f"Joint index out of range: {{requested_index}}")
        joint = joints[requested_index]
        joint["target_attr"].Set(float(target_value))
        updates.append({{
            "index": requested_index,
            "name": joint["name"],
            "target_position": joint["target_attr"].Get(),
        }})

    print(json.dumps({{
        "success": True,
        "prim_path": requested_path,
        "search_root": search_root,
        "updated_joints": updates,
        "joint_order": [joint["name"] for joint in joints],
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint positions result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint positions: {str(e)}")
        return f"Error setting joint positions: {str(e)}"


@mcp.tool()
def get_joint_attributes(
    ctx: Context,
    prim_path: str,
    joint_names: Optional[List[str]] = None,
    attribute_names: Optional[List[str]] = None,
    driven_only: bool = False,
) -> str:
    """
    Get raw USD attributes for robot joints.

    Parameters:
    - prim_path: USD path to the robot articulation root
    - joint_names: Optional list of joint names to include
    - attribute_names: Optional list of attribute names to read
    - driven_only: If True, return only joints with drive target attributes
    """
    joint_names_literal = repr(joint_names)
    attribute_names_literal = repr(
        attribute_names or [
            "drive:angular:physics:targetPosition",
            "drive:angular:physics:stiffness",
            "drive:angular:physics:damping",
            "drive:angular:physics:maxForce",
            "physics:lowerLimit",
            "physics:upperLimit",
            "physics:excludeFromArticulation",
            "physxJoint:jointFriction",
            "state:angular:physics:position",
            "state:angular:physics:velocity",
        ]
    )
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
requested_joint_names = set({joint_names_literal}) if {joint_names_literal} is not None else None
attribute_names = {attribute_names_literal}
driven_only = {repr(driven_only)}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    rows = []
    joint_root = f"{{search_root}}/joints/"
    for prim in stage.Traverse():
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(joint_root):
            continue

        name = prim.GetName()
        if requested_joint_names is not None and name not in requested_joint_names:
            continue

        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        driven = target_attr.IsValid()
        if driven_only and not driven:
            continue

        attrs = {{}}
        for attr_name in attribute_names:
            attr = prim.GetAttribute(attr_name)
            attrs[attr_name] = attr.Get() if attr.IsValid() else None

        rows.append({{
            "name": name,
            "path": prim_path_str,
            "driven": driven,
            "attributes": attrs,
        }})

    rows.sort(key=lambda row: row["name"])
    print(json.dumps({{
        "prim_path": requested_path,
        "search_root": search_root,
        "driven_only": driven_only,
        "attribute_names": attribute_names,
        "joints": rows,
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Joint attributes for {prim_path}:\n{output}"
    except Exception as e:
        logger.error(f"Error getting joint attributes: {str(e)}")
        return f"Error getting joint attributes: {str(e)}"


@mcp.tool()
def set_joint_targets_by_name(ctx: Context, prim_path: str, targets: Dict[str, float]) -> str:
    """
    Set drive target positions by joint name.

    Parameters:
    - prim_path: USD path to the robot articulation root
    - targets: Mapping of joint name to target position in degrees
    """
    targets_literal = repr(targets)
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
targets = {targets_literal}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    updates = []
    for joint_name, target_value in targets.items():
        prim = stage.GetPrimAtPath(f"{{search_root}}/joints/{{joint_name}}")
        if not prim.IsValid():
            raise ValueError(f"Joint not found: {{joint_name}}")
        attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not attr.IsValid():
            raise ValueError(f"Joint has no drive target attribute: {{joint_name}}")
        attr.Set(float(target_value))
        updates.append({{
            "name": joint_name,
            "target_position": attr.Get(),
        }})

    print(json.dumps({{
        "success": True,
        "prim_path": requested_path,
        "search_root": search_root,
        "updated_joints": updates,
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint targets result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint targets by name: {str(e)}")
        return f"Error setting joint targets by name: {str(e)}"


@mcp.tool()
def set_joint_drive_params(
    ctx: Context,
    prim_path: str,
    joint_names: List[str],
    stiffness: Optional[float] = None,
    damping: Optional[float] = None,
    max_force: Optional[float] = None,
) -> str:
    """
    Set drive parameters for named joints.

    Parameters:
    - prim_path: USD path to the robot articulation root
    - joint_names: List of joint names to update
    - stiffness: Optional drive stiffness to apply
    - damping: Optional drive damping to apply
    - max_force: Optional drive max force to apply
    """
    joint_names_literal = repr(joint_names)
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
joint_names = {joint_names_literal}
stiffness = {repr(stiffness)}
damping = {repr(damping)}
max_force = {repr(max_force)}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    updates = []
    for joint_name in joint_names:
        prim = stage.GetPrimAtPath(f"{{search_root}}/joints/{{joint_name}}")
        if not prim.IsValid():
            raise ValueError(f"Joint not found: {{joint_name}}")

        result = {{"name": joint_name}}
        if stiffness is not None:
            attr = prim.GetAttribute("drive:angular:physics:stiffness")
            if not attr.IsValid():
                raise ValueError(f"Joint has no stiffness attribute: {{joint_name}}")
            attr.Set(float(stiffness))
            result["stiffness"] = attr.Get()
        if damping is not None:
            attr = prim.GetAttribute("drive:angular:physics:damping")
            if not attr.IsValid():
                raise ValueError(f"Joint has no damping attribute: {{joint_name}}")
            attr.Set(float(damping))
            result["damping"] = attr.Get()
        if max_force is not None:
            attr = prim.GetAttribute("drive:angular:physics:maxForce")
            if not attr.IsValid():
                raise ValueError(f"Joint has no maxForce attribute: {{joint_name}}")
            attr.Set(float(max_force))
            result["max_force"] = attr.Get()

        updates.append(result)

    print(json.dumps({{
        "success": True,
        "prim_path": requested_path,
        "search_root": search_root,
        "updated_joints": updates,
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint drive params result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint drive params: {str(e)}")
        return f"Error setting joint drive params: {str(e)}"


@mcp.tool()
def set_joint_limits(ctx: Context, prim_path: str, limits: Dict[str, Dict[str, float]]) -> str:
    """
    Set lower and upper limits for named joints.

    Parameters:
    - prim_path: USD path to the robot articulation root
    - limits: Mapping of joint name to {"lower": value, "upper": value}
    """
    limits_literal = repr(limits)
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
limits = {limits_literal}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    updates = []
    for joint_name, limit_cfg in limits.items():
        prim = stage.GetPrimAtPath(f"{{search_root}}/joints/{{joint_name}}")
        if not prim.IsValid():
            raise ValueError(f"Joint not found: {{joint_name}}")

        result = {{"name": joint_name}}
        if "lower" in limit_cfg:
            attr = prim.GetAttribute("physics:lowerLimit")
            if not attr.IsValid():
                raise ValueError(f"Joint has no lower limit attribute: {{joint_name}}")
            attr.Set(float(limit_cfg["lower"]))
            result["lower"] = attr.Get()
        if "upper" in limit_cfg:
            attr = prim.GetAttribute("physics:upperLimit")
            if not attr.IsValid():
                raise ValueError(f"Joint has no upper limit attribute: {{joint_name}}")
            attr.Set(float(limit_cfg["upper"]))
            result["upper"] = attr.Get()

        updates.append(result)

    print(json.dumps({{
        "success": True,
        "prim_path": requested_path,
        "search_root": search_root,
        "updated_joints": updates,
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint limits result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint limits: {str(e)}")
        return f"Error setting joint limits: {str(e)}"


@mcp.tool()
def set_joint_attributes(
    ctx: Context,
    prim_path: str,
    updates: Dict[str, Dict[str, Any]],
) -> str:
    """
    Set arbitrary USD joint attributes for named joints.

    Parameters:
    - prim_path: USD path to the robot articulation root
    - updates: Mapping of joint name to attribute/value mapping
    """
    updates_literal = repr(updates)
    code = f'''
import omni.usd
import json

stage = omni.usd.get_context().get_stage()
requested_path = "{prim_path}"
search_root = requested_path.removesuffix("/root_joint")
updates = {updates_literal}

if not stage or not stage.GetPrimAtPath(search_root).IsValid():
    print(json.dumps({{"error": f"Prim not found: {{requested_path}}"}}))
else:
    applied = []
    for joint_name, attr_updates in updates.items():
        prim = stage.GetPrimAtPath(f"{{search_root}}/joints/{{joint_name}}")
        if not prim.IsValid():
            raise ValueError(f"Joint not found: {{joint_name}}")

        result = {{"name": joint_name, "attributes": {{}}}}
        for attr_name, attr_value in attr_updates.items():
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                raise ValueError(f"Joint attribute not found: {{joint_name}}.{{attr_name}}")
            attr.Set(attr_value)
            result["attributes"][attr_name] = attr.Get()
        applied.append(result)

    print(json.dumps({{
        "success": True,
        "prim_path": requested_path,
        "search_root": search_root,
        "updated_joints": applied,
    }}, indent=2))
'''
    try:
        isaac = get_isaac_connection()
        response = isaac.send_command("execute_script", {"code": code})
        output = response.get("output", "")
        return f"Set joint attributes result:\n{output}"
    except Exception as e:
        logger.error(f"Error setting joint attributes: {str(e)}")
        return f"Error setting joint attributes: {str(e)}"


@mcp.tool()
def simulation_control(ctx: Context, action: str = "status", frames: int = 60) -> str:
    """
    Control Isaac Sim simulation playback.
    
    Parameters:
    - action: One of "play", "stop", "pause", "step", or "status"
    - frames: Number of frames to step (only for "step" action, max 60)
    
    Actions:
    - play: Start the simulation
    - stop: Stop and reset the simulation
    - pause: Pause the simulation (can resume with play)
    - step: Step forward by N frames (max 60 to avoid blocking)
    - status: Get current simulation state
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("simulation_control", {"action": action, "frames": frames})
        
        msg = result.get("message", "")
        is_playing = result.get("is_playing", False)
        current_time = result.get("current_time", 0)
        
        return f"Simulation {action}: {msg}\nPlaying: {is_playing}, Time: {current_time:.2f}s"
    except Exception as e:
        logger.error(f"Error controlling simulation: {str(e)}")
        return f"Error controlling simulation: {str(e)}"


@mcp.tool()
def capture_logs(ctx: Context, action: str = "get", clear: bool = False, filter_level: str = "all") -> str:
    """
    Capture and retrieve Isaac Sim console logs for debugging.
    
    Parameters:
    - action: "start" to begin capturing, "stop" to end, "get" to retrieve, "clear" to empty buffer
    - clear: If True, clear the log buffer after returning logs
    - filter_level: "all", "error", "warning", or "info"
    
    Workflow:
    1. capture_logs(action="start") - Begin capturing
    2. simulation_control(action="play") - Run simulation
    3. simulation_control(action="stop") - Stop simulation  
    4. capture_logs(action="get", filter_level="error") - Get errors
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("capture_logs", {
            "action": action,
            "clear": clear,
            "filter_level": filter_level
        })
        
        if action == "get":
            logs = result.get("logs", [])
            physics_errors = result.get("physics_errors", [])
            
            output = f"Logs captured: {len(logs)}\n"
            
            if physics_errors:
                output += f"\nPHYSICS ERRORS ({len(physics_errors)}):\n"
                for err in physics_errors:
                    output += f"  - {err}\n"
            
            if logs:
                output += f"\nRECENT LOGS:\n"
                for log in logs[-20:]:  # Show last 20
                    level = log.get("level", "?")
                    msg = log.get("message", "")[:200]  # Truncate long messages
                    output += f"  [{level}] {msg}\n"
            
            return output
        else:
            return f"Log capture: {result.get('message', action)}"
            
    except Exception as e:
        logger.error(f"Error with log capture: {str(e)}")
        return f"Error with log capture: {str(e)}"


@mcp.tool("create_physics_scene")
def create_physics_scene(
    objects: List[Dict[str, Any]] = [],
    floor: bool = True,
    gravity: List[float] = [0, 0, -9.81],
    scene_name: str = "physics_scene"
) -> str:
    """Create a physics scene with multiple objects. Before create physics scene, you need to call get_scene_info() first to verify availability of connection.
    Units are stage units (meters in Isaac Sim 6) and the stage is Z-up.

    Args:
        objects: List of objects to create. Each object should have at least 'type' and 'position'.
            type: Cube | Sphere | Cylinder | Cone. Optional keys: size (Cube), radius, height,
            rotation as quaternion [w, x, y, z], scale, color [r, g, b, a], mass, physics_enabled, is_kinematic.
        objects  = [
        {"path": "/World/Cube", "type": "Cube", "size": 0.2, "position": [0, 0, 1.0]},
        {"path": "/World/Sphere", "type": "Sphere", "radius": 0.05, "position": [0.5, 0, 2.0]},
        {"path": "/World/Cone", "type": "Cone", "height": 0.08, "radius": 0.03, "position": [-0.5, 0, 1.5]}
         ]
        floor: Whether to create a floor. deafult is True
        gravity: The gravity vector. Default is [0, 0, -9.81] (m/s^2).
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
        logger.debug("execute_script code length: %s", len(code))
        
        result = isaac.send_command("execute_script", {"code": code})
        logger.debug("execute_script result: %s", result)
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

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_SERVER_DIR))

from lab_tools import register as register_lab_tools
from asset_mcp_tools import register as register_asset_tools

register_lab_tools(mcp, get_lab_connection, logger)
register_asset_tools(mcp, get_isaac_connection, logger)

# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()