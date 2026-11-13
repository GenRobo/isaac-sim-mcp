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

"""Asset tools: scripts the extension's `execute_script` runs for stage inspection, NVIDIA asset validation, RTX
renders and a Newton drop test of one USD asset. Each function returns the code for the extension; `gr_args` is
embedded as a JSON literal so JSON null/true/false never reach Python as names.

The scripts need the extension's external-driver mode (`update()` advances a Kit frame): renders and physics
play cannot happen inside Kit's update event. Measured on Isaac Sim 6.2 / T4 (2026-11-13): Replicator's default camera
clipping hides a 0.3 m prop from under a metre, so cameras get an explicit range; editing a stage while Kit's loaders
still run is a USD threading violation, so the drop test builds its own stage and references the asset last; the
experimental Newton bridge diverges for colliders offset from the body origin, which the drop test reports as
`diverged` rather than a verdict.

No module here imports Kit: it is importable by any client (the FastMCP server, a stdio bridge) and by tests.
"""
import json

STAGE_OPEN = r'''
import omni.usd
import omni.timeline
from pxr import UsdGeom, UsdPhysics
ctx = omni.usd.get_context()
# Settle whatever the previous script left running and let Kit tear the old stage down before opening another:
# a reopen while Replicator still edits the old layer is a USD threading violation.
omni.timeline.get_timeline_interface().stop()
try:
    import omni.replicator.core as rep
    rep.orchestrator.stop()
    rep.orchestrator.wait_until_complete()
except Exception:
    pass
for _ in range(3):
    update()
ctx.close_stage()
for _ in range(3):
    update()
if not ctx.open_stage(gr_args["path"]):
    raise RuntimeError("Kit could not open the stage")
for _ in range(1200):
    status = ctx.get_stage_loading_status()
    if status[2] <= status[1]:
        break
    update()
else:
    raise RuntimeError("The stage is still loading after 1200 updates")
for _ in range(3):
    update()
stage = ctx.get_stage()
root = stage.GetDefaultPrim()
prims = list(stage.Traverse())
bodies = [p.GetPath().pathString for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
bound = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(root).GetRange() if root else None
result = {"path": gr_args["path"], "defaultPrim": root.GetPath().pathString if root else None,
          "primCount": len(prims), "rigidBodies": bodies[:50],
          "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage), "kilogramsPerUnit": UsdPhysics.GetStageKilogramsPerUnit(stage),
          "upAxis": UsdGeom.GetStageUpAxis(stage),
          "worldBound": {"min": list(bound.GetMin()), "max": list(bound.GetMax())} if bound and not bound.IsEmpty() else None}
'''

VALIDATE = STAGE_OPEN + r'''
import omni.asset_validator.core as av
engine = av.ValidationEngine(initRules=True)
results = engine.validate(stage)
issues = []
for issue in (results.issues() if callable(getattr(results, "issues", None)) else getattr(results, "issues", [])):
    rule = getattr(issue, "rule", None)
    issues.append({"rule": getattr(rule, "__name__", str(rule)), "severity": str(getattr(issue, "severity", "")),
                   "message": str(getattr(issue, "message", ""))[:400], "at": str(getattr(issue, "at", ""))[:200]})
result = {"path": gr_args["path"], "ruleCount": len(engine.rules) if hasattr(engine, "rules") else None,
          "issueCount": len(issues), "issues": issues[:200]}
'''

RENDER = STAGE_OPEN + r'''
import os
import time
import omni.replicator.core as rep
centre = [(bound.GetMin()[i] + bound.GetMax()[i]) / 2 for i in range(3)] if bound and not bound.IsEmpty() else [0, 0, 0]
size = max([bound.GetMax()[i] - bound.GetMin()[i] for i in range(3)] + [0.1]) if bound and not bound.IsEmpty() else 1.0
look_at = gr_args.get("lookAt") or centre
position = gr_args.get("cameraPosition") or [centre[0] + 1.6 * size, centre[1] - 1.6 * size, centre[2] + 1.1 * size]
os.makedirs(gr_args["outputDir"], exist_ok=True)
started_at = time.time()
with rep.new_layer(name="gr_render"):
    rep.create.light(light_type="dome", intensity=300)
    rep.create.light(light_type="distant", intensity=1500, rotation=(315, 0, 0))
    # Replicator's default near clip hides a prop that fills the frame from under a metre away.
    camera = rep.create.camera(position=tuple(position), look_at=tuple(look_at), clipping_range=(0.01, 1000.0))
    product = rep.create.render_product(camera, (gr_args.get("width", 960), gr_args.get("height", 720)))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=gr_args["outputDir"], rgb=True)
    writer.attach([product])
    # A paused timeline keeps physics out of a render step; the subframes let the path tracer converge per frame.
    for _ in range(gr_args.get("frames", 1)):
        rep.orchestrator.step(rt_subframes=16, pause_timeline=True)
    rep.orchestrator.wait_until_complete()
    writer.detach()
    product.destroy()
rep.orchestrator.stop()
# Only this call's frames: the directory may hold earlier renders.
files = sorted(name for name in os.listdir(gr_args["outputDir"])
               if name.endswith(".png") and os.path.getmtime(os.path.join(gr_args["outputDir"], name)) >= started_at - 1)
result = {"outputDir": gr_args["outputDir"], "files": files, "cameraPosition": position, "lookAt": look_at,
          "resolution": [gr_args.get("width", 960), gr_args.get("height", 720)]}
'''

SIMULATE = r'''
import math
import time
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.prims import RigidPrim
# Measure the asset with plain USD first: Kit's loaders run on other threads once a stage is open in the context,
# and editing a stage while they run is a USD threading violation. The scene is therefore built in a fresh stage
# before the asset is referenced into it, and nothing is edited afterwards.
asset = Usd.Stage.Open(gr_args["path"])
if asset is None:
    raise RuntimeError("Kit could not open the stage")
if abs(UsdGeom.GetStageMetersPerUnit(asset) - 1.0) > 1e-9 or UsdGeom.GetStageUpAxis(asset) != UsdGeom.Tokens.z:
    raise RuntimeError("the drop test expects a metres, Z-up stage")
root = asset.GetDefaultPrim()
if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI):
    raise RuntimeError("The stage's default prim must be the rigid body")
bound = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(root).GetRange()
if bound.IsEmpty():
    raise RuntimeError("The asset has no renderable bounds")
ctx = omni.usd.get_context()
omni.timeline.get_timeline_interface().stop()
try:
    import omni.replicator.core as rep
    rep.orchestrator.stop()
    rep.orchestrator.wait_until_complete()
except Exception:
    pass
for _ in range(3):
    update()
ctx.new_stage()
for _ in range(3):
    update()
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
stage.SetDefaultPrim(world.GetPrim())
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
ground.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.5))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
drop = gr_args.get("dropHeight", 0.0)
# The asset's lowest render point starts `drop` metres above the ground; the asset keeps its own transforms.
lift = drop - bound.GetMin()[2]
holder = UsdGeom.Xform.Define(stage, Sdf.Path("/World/asset"))
holder.AddTranslateOp().Set(Gf.Vec3d(0, 0, lift))
holder.GetPrim().GetReferences().AddReference(gr_args["path"])
for _ in range(1200):
    status = ctx.get_stage_loading_status()
    if status[2] <= status[1]:
        break
    update()
for _ in range(5):
    update()
bodies = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath("/World/asset")) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
if not bodies:
    raise RuntimeError("The referenced asset exposes no rigid body")
timeline = omni.timeline.get_timeline_interface()
timeline.play()
update()
# While the simulation plays, state is read only through the physics API (the RigidPrim tensor view): Newton and
# PhysX write poses to Fabric, and raw USD xform reads would return the authored start pose.
view = RigidPrim(bodies)
positions, orientations = view.get_world_poses()
start = [[float(v) for v in positions[i]] + [float(v) for v in orientations[i]] for i in range(len(bodies))]
time_start = timeline.get_current_time()
wall = time.time()
# Updates until the timeline has advanced the requested seconds, then half a second more for the rest check.
target = gr_args.get("seconds", 3.0)
updates = 0
while timeline.get_current_time() - time_start < target and updates < 60 * 60:
    update()
    updates += 1
positions, orientations = view.get_world_poses()
mid = [[float(v) for v in positions[i]] for i in range(len(bodies))]
time_mid = timeline.get_current_time()
while timeline.get_current_time() - time_mid < 0.5 and updates < 60 * 60:
    update()
    updates += 1
positions, orientations = view.get_world_poses()
end = [[float(v) for v in positions[i]] + [float(v) for v in orientations[i]] for i in range(len(bodies))]
simulated = timeline.get_current_time() - time_start
timeline.stop()
def finite(values):
    return [round(v, 5) if math.isfinite(v) else None for v in values]
report = []
for i, body in enumerate(bodies):
    diverged = not all(math.isfinite(v) for v in end[i] + mid[i])
    drift = None if diverged else math.dist(mid[i], end[i][:3])
    # Origin-to-lowest-point offset at the start (no rotation assumed): a body whose estimated lowest
    # point ends well below the ground plane fell through it.
    lowest = None if diverged else end[i][2] - (start[i][2] - drop)
    report.append({"prim": body, "start": finite(start[i]), "end": finite(end[i]), "diverged": diverged,
                   "driftLastHalfSecond": None if diverged else round(drift, 6), "atRest": not diverged and drift < 0.001,
                   "endLowestEstimate": None if diverged else round(lowest, 4), "belowGround": not diverged and lowest < -0.02})
result = {"path": gr_args["path"], "engine": "newton", "simulatedSeconds": round(simulated, 3), "updates": updates,
          "wallSeconds": round(time.time() - wall, 2),
          "dropHeight": drop, "bodies": report,
          "verdict": "diverged" if any(b["diverged"] for b in report)
          else "pass" if all(b["atRest"] and not b["belowGround"] for b in report)
          else "penetrated" if any(b["belowGround"] for b in report) else "moving"}
'''

TEMPLATES = {"stage_open": STAGE_OPEN, "validate": VALIDATE, "render": RENDER, "simulate": SIMULATE}


def script(name, arguments):
    """The execute_script code for one asset tool: `gr_args` embedded as a JSON string literal, then the template."""
    template = TEMPLATES[name]
    return f"import json as _gr_json\ngr_args = _gr_json.loads({json.dumps(json.dumps(arguments))})\n" + template
