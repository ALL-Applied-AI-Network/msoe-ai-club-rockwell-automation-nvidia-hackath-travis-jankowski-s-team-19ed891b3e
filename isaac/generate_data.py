"""Render randomized table-camera frames from the calibrated twin (runs under Isaac Sim's Python).

    $ISAAC/python.sh isaac/generate_data.py --out data/raw/sim_v1 --frames 2000 --seed 1

Writes camera.json, manifest.json and per frame NNNNNN.jpg + NNNNNN.json, the
ground-truth layout ({"objects": [...]}, robot-base metres). Labels are made on
the host by smartgrasp.dataset from these exact poses, so nothing here depends
on Replicator annotator formats. Physics never runs: objects are posed directly.
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--subframes", type=int, default=4, help="RTX accumulation frames per image")
args = ap.parse_args()
out = os.path.abspath(args.out)  # resolve before Kit changes the working directory

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smartgrasp.geometry import COLOURS, twin as C  # noqa: E402
from smartgrasp.scene import POOL, sample_layout  # noqa: E402

# Build the twin with a pool of objects to pose or hide per frame: 3 cubes per colour.
C.SORT_CUBES = [(0.25 + 0.03 * i, -0.25, c) for i, c in enumerate(c for c in COLOURS for _ in range(POOL["cube"]))]
C.SORT_CUBE_YAWS_DEG = [0.0] * len(C.SORT_CUBES)
import cell  # noqa: E402  (imports scene_config, so it sees the pool above)

World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
for prim in stage.Traverse():  # kinematic, as in capture_scene.py: nothing falls or drifts
    if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
UsdGeom.Imageable(stage.GetPrimAtPath(C.ROBOT_PRIM)).MakeInvisible()  # unposed arm would block the view

pools = {("cube", c): [] for c in COLOURS}
for i, (_, _, colour) in enumerate(C.SORT_CUBES):
    pools[("cube", colour)].append(stage.GetPrimAtPath(f"/World/SortCube{i}"))
pools[("cap", "yellow")] = [stage.GetPrimAtPath(f"/World/SortCap{i}") for i in range(POOL["cap"])]
pools[("tube", "clear")] = [stage.GetPrimAtPath(f"/World/SortVial{i}") for i in range(POOL["tube"])]
lights = [UsdLux.LightAPI(stage.GetPrimAtPath(f"/World/{n}Light")) for n in ("Dome", "Key", "Fill", "Rim")]
base_intensity = [light.GetIntensityAttr().Get() for light in lights]
surface = C.TABLE_TOP_Z + C.SURFACE_DZ


def apply(layout, rng):
    used = {key: 0 for key in pools}
    for obj in layout:
        key = (obj["category"], obj["colour"])
        prim = pools[key][used[key]]
        used[key] += 1
        (x, y), yaw = obj["xy"], obj["yaw_deg"]
        if obj["category"] == "tube":  # cell.py poses tubes by their end, lying along +X before the yaw
            a = np.radians(yaw)
            x, y, z = x - C.VIAL_LENGTH / 2 * np.cos(a), y - C.VIAL_LENGTH / 2 * np.sin(a), surface + C.VIAL_RADIUS + 0.001
        else:
            z = surface
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
        if obj["category"] != "cap":
            prim.GetAttribute("xformOp:rotateZ").Set(float(yaw))
        UsdGeom.Imageable(prim).MakeVisible()
    for key, prims in pools.items():
        for prim in prims[used[key]:]:
            UsdGeom.Imageable(prim).MakeInvisible()
    for light, base in zip(lights, base_intensity):  # brightness and white balance
        light.GetIntensityAttr().Set(float(base * rng.uniform(0.5, 1.6)))
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(rng.uniform(3500, 6500)))


os.makedirs(out, exist_ok=True)
gcam = UsdGeom.Camera(stage.GetPrimAtPath(cell.TABLE_CAM_PATH))
W, H = C.TABLE_CAM["resolution"]  # the real table camera's native resolution
xf = UsdGeom.Xformable(gcam.GetPrim()).ComputeLocalToWorldTransform(0)
camera = {  # same schema as fanuc_er4ia_sim/capture_scene.py, read back from the stage
    "camera": "table",
    "resolution": [W, H],
    "focal_length_mm": float(gcam.GetFocalLengthAttr().Get()),
    "horizontal_aperture_mm": float(gcam.GetHorizontalApertureAttr().Get()),
    "vertical_aperture_mm": float(gcam.GetVerticalApertureAttr().Get()),
    "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
    "table_top_z": float(surface),
    "robot_base_xy": [0.0, 0.0],
}
with open(os.path.join(out, "camera.json"), "w") as f:
    json.dump(camera, f, indent=2)
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump({"generator": "isaac/generate_data.py", "frames": args.frames, "seed": args.seed,
               "resolution": [W, H], "rt_subframes": args.subframes,
               "randomized": ["object poses and counts", "light intensity", "colour temperature"]}, f, indent=2)

rep.orchestrator.set_capture_on_play(False)  # stepping renders without playing the timeline
rgb = rep.annotators.get("rgb")
rgb.attach(rep.create.render_product(cell.TABLE_CAM_PATH, (W, H)))
rng = np.random.default_rng(args.seed)
for _ in range(30):  # let materials compile before the first saved frame
    rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

for i in range(args.frames):
    layout = sample_layout(rng)
    apply(layout, rng)
    rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
    image = np.asarray(rgb.get_data())[:, :, :3]
    cv2.imwrite(os.path.join(out, f"{i:06d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(out, f"{i:06d}.json"), "w") as f:
        json.dump({"objects": layout}, f)
    if i % 100 == 0:
        print(f"[generate] {i}/{args.frames}", flush=True)

rep.orchestrator.wait_until_complete()
app.close()
