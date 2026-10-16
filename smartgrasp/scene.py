"""Random tabletop layouts for synthetic data (shared by Isaac Sim and the laptop fake twin).

A layout is a list of ground-truth objects:
    {"category": "cube"|"cap"|"tube", "colour": str, "xy": [x, y], "yaw_deg": float}
in robot-base metres. Objects never overlap, so no cube is hidden behind another.
"""

import numpy as np

from .geometry import COLOURS, twin

# Prims available in the Isaac scene: cubes per colour, caps, tubes.
POOL = {"cube": 3, "cap": len(twin.SORT_CAPS), "tube": len(twin.SORT_VIALS)}
AREA = ((0.20, 0.52), (-0.26, 0.30))  # x, y ranges for object centres (inside the BEV view)
RADIUS = {"cube": twin.CUBE_EDGE / np.sqrt(2), "cap": twin.CAP_RADIUS, "tube": twin.VIAL_RADIUS}
GAP = 0.003  # minimum free space between objects, metres


def _clearance(xy, obj):
    """Distance from a point to an object's footprint (tubes are line segments)."""
    p, c = np.asarray(xy, float), np.asarray(obj["xy"], float)
    if obj["category"] == "tube":
        a = np.radians(obj["yaw_deg"])
        d = np.array([np.cos(a), np.sin(a)])
        p = p - np.clip((p - c) @ d, -twin.VIAL_LENGTH / 2, twin.VIAL_LENGTH / 2) * d
    return np.linalg.norm(p - c) - RADIUS[obj["category"]]


def _place(rng, objects, category, colour, tries=200):
    for _ in range(tries):
        obj = {
            "category": category,
            "colour": colour,
            "xy": [float(v) for v in rng.uniform(*zip(*AREA))],
            "yaw_deg": float(rng.uniform(-45, 45) if category == "cube" else rng.uniform(0, 180)),
        }
        points = [obj["xy"]]
        if category == "tube":  # check along the whole tube, not just its centre
            a = np.radians(obj["yaw_deg"])
            points = [np.add(obj["xy"], t * np.array([np.cos(a), np.sin(a)]))
                      for t in np.linspace(-twin.VIAL_LENGTH / 2, twin.VIAL_LENGTH / 2, 15)]
        if all(_clearance(p, o) > RADIUS[category] + GAP for p in points for o in objects):
            objects.append(obj)
            return


def sample_layout(rng):
    """1-3 cubes of each colour plus 0-2 caps and 0-2 tubes as distractors."""
    objects = []
    for _ in range(rng.integers(0, POOL["tube"] + 1)):
        _place(rng, objects, "tube", "clear")
    for _ in range(rng.integers(0, POOL["cap"] + 1)):
        _place(rng, objects, "cap", "yellow")
    for colour in COLOURS:
        for _ in range(rng.integers(1, POOL["cube"] + 1)):
            _place(rng, objects, "cube", colour)
    return objects
