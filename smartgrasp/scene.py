"""Random tabletop layouts for synthetic data (shared by Isaac Sim and the laptop fake twin).

A layout is a list of ground-truth objects (geometry.SceneObject):
    {"category": "cube"|"cap"|"tube", "colour": str, "xy": [x, y], "yaw_deg": float}
in robot-base metres. Objects never overlap, so no cube is hidden behind another.
"""

import numpy as np
from numpy.typing import ArrayLike

from .geometry import COLOURS, Category, FloatArray, SceneObject, twin

# Prims available in the Isaac scene: cubes per colour, caps, tubes.
POOL: dict[Category, int] = {"cube": 3, "cap": len(twin.SORT_CAPS), "tube": len(twin.SORT_VIALS)}
# (x, y) ranges in robot-base metres for object centres, inside the BEV view.
AREA: tuple[tuple[float, float], tuple[float, float]] = ((0.20, 0.52), (-0.26, 0.30))
# Footprint radius per category in metres; a cube's is its half-diagonal.
RADIUS: dict[Category, float] = {"cube": twin.CUBE_EDGE / np.sqrt(2), "cap": twin.CAP_RADIUS, "tube": twin.VIAL_RADIUS}
GAP: float = 0.003  # minimum free space between objects, metres


def _axis(yaw_deg: float) -> FloatArray:
    """Unit vector along a yaw.

    Args:
        yaw_deg: Rotation about +Z in degrees.

    Returns:
        (2,) unit vector (cos, sin) in the XY plane.
    """
    yaw = np.radians(yaw_deg)
    return np.array([np.cos(yaw), np.sin(yaw)])


def _clearance(xy: ArrayLike, obj: SceneObject) -> float:
    """Distance from a point to an object's footprint (tubes are line segments).

    Args:
        xy: Point in robot-base metres.
        obj: Object already on the table.

    Returns:
        Distance in metres; negative when the point lies inside the footprint.
    """
    point, centre = np.asarray(xy, float), np.asarray(obj["xy"], float)
    if obj["category"] == "tube":
        axis = _axis(obj["yaw_deg"])
        point = point - np.clip((point - centre) @ axis, -twin.VIAL_LENGTH / 2, twin.VIAL_LENGTH / 2) * axis
    return np.linalg.norm(point - centre) - RADIUS[obj["category"]]


def _random_object(rng: np.random.Generator, category: Category, colour: str) -> SceneObject:
    """Draw one object at a random position and yaw inside AREA.

    Args:
        rng: Random generator.
        category: What to place.
        colour: Its colour.

    Returns:
        The new object; cubes get a yaw in [-45, 45) degrees, caps and tubes in [0, 180).
    """
    (x_min, x_max), (y_min, y_max) = AREA
    return {
        "category": category,
        "colour": colour,
        "xy": [float(v) for v in rng.uniform((x_min, y_min), (x_max, y_max))],
        "yaw_deg": float(rng.uniform(-45, 45) if category == "cube" else rng.uniform(0, 180)),
    }


def _probe_points(obj: SceneObject) -> list[ArrayLike]:
    """Points that must stay clear of other objects: the centre, or 15 points along a tube.

    Args:
        obj: The object being placed.

    Returns:
        Points in robot-base metres.
    """
    if obj["category"] != "tube":
        return [obj["xy"]]
    axis = _axis(obj["yaw_deg"])
    return [np.add(obj["xy"], t * axis) for t in np.linspace(-twin.VIAL_LENGTH / 2, twin.VIAL_LENGTH / 2, 15)]


def _place(
    rng: np.random.Generator, objects: list[SceneObject], category: Category, colour: str, tries: int = 200
) -> None:
    """Add one object at a random free spot, or nothing if every try overlaps.

    Args:
        rng: Random generator.
        objects: Objects already placed; the new one is appended.
        category: What to place.
        colour: Its colour.
        tries: Random spots to try before giving up.
    """
    for _ in range(tries):
        obj = _random_object(rng, category, colour)
        if all(_clearance(p, o) > RADIUS[category] + GAP for p in _probe_points(obj) for o in objects):
            objects.append(obj)
            return


def sample_layout(rng: np.random.Generator) -> list[SceneObject]:
    """Draw a random layout: 1-3 cubes of each colour plus 0-2 caps and 0-2 tubes as distractors.

    Args:
        rng: Random generator.

    Returns:
        The objects, tubes first, then caps, then cubes by colour.
    """
    objects: list[SceneObject] = []
    for _ in range(rng.integers(0, POOL["tube"] + 1)):
        _place(rng, objects, "tube", "clear")
    for _ in range(rng.integers(0, POOL["cap"] + 1)):
        _place(rng, objects, "cap", "yellow")
    for colour in COLOURS:
        for _ in range(rng.integers(1, POOL["cube"] + 1)):
            _place(rng, objects, "cube", colour)
    return objects
