"""Task spec defaults for pour_water.

Task: Grasp cup/bottle, lift it, tilt ~60° to simulate pouring.
Object: TODO — set object_usd to your pour-water cup USD before running.
Trajectory: iPhone-recorded pourwater tilting motion (position/rpy format).
"""

from pathlib import Path

_D3CB   = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM    = Path("/juno/u/jiaqis7/sam-3d-objects")

# TODO: replace with the actual pourwater cup USD once available
_POURWATER_USD = ""   # e.g. "/juno/u/jiaqis7/sam-3d-objects/pourwater/pourwater.usd"

DEFAULTS = {
    "robot_usd":          "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "object_usd":         _POURWATER_USD,
    "trajectory_json":    str(_SAM / "object_trajectory/pourwater/trajectory.json"),
    "object_scale":       1.0,
    "object_mass":        0.3,
    "success_tolerance":  0.05,
    "task_description": (
        "Grasp the cup or bottle (cylindrical) with a power grasp, lift it, then tilt "
        "it ~60° following the recorded pouring trajectory (wrist-tilt motion). "
        "Goal: match GOAL_KEYPOINTS at the tilted end pose."
    ),
}
