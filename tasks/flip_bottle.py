"""Task spec defaults for flip_bottle.

Task: Grasp bottle, lift it, flip 180° so it stands on its cap.
Object: bottle twin (SAM-reconstructed).
Trajectory: D3CB bottle world-frame trajectory (skip-upright variant).
"""

from pathlib import Path

_D3CB   = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM    = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":        str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":         str(_SAM / "meshes/bottle/bottle_twin/bottle_twin.usd"),
    "trajectory_json":    "",   # no trajectory — Claude generates primitive flip skill
    "object_scale":       1.0,
    "object_mass":        0.3,
    "success_tolerance":  0.05,
    "task_description": (
        "Object: plastic bottle, cylindrical (~0.07 m diameter, ~0.22 m tall), upright. "
        "Goal: grasp the bottle, flip it ~180° (cap-up → cap-down), and end with "
        "all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS at the inverted pose."
    ),
}
