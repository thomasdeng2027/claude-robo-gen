"""Task spec defaults for drill_rotate.

Task: Grasp drill, lift it, rotate in-hand along recorded trajectory.
Object: drill twin (SAM-reconstructed).
Trajectory: D3CB world-frame drill_1_1 trajectory.
"""

from pathlib import Path

_D3CB   = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM    = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":        str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":         str(_SAM / "meshes/drill/drill_twin/drill_twin.usd"),
    "trajectory_json":    str(_SAM / "object_trajectory/drill_1_1/trajectory_world_frame.json"),
    "object_scale":       1.0,
    "object_mass":        0.4,
    "success_tolerance":  0.05,
    "task_description": (
        "Grasp the power drill by its handle with a power grasp, lift it, then carry "
        "it through a recorded in-hand rotation trajectory (wrist-dominated motion). "
        "Object: drill_twin USD.  Goal: match GOAL_KEYPOINTS at the trajectory end pose."
    ),
}
