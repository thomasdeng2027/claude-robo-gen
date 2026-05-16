"""Task spec defaults for drill_lift.

Task: Grasp drill, lift >= 10 cm from initial Z, transport along recorded trajectory.
Object: drill twin (SAM-reconstructed).
Trajectory: iPhone-recorded drill_1 lifting motion (orientation_rpy format).
"""

from pathlib import Path

_D3CB = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM  = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":           str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_SAM / "meshes/drill/drill_twin/drill_twin.usd"),
    "trajectory_json":       str(_SAM / "object_trajectory/drill/trajectory.json"),
    "object_scale":          1.0,
    "object_mass":           0.4,
    "success_tolerance":     0.05,
    # Identity: the drill mesh is baked lying flat — natural resting pose on a table.
    "object_orientation":    (1.0, 0.0, 0.0, 0.0),
    # Small drop so the drill settles onto the table surface.
    "object_z_settle_offset": 0.02,
    "task_description": (
        "Object: power drill, elongated body (~0.05 m handle diameter, ~0.25 m long), "
        "resting flat on the table.  "
        "Goal: grasp the drill and transport it along the recorded trajectory, "
        "ending with all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
