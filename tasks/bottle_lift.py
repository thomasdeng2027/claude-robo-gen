"""Task spec defaults for bottle_lift.

Task: Grasp bottle, lift >= 12 cm from initial Z, transport along recorded trajectory.
Object: bottle twin (SAM-reconstructed).
Trajectory: iPhone-recorded bottle_2 lifting motion (position/rpy format).
"""

from pathlib import Path

_D3CB   = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM    = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":           str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_SAM / "meshes/bottle/bottle_twin/bottle_twin.usd"),
    "trajectory_json":       str(_SAM / "object_trajectory/bottle_2/trajectory_lift_synthetic.json"),
    "object_scale":          1.0,
    "object_mass":           0.3,
    "success_tolerance":     0.1,
    # 180° rotation around Z so the USD mesh (baked cap-up) stands upright.
    "object_orientation":    (0.0, 0.0, 1.0, 0.0),
    # Small drop so the bottle falls and settles onto the table surface.
    "object_z_settle_offset": 0.04,
    "task_description": (
        "Object: upright plastic bottle, cylindrical (~0.07 m diameter, ~0.22 m tall). "
        "Goal: grasp the bottle and transport it along the recorded lifting trajectory, "
        "ending with all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
