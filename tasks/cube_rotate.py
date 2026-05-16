"""Task spec defaults for cube_rotate.

Task: Grasp cube, lift it, rotate in-hand along recorded trajectory.
Object: blue cube (bench_assets cousin).
Trajectory: D3CB world-frame cube grasping trajectory.
"""

from pathlib import Path

_D3CB   = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM    = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":           str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_D3CB / "assets/objects/blue_cube/cube.usd"),
    "trajectory_json":       "",   # no trajectory — goal = hold cube at initial pose
    "object_scale":          1.0,
    "object_mass":           0.2,
    "success_tolerance":     0.05,
    # Force identity orientation: the saved_poses quaternion for this session
    # may not correspond to the cube and produces an unstable/tilted initial state.
    "object_orientation":    (1.0, 0.0, 0.0, 0.0),
    # 0.0 settle offset: the saved-pose z places the cube close to the table surface.
    # Adding extra offset causes the cube to bounce.
    "object_z_settle_offset": 0.0,
    "task_description": (
        "Object: blue cube, ~0.05 m per side (small — use PRECISION_GRASP_TARGET). "
        "No recorded trajectory: GOAL_POS = object initial position, GOAL_ROT = initial orientation. "
        "Goal: grasp and hold the cube so final keypoints are within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
