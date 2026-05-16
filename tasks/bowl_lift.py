"""Task spec defaults for bowl_lift.

Task: Grasp bowl, lift >= 10 cm from initial Z, transport along recorded trajectory.
Object: OakInk2 bowl (C12001, bench_assets).
Trajectory: iPhone-recorded bowl lifting motion (orientation_rpy format).
"""

from pathlib import Path

_D3CB = Path("/juno/u/jingyuny/projects/p_vla/claude-data-gen")
_SAM  = Path("/juno/u/jiaqis7/sam-3d-objects")

DEFAULTS = {
    "robot_usd":            "/juno/u/jiaqis7/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd",
    "saved_poses":        str(_D3CB / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":         str(_SAM / "meshes/bowl/twin/twin.usd"),
    "trajectory_json":    str(_SAM / "object_trajectory/pourwater/trajectory.json"),
    "object_scale":       1.0,
    "object_mass":        0.2,
    "success_tolerance":  0.05,
    # Identity: check orientation in smoke test and adjust if bowl appears upside down.
    "object_orientation": (1.0, 0.0, 0.0, 0.0),
    "task_description": (
        "Object: ceramic bowl, wide and shallow (~0.16 m diameter, ~0.06 m tall). "
        "Goal: grasp the bowl and transport it along the recorded lifting trajectory, "
        "ending with all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
