# so101_leader_description

This package contains a MoveIt-ready SO-101 URDF for the leader variant, derived from the working follower setup.

## Contents

- `urdf/so101.urdf`: leader variant URDF (same joint structure as follower setup)
- `meshes/*.stl`: simulation meshes plus leader meshes (`wrist_roll_so101.stl`, `trigger_so101.stl`, `handle_so101.stl`)

## Build

From your ROS 2 workspace root:

```bash
colcon build --packages-select so101_leader_description
source install/setup.bash
```

## Use With MoveIt

The corresponding MoveIt package is `so101_leader_moveit_config`, already wired to this description package.

## Notes

- This leader package was created by cloning your working follower packages and replacing leader-specific wrist/gripper geometry.
- Kinematic chain and joint names were intentionally kept the same so the same planning structure remains valid.
