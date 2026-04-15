# so101_description

This package contains a MoveIt-ready SO-101 URDF for a single (individual) robot.

## Contents

- `urdf/so101.urdf`: based on `Simulation/SO101/so101_new_calib.urdf`
- `meshes/*.stl`: copied from `Simulation/SO101/assets`

## Fixes Applied for MoveIt Setup Assistant

- Converted mesh paths from relative (`assets/...`) to package URIs:
  - `package://so101_description/meshes/...`
- Removed invalid URDF element inside `gripper_frame_link`:
  - a direct `<origin .../>` child of `<link>`

## Build

From your ROS 2 workspace root:

```bash
colcon build --packages-select so101_description
source install/setup.bash
```

## Run MoveIt Setup Assistant

```bash
ros2 run moveit_setup_assistant moveit_setup_assistant
```

Then:

1. Create New MoveIt Configuration Package.
2. Load URDF from package:
   - `so101_description/urdf/so101.urdf`
3. Suggested planning groups:
   - `arm`: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`
   - `gripper`: `gripper`
4. End effector tip link suggestion:
   - `gripper_frame_link`

## Notes

- This URDF is follower-style geometry from the existing simulation model.
- If you need a leader-specific model, create a leader variant URDF and regenerate the MoveIt config.
