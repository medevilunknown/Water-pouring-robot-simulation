# so101_unified_bringup

This package provides a self-contained MoveIt and Gazebo bringup for so101.

It contains:

- A combined Gazebo + MoveIt launch flow
- A C++ `moveit_server` node ported from `ras_moveit`
- The service definitions required by that server
- A local OMPL planning config used by the server planner IDs
- A local world file and RViz config

## Launch

```bash
ros2 launch so101_unified_bringup main.launch.py
```

## Server Only

```bash
ros2 launch so101_unified_bringup moveit_server.launch.py
```

## Useful arguments

- `rviz:=true|false`
- `world:=/absolute/path/to/world.sdf`
- `world_name:=empty`
- `entity_name:=so101`
- `x:=-0.55 y:=0.0 z:=0.7774 yaw:=0.0`
- `bridge_config:=/absolute/path/to/ros_gz_bridge.yaml`

## Services

The package exposes the same service names used by the original RAS MoveIt server, including:

- `/create_traj`
- `/move_to_joint_states`
- `/rotate_effector`
- `/sync_arm`
- `/place_object`
- `/pick_object`
- `/pick_front`
- `/pick_right`
- `/pick_left`
- `/pick_rear`
