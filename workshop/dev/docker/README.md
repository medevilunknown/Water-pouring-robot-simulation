# SO-101 Physical AI Hackathon - Docker Setup & Troubleshooting Guide

This guide covers how to set up the Docker environment for the SO-101 robot simulation, build the workspace, run the pick-and-place services, and fix common issues encountered during the initial setup.

## 1. Prerequisites & Starting the Container

Before launching the container, ensure you allow X11 forwarding so that GUI applications (like MuJoCo, Gazebo, and RViz) can open on your host screen:

```bash
xhost +local:docker
```

Start and enter the container:
```bash
docker compose up -d
docker exec -it lerobot_hackathon_env bash
```

## 2. Cloning the ROS 2 Repositories

**The Issue:** The `Dockerfile` clones the `so101` repositories during the image build, but you might notice they are missing when you enter the container. This happens because `docker-compose.yml` mounts your host's `./workspace` over the container's `/home/hacker/workspace`, which hides the files cloned during the build.

**The Fix:** You need to clone the repositories manually into your host's mapped `workspace/src` directory (or run this inside the container):

```bash
cd /home/hacker/workspace
mkdir -p src && cd src
git clone https://github.com/sahillathwal/so101_moveit_config.git
git clone https://github.com/sahillathwal/so101_leader_moveit_config.git
git clone https://github.com/sahillathwal/so101_leader_description.git
git clone https://github.com/sahillathwal/so101_description.git
git clone https://github.com/sahillathwal/so101_gazebo.git
git clone https://github.com/sahillathwal/so101_mujoco.git
git clone https://github.com/sahillathwal/so101_unified_bringup.git
```

## 3. Building the ROS 2 Workspace

**The Issue:** Running `colcon build` fails with `ModuleNotFoundError: No module named 'catkin_pkg'` or `CMake Error`.
**The Reason:** The Docker container automatically activates the `lerobot_venv` virtual environment (Python 3.12) to support AI dependencies like PyTorch. However, ROS 2 Humble relies on the system Python (Python 3.10) which contains the required `catkin_pkg` and ROS libraries. 

**The Fix:** You must temporarily deactivate the virtual environment before building the workspace:

```bash
# Unset the VIRTUAL_ENV and correct the PATH
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
export PATH=$(echo $PATH | tr ':' '\n' | grep -v lerobot_venv | tr '\n' ':' | sed 's/:\$//')

# Source ROS 2 and build
source /opt/ros/humble/setup.bash
cd /home/hacker/workspace
colcon build --symlink-install

# Source your newly built workspace
source install/setup.bash
```

*(Note: If you need to run AI scripts later, you can reactivate the environment using `source /opt/lerobot_venv/bin/activate`)*

## 4. Cartesian Planning Fails (Fixing the IK Solver Timeout)

**The Issue:** Joint-space moves work fine, but requesting a Cartesian target (e.g. `ros2 service call /pick_front ...`) fails immediately with a message like `Failed to move to pick pose` or no valid plans found.
**The Reason:** The SO-101 is a 5 Degree-of-Freedom (DOF) arm. MoveIt uses the KDL kinematics solver by default, which struggles with 5-DOF arms. The default configuration gives the solver an extremely low timeout (`0.05` seconds / 50ms), causing it to fail before it can find a valid mathematical solution for the Cartesian pose.

**The Fix:** Increase the IK solver timeout in the main arm config.
Edit the file `workspace/src/so101_moveit_config/config/kinematics.yaml`:

```yaml
arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.5   # <--- Change this from 0.05 to 0.5
  position_only_ik: true
```

*Make sure you edit the `so101_moveit_config` package, not the `so101_leader_moveit_config` wrapper.*
After saving, kill your simulation, rebuild the workspace (`colcon build --packages-select so101_moveit_config`), and restart.

## 5. Running the Simulation & Testing Pick/Place

Launch the complete Gazebo + MoveIt + RViz stack:
```bash
ros2 launch so101_unified_bringup main.launch.py
```

In a **second terminal window** (inside the container, with the workspace sourced), you can send service calls to trigger the automated pick and place workflows:

**Move to a safe joint home:**
```bash
ros2 service call /move_to_joint_states so101_unified_bringup/srv/JointReq "{joints: {name: ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'], position: [0.0, -0.5, 0.5, 0.0, 0.0]}}"
```

**Pick an object from the table:**
*(Example object defined in `empty_world.sdf` at approx `X=-0.30, Y=0.0, Z=0.80`)*
```bash
ros2 service call /pick_front so101_unified_bringup/srv/PickFront "{target_pose: {position: {x: -0.30, y: 0.0, z: 0.80}, orientation: {w: 1.0}}, grip_state: true}"
```

**Place the object:**
```bash
ros2 service call /place_object so101_unified_bringup/srv/PlaceObject "{target_pose: {position: {x: -0.40, y: -0.15, z: 0.82}, orientation: {w: 1.0}}, grip_state: false}"
```
