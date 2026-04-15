# SO-101 Hackathon Simulation Guide

This guide covers the steps to set up the SO-101 robot simulation environment and start controlling it using your keyboard.

## 1. Environment Preparation

Ensure you are inside the provided Docker container. If you aren't already, start it and access the bash shell:

```bash
# Start the container (from the project directory on your host)
docker compose up -d

# Enter the container
docker exec -it lerobot_hackathon_env bash
```

## 2. Workspace Setup

Inside the container, you need to build the ROS 2 workspace. **Important: Ensure your AI virtual environment is NOT active while building ROS packages.**

```bash
# Deactivate venv if you see (lerobot_venv) in your prompt
deactivate 2>/dev/null

# Source ROS 2 system environment
source /opt/ros/humble/setup.bash

# Build the workspace

# Source your local workspace
source install/setup.bash
```

## 3. Launching the Simulation

Open a terminal and run the MuJoCo bridge. This command will open the 3D viewer window and start the simulation.

```bash
python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml
```

> [!NOTE]
> If the 3D viewer doesn't open, ensure your X11 forwarding is configured correctly or that you have a display head attached.

## 4. Keyboard Teleoperation

Open a **separate terminal**, source the workspace, and start the teleoperation script:

```bash
# Source workspace again in the second terminal
source /home/hacker/workspace/install/setup.bash

# Start teleop
python3 src/so101_mujoco/scripts/so101_keyboard_teleop.py
```

### Controls:
- **Arrow Keys**: Move X/Y axis
- **W / S**: Move Z axis (Up/Down)
- **Q / E**: Wrist Roll
- **R / F**: Wrist Pitch
- **SPACE**: Toggle Gripper
- **ESC**: Quit

