# SO-101 Physical AI Pouring Simulation

This repository contains a state-of-the-art autonomous robot pouring simulation for the SO-101 6-DOF robotic arm in MuJoCo. The system integrates advanced Computer Vision, Large Language Model (LLM) planning, and Reinforcement Learning (RL) templates.

## 🚀 Key Features

- **3D Vision Perception**: YOLOv8-based object detection combined with MuJoCo depth-map projection to localize the bottle and glass in 3D space with millimeter precision.
- **Autonomous LLM Planning**: High-level task execution powered by Llama-3 (via Groq), featuring a "Self-Reflection" loop that learns from past episode performance.
- **Collision-Free IK**: A custom 4DOF Inverse Kinematics solver designed specifically for side-grasping, ensuring the wrist-mounted camera never strikes the bottle.
- **RL-Ready Environment**: A standard Gymnasium environment `So101PourEnv` for training low-level control policies.

## 🏗 System Architecture

1. **Perception (`vision_module.py`)**:
   - Captures RGB-D frames from the `external_camera`.
   - Runs YOLOv8 to find 2D bounding boxes.
   - Projects 2D centers to 3D world coordinates using camera intrinsics and the depth buffer.

2. **Planning (`autonomous_agent.py`)**:
   - Observations (3D coords, water levels, joint states) are serialized to JSON.
   - Llama-3 decides the next high-level action (e.g., `MOVE_TO_BOTTLE`, `TILT_POUR`).
   - After each episode, the agent "reflects" on the pour volume and updates its `knowledge.json`.

3. **Control (`pour_env.py`)**:
   - Implements a standard Gymnasium API.
   - Reward function optimizes for pour accuracy while penalizing collisions and overspills.

## 🛠 Prerequisites

- `mujoco >= 3.x`
- `ultralytics` (YOLOv8)
- `groq` (For Llama-3 planning)
- `opencv-python`
- `gymnasium`

## 🏃 Getting Started

### 1. Set up API Key
```bash
export GROQ_API_KEY="your-groq-api-key"
```

### 2. Run Autonomous Agent
```bash
mjpython autonomous_agent.py
```

### 3. Training RL (Infrastructure)
```python
from pour_env import So101PourEnv
env = So101PourEnv()
# Use with Stable Baselines 3:
# model = PPO("MlpPolicy", env, verbose=1).learn(10000)
```

## 📐 Definitions

- **Side-Grasp IK**: Locks the `wrist_roll` joint at `-1.57` rad to keep the camera upright. Targets the site `gripperframe` with a horizontal approach.
- **Reflection Logic**: A form of RLAIF where the LLM critiques its own trajectory to improve precision in the next iteration.

## 📸 Debugging
Check `vision_debug.jpg` after running the agent to see the 3D projection anchors overlaid on the YOLO detections.
