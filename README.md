# 🤖 SO-101 Water Pouring Robot Simulation

An autonomous pick-and-pour simulation for the **SO-101 6-DOF robotic arm** in MuJoCo. The system integrates **Computer Vision (YOLOv8)**, **LLM-based planning (Llama-3)**, and **Reinforcement Learning (Gymnasium)** to enable a robot to detect a water bottle, grasp it, transport it to a beaker, and pour a precise volume of water.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Autonomous Agent Loop                      │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 1.DETECT │──▶│2.APPROACH│──▶│ 3.GRASP  │──▶│ 4.LIFT   │ │
│  │ YOLOv8   │   │ 4DOF IK  │   │ +VERIFY  │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │                              │               │       │
│       │              ┌───────────────┘               │       │
│       │              │ retry (up to 5 trials)        ▼       │
│       │              │                         ┌──────────┐  │
│       │              └─────────────────────────│ 5.MOVE   │  │
│       │                                        │ to beaker│  │
│       │                                        └────┬─────┘  │
│       │                                             ▼        │
│       │                                        ┌──────────┐  │
│       │                                        │ 6.POUR   │  │
│       │                                        │ particles│  │
│       │                                        └────┬─────┘  │
│       │                                             ▼        │
│       │                                        ┌──────────┐  │
│       └────────────────────────────────────────│ 7.REFLECT│  │
│                                                │ LLM eval │  │
│                                                └──────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
workshop/dev/docker/workspace/src/so101_mujoco/
├── mujoco/
│   ├── scene.xml                # MuJoCo world: table, bottle, beaker, particles
│   ├── so101_new_calib.xml      # Robot URDF with calibrated joint limits
│   ├── autonomous_agent.py      # 🧠 Main agent: vision → plan → grasp → pour → reflect
│   ├── pour_pipeline.py         # 🔧 Proven deterministic pipeline (no LLM needed)
│   ├── vision_module.py         # 👁 YOLOv8 + MuJoCo depth → 3D object coordinates
│   ├── pour_env.py              # 🎮 Gymnasium RL environment for training
│   ├── knowledge.json           # 📝 Persistent memory: lessons learned across episodes
│   ├── llama_pour_agent.py      # LLM-only agent (legacy)
│   ├── llama_gazebo_agent.py    # Gazebo bridge agent (for ROS2 stack)
│   ├── yolo_pour_agent.py       # YOLO detection integration
│   ├── side_grasp_demo.py       # Standalone side-grasp test
│   └── README.md                # Component-level documentation
├── meshes/                      # Robot STL mesh files
└── scripts/                     # Teleop & viewer utilities
```

## 🔑 Key Components

### 1. 3D Vision Perception (`vision_module.py`)
- **YOLOv8** detects `bottle` and `cup` bounding boxes in RGB frames
- **MuJoCo depth buffer** converts 2D pixel centers → 3D world coordinates
- Saves `vision_debug.jpg` for visual verification

### 2. 4-DOF Inverse Kinematics (`autonomous_agent.py`)
- Targets the `gripperframe` site (located at **fingertips**)
- Locks `wrist_roll = -π/2` to keep the D435i camera upright
- **500 Jacobian-transpose iterations** with gains `4.0` (position) / `1.5` (orientation)
- Clamps solutions to physical joint limits
- Critical insight: IK target is **6cm past** the bottle center in the finger direction

### 3. Adaptive Grasp with Verification
- **5 trial strategies** with different approach angles:
  | Trial | Y Offset | Z Shift | Direction |
  |-------|----------|---------|-----------|
  | 1     | +0.12    | +0.01   | Side +Y   |
  | 2     | +0.12    | +0.03   | High +Y   |
  | 3     | -0.12    | +0.01   | Side -Y   |
  | 4     | -0.12    | +0.03   | High -Y   |
  | 5     | +0.09    | -0.01   | Low +Y    |
- **Verification**: Gripper joint `qpos > 0.15` = bottle secured; `< 0.15` = empty closure

### 4. Water Particle System (`scene.xml`)
- 60 sphere particles hidden underground at start
- Teleported to bottle neck during pour, with randomized velocity
- Visual fill levels update in real-time on both bottle and beaker

### 5. LLM Self-Reflection (`knowledge.json`)
- After each episode, **Llama-3.3-70b** via Groq API evaluates performance
- Generates actionable rules: *"Tilt more aggressively"* or *"Stop pour earlier"*
- Rules persist across episodes for continuous improvement

### 6. RL Environment (`pour_env.py`)
- Standard **Gymnasium** interface (`reset`, `step`, `render`)
- **Observation**: joint positions/velocities, gripper width, bottle/beaker 3D coords, beaker fill level
- **Reward**: distance-to-bottle penalty + pour accuracy bonus + overspill penalty
- Ready for **Stable Baselines3** (PPO, SAC, etc.)

## 🛠 Prerequisites

```bash
pip install mujoco ultralytics groq opencv-python gymnasium numpy
```

- **MuJoCo ≥ 3.3** (tested with 3.3.7)
- **GROQ_API_KEY** environment variable (for LLM reflection)

## 🚀 Quick Start

### Run the Proven Pipeline (no LLM needed)
```bash
cd workshop/dev/docker/workspace/src/so101_mujoco/mujoco
/Users/rohitkunjam/Library/Python/3.9/bin/mjpython pour_pipeline.py
```

### Run the Full Autonomous Agent (with LLM reflection)
```bash
cd workshop/dev/docker/workspace/src/so101_mujoco/mujoco
export GROQ_API_KEY="your-key-here"
/Users/rohitkunjam/Library/Python/3.9/bin/mjpython autonomous_agent.py
```

### Train with RL
```python
from pour_env import So101PourEnv
env = So101PourEnv()
obs, _ = env.reset()
# Use with Stable Baselines3:
# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", env, verbose=1).learn(10000)
```

## 📐 Technical Details

### IK Mathematics
The 4-DOF solver uses Jacobian transpose with locked wrist roll:

```
δq = Jₚᵀ · e_pos · α_pos + Jᵣᵀ · e_rot · α_rot
```

Where:
- `Jₚ`: 3×4 position Jacobian (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex)
- `e_pos`: position error (target - current gripperframe)
- `e_rot`: `cross(current_finger_axis, desired_finger_direction)`
- `α_pos = 4.0`, `α_rot = 1.5` (empirically tuned)

### Grasp Geometry
```
gripperframe (fingertips)
        ↓
    ┌───┤───┐
    │       │  ← moving jaw
    │  [○]  │  ← bottle (radius ≈ 3cm)
    │       │  ← fixed jaw
    └───────┘

IK target = bottle_center + finger_dir × 0.06m
(places fingertips 6cm past bottle center)
```

### Water Flow Model
```
flow_rate = 25 ml/s × min(1, (tilt° - 30°) / 60°)
```
Pour starts when bottle tilt exceeds 30° and reaches full rate at 90°.

## 🎥 Demo

Run either pipeline and the MuJoCo viewer will open showing:
1. Robot arm approaching the bottle
2. Side-grasp with camera clearance
3. Lift and transport
4. Particle-based water pouring
5. LLM reflection output in terminal

## 📝 License

This project is part of the Physical AI Challenge 2026.
