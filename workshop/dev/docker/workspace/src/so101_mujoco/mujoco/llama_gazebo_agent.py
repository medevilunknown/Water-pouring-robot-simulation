"""
Llama 3 Controlled Robot Pouring Agent
=======================================
Uses Groq API (Llama 3.3-70b) to control the SO-101 robot arm in MuJoCo.
The agent observes simulation state and decides actions to:
  1. Pick up the water bottle
  2. Pour water into the beaker
  3. Stop before overspilling

Usage:
  export GROQ_API_KEY="your-key-here"
  mjpython llama_pour_agent.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import json
import math
import socket
from groq import Groq

# ─── UDP Gazebo Bridge ──────────────────────────────────────────
class GazeboUDPBridge:
    """Transmits joint positions to the ROS 2 Gazebo simulation container."""
    def __init__(self, ip="127.0.0.1", port=9876):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send_joint_state(self, q_arm, gripper_val):
        msg = {
            "shoulder_pan": float(q_arm[0]),
            "shoulder_lift": float(q_arm[1]),
            "elbow_flex": float(q_arm[2]),
            "wrist_flex": float(q_arm[3]),
            "wrist_roll": float(q_arm[4]),
            "gripper": float(gripper_val)
        }
        try:
            self.sock.sendto(json.dumps(msg).encode(), (self.ip, self.port))
        except Exception:
            pass

# ─── Configuration ───────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
BEAKER_CAPACITY_ML = 150.0
TARGET_FILL_ML = 100.0          # Pour to the 100ml mark
BOTTLE_WATER_ML = 120.0         # Bottle starts with 120ml
POUR_RATE_ML_PER_SEC = 25.0     # Flow rate when fully tilted (90°)
MAX_STEPS = 30                  # Max LLM decision steps before forced stop

# ─── MuJoCo Setup ───────────────────────────────────────────────
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joint_names]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
dof_ids = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]

grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
bottle_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
beaker_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")

# ─── IK Solver ───────────────────────────────────────────────────
def solve_ik(target_pos, target_z_axis=None):
    """Jacobian-transpose IK for the 5-DOF arm."""
    q = d.qpos.copy()
    arm_dofs = dof_ids[:5]
    arm_qpos = qpos_ids[:5]

    for _ in range(500):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        pos = d.site_xpos[grasp_site]
        err_pos = target_pos - pos

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, grasp_site)

        Jp = jacp[:, arm_dofs]
        delta_q = Jp.T @ err_pos * 3.0

        if target_z_axis is not None:
            mat = d.site_xmat[grasp_site].reshape(3, 3)
            z_axis = mat[:, 2]
            err_rot = np.cross(z_axis, target_z_axis)
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 1.0

        q[arm_qpos] += delta_q
        d.qpos[:] = q

    return q[arm_qpos]

# ─── Water Flow Simulator ───────────────────────────────────────
class WaterSimulator:
    """Simulates water transfer based on bottle tilt angle."""

    def __init__(self, bottle_ml=BOTTLE_WATER_ML, beaker_capacity=BEAKER_CAPACITY_ML):
        self.bottle_ml = bottle_ml
        self.beaker_ml = 0.0
        self.beaker_capacity = beaker_capacity
        self.is_pouring = False
        self.pour_start_time = None
        self.total_poured = 0.0

    def update(self, tilt_angle_deg, dt):
        """Update water levels based on tilt angle. Returns current beaker level."""
        if tilt_angle_deg > 30 and self.bottle_ml > 0:
            # Flow rate scales with tilt angle (30° to 90° → 0% to 100% flow)
            flow_factor = min(1.0, (tilt_angle_deg - 30) / 60.0)
            flow = POUR_RATE_ML_PER_SEC * flow_factor * dt
            actual_flow = min(flow, self.bottle_ml)
            self.bottle_ml -= actual_flow
            self.beaker_ml += actual_flow
            self.total_poured += actual_flow
            self.is_pouring = True
        else:
            self.is_pouring = False

        return self.beaker_ml

    @property
    def beaker_pct(self):
        return (self.beaker_ml / self.beaker_capacity) * 100

    @property
    def is_overspilling(self):
        return self.beaker_ml >= self.beaker_capacity

    def status_str(self):
        return (f"Bottle: {self.bottle_ml:.1f}ml | "
                f"Beaker: {self.beaker_ml:.1f}/{self.beaker_capacity:.0f}ml "
                f"({self.beaker_pct:.0f}%) | "
                f"Pouring: {'YES' if self.is_pouring else 'no'}")

# ─── State Observer ──────────────────────────────────────────────
def get_bottle_tilt_deg():
    """Get bottle tilt angle from vertical in degrees."""
    bottle_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
    # Freejoint qpos: [x,y,z, qw,qx,qy,qz]
    bottle_qposadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")]
    quat = d.qpos[bottle_qposadr+3:bottle_qposadr+7]
    # Convert quaternion to rotation matrix
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    # z-axis of bottle body
    z_axis = rot[:, 2]
    # Angle from vertical [0,0,1]
    cos_angle = np.clip(np.dot(z_axis, [0, 0, 1]), -1, 1)
    return math.degrees(math.acos(cos_angle))


def get_object_bbox(renderer, obj_name):
    """Simulated Object Detection: Returns [x_min, y_min, x_max, y_max] from semantic segmentation."""
    geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
    if geom_id == -1: return None
    
    # Enable segmentation exactly once per agent init
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera="external_camera")
    seg = renderer.render()
    
    # seg shape is (H, W, 2). Channel 0 is obj type (mjOBJ_GEOM = 5), Channel 1 is obj ID
    mask = (seg[:, :, 0] == 5) & (seg[:, :, 1] == geom_id)
    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0: return None # Not visible
    
    return [int(np.min(x_idx)), int(np.min(y_idx)), int(np.max(x_idx)), int(np.max(y_idx))]

def pixel_to_world(u, v, known_z, cam_name="external_camera"):
    """Projects a 2D pixel coordinate (u,v) from camera view to 3D given the object's real Z."""
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = d.cam_xpos[cam_id]
    cam_mat = d.cam_xmat[cam_id].reshape(3, 3)
    fovy = m.cam_fovy[cam_id]
    
    width = 640
    height = 480
    focal_y = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    focal_x = focal_y # Assuming square pixels
    
    cx = width / 2.0
    cy = height / 2.0
    
    dx = (u - cx) / focal_x
    dy = -(v - cy) / focal_y
    dz = -1.0
    
    ray_cam = np.array([dx, dy, dz])
    ray_cam /= np.linalg.norm(ray_cam)
    
    ray_world = cam_mat @ ray_cam
    
    if abs(ray_world[2]) < 1e-6:
        return None
        
    t = (known_z - cam_pos[2]) / ray_world[2]
    if t > 0:
        return cam_pos + t * ray_world
    return None

def get_cv_grasping_target(renderer, obj_name, known_z):
    """Estimates the 3D position of an object given its 2D mask bounds."""
    bbox = get_object_bbox(renderer, obj_name)
    if not bbox:
        return None
    u = (bbox[0] + bbox[2]) / 2.0
    v = (bbox[1] + bbox[3]) / 2.0
    return pixel_to_world(u, v, known_z)


def build_observation(water_sim, phase, gripper_open, renderer=None):
    """Build a text observation for the LLM."""
    mujoco.mj_kinematics(m, d)

    grip_pos = d.site_xpos[grasp_site]
    bottle_pos = d.site_xpos[bottle_site]
    beaker_pos = d.site_xpos[beaker_site]
    joint_pos = [d.qpos[qpos_ids[i]] for i in range(6)]
    tilt = get_bottle_tilt_deg()
    
    # SIMULATED OBJECT DETECTION 
    detections = {}
    if renderer:
        bottle_box = get_object_bbox(renderer, "bottle_visual_shell")
        beaker_box = get_object_bbox(renderer, "target_visual_shell")
        if bottle_box: detections["bottle"] = f"bbox {bottle_box}"
        if beaker_box: detections["beaker"] = f"bbox {beaker_box}"

    obs = {
        "phase": phase,
        "gripper": "open" if gripper_open else "closed",
        "vision_object_detection": detections,
        "gripper_pos": [round(x, 4) for x in grip_pos],
        "bottle_pos": [round(x, 4) for x in bottle_pos],
        "beaker_pos": [round(x, 4) for x in beaker_pos],
        "bottle_tilt_deg": round(tilt, 1),
        "water": {
            "bottle_ml": round(water_sim.bottle_ml, 1),
            "beaker_ml": round(water_sim.beaker_ml, 1),
            "beaker_capacity_ml": water_sim.beaker_capacity,
            "beaker_fill_pct": round(water_sim.beaker_pct, 1),
            "target_fill_ml": TARGET_FILL_ML,
            "is_overspilling": water_sim.is_overspilling,
        },
        "joint_positions_rad": [round(j, 3) for j in joint_pos],
    }
    return json.dumps(obs, indent=2)

# ─── LLM Agent ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot arm controller. You control a SO-101 6-DOF robotic arm in a MuJoCo simulation.

Your task: Pick up a water bottle, move it above a beaker, pour water to approximately 100ml (the target), then stop and return the bottle. Do NOT overspill — the beaker capacity is 150ml.

Each step, you receive a JSON observation of the current state. You must respond with EXACTLY ONE action from the list below. Respond with ONLY the action name (and parameter if needed), nothing else.

Available actions:
  MOVE_ABOVE_BOTTLE    - Move gripper above the bottle (pre-grasp)
  LOWER_TO_BOTTLE      - Lower gripper to grasp position on bottle
  CLOSE_GRIPPER        - Close gripper to grab bottle
  LIFT_BOTTLE          - Lift the bottle up
  MOVE_TO_BEAKER       - Move bottle above the beaker
  TILT_POUR <angle>    - Tilt bottle to pour. Angle 0-90 degrees. Start with 45, increase if flow is slow. 
  STOP_POUR            - Return bottle to upright position
  RETURN_HOME          - Put bottle down and return arm to home
  DONE                 - Task complete

Strategy tips:
- First pick up the bottle: MOVE_ABOVE_BOTTLE → LOWER_TO_BOTTLE → CLOSE_GRIPPER → LIFT_BOTTLE
- Move to beaker: MOVE_TO_BEAKER
- Pour carefully: TILT_POUR 45 to start, watch the beaker_ml level
- When beaker_ml reaches ~90-100ml, STOP_POUR immediately (water keeps flowing briefly)
- Stop early to avoid overspilling! Aim for 90-100ml, never exceed 140ml
- After stopping: RETURN_HOME → DONE

CRITICAL: Respond with ONLY the action. No explanation, no markdown, no extra text."""


def query_llama(client, observation, conversation_history):
    """Query Llama 3 via Groq API."""
    conversation_history.append({
        "role": "user",
        "content": f"Current state:\n{observation}\n\nWhat action should I take?"
    })

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:],
            temperature=0.1,
            max_tokens=50,
        )
        action = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": action})
        return action
    except Exception as e:
        print(f"  ⚠ LLM Error: {e}")
        return None


def parse_action(action_str):
    """Parse LLM output into (action_name, param)."""
    if not action_str:
        return None, None
    parts = action_str.strip().split()
    name = parts[0].upper()
    # Handle multi-word action names
    if len(parts) >= 2 and parts[0].upper() + "_" + parts[1].upper() in [
        "MOVE_ABOVE", "MOVE_TO", "LOWER_TO", "CLOSE_GRIPPER",
        "LIFT_BOTTLE", "STOP_POUR", "RETURN_HOME", "TILT_POUR"
    ]:
        name = parts[0].upper() + "_" + parts[1].upper()
        param = parts[2] if len(parts) > 2 else None
    elif len(parts) >= 3 and parts[0].upper() + "_" + parts[1].upper() + "_" + parts[2].upper() in [
        "MOVE_ABOVE_BOTTLE", "LOWER_TO_BOTTLE", "MOVE_TO_BEAKER"
    ]:
        name = parts[0].upper() + "_" + parts[1].upper() + "_" + parts[2].upper()
        param = parts[3] if len(parts) > 3 else None
    else:
        param = parts[1] if len(parts) > 1 else None

    return name, param


# ─── Motion Controller ──────────────────────────────────────────
class MotionController:
    """Executes high-level actions as IK-based smooth motions."""

    def __init__(self, viewer, water_sim, gazebo_bridge, renderer=None):
        self.viewer = viewer
        self.water_sim = water_sim
        self.gazebo_bridge = gazebo_bridge
        self.renderer = renderer
        self.gripper_val = 1.0  # 1.0 = open, -0.1 = closed
        self.q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])

    def move_to(self, target_q, gripper_val, duration=2.0):
        """Smooth interpolated motion to target joint config."""
        start_q = d.qpos[qpos_ids[:5]].copy()
        start_time = d.time
        while d.time - start_time < duration:
            alpha = (d.time - start_time) / duration
            alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)

            d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
            d.ctrl[act_ids[5]] = gripper_val

            # Send state to Gazebo
            self.gazebo_bridge.send_joint_state(d.ctrl[act_ids[:5]], gripper_val)

            # Update water simulation
            tilt = get_bottle_tilt_deg()
            self.water_sim.update(tilt, m.opt.timestep)

            mujoco.mj_step(m, d)
            self.viewer.sync()
            time.sleep(m.opt.timestep)

        self.gripper_val = gripper_val

    def wait(self, duration=1.0):
        """Hold current position for duration."""
        start = d.time
        while d.time - start < duration:
            tilt = get_bottle_tilt_deg()
            # Send state to Gazebo
            self.gazebo_bridge.send_joint_state(d.ctrl[act_ids[:5]], self.gripper_val)
            self.water_sim.update(tilt, m.opt.timestep)
            mujoco.mj_step(m, d)
            self.viewer.sync()
            time.sleep(m.opt.timestep)

    def execute(self, action_name, param=None):
        """Execute a high-level action. Returns success bool."""
        mujoco.mj_kinematics(m, d)
        
        # Setup precise Z heights for the bounds of these objects based on their properties
        bottle_z_center = 0.068
        beaker_z_center = 0.037
        cv_bottle_raw = get_cv_grasping_target(self.renderer, "bottle_visual_shell", bottle_z_center) if self.renderer else None
        cv_beaker_raw = get_cv_grasping_target(self.renderer, "target_visual_shell", beaker_z_center) if self.renderer else None

        # Build fully qualified XYZ using physics Z for robust IK positioning, but visual XY for pure localization!
        bottle_pos = d.site_xpos[bottle_site].copy()
        if cv_bottle_raw is not None:
            bottle_pos[0] = cv_bottle_raw[0]
            bottle_pos[1] = cv_bottle_raw[1]

        beaker_pos = d.site_xpos[beaker_site].copy()
        if cv_beaker_raw is not None:
            beaker_pos[0] = cv_beaker_raw[0]
            beaker_pos[1] = cv_beaker_raw[1]


        if action_name == "MOVE_ABOVE_BOTTLE":
            q = solve_ik(bottle_pos + np.array([0, 0, 0.08]), np.array([0, 0, -1]))
            self.move_to(q, 1.0, 2.0)
            return True

        elif action_name in ("LOWER_TO_BOTTLE", "LOWER_TO"):
            q = solve_ik(bottle_pos + np.array([0, 0, -0.01]), np.array([0, 0, -1]))
            self.move_to(q, 1.0, 1.5)
            return True

        elif action_name in ("CLOSE_GRIPPER", "CLOSE"):
            q_cur = d.qpos[qpos_ids[:5]].copy()
            self.move_to(q_cur, 0.0, 1.0)
            self.wait(0.5)
            self.gripper_val = -0.1
            return True

        elif action_name in ("LIFT_BOTTLE", "LIFT"):
            mujoco.mj_kinematics(m, d)
            bottle_pos = d.site_xpos[bottle_site].copy()
            q = solve_ik(bottle_pos + np.array([0, 0, 0.15]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 1.5)
            return True

        elif action_name in ("MOVE_TO_BEAKER", "MOVE_TO"):
            q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 2.0)
            return True

        elif action_name in ("TILT_POUR", "TILT"):
            angle_deg = 45.0
            if param:
                try:
                    angle_deg = float(param)
                except ValueError:
                    angle_deg = 45.0
            angle_deg = min(90.0, max(10.0, angle_deg))  # Clamp

            # Tilt by rotating wrist roll
            q_cur = d.qpos[qpos_ids[:5]].copy()
            q_tilt = q_cur.copy()
            q_tilt[4] += math.radians(angle_deg) * 1.5  # Scale for effect
            self.move_to(q_tilt, -0.1, 1.5)

            # Hold pour for 2 seconds while monitoring water flow
            self.wait(2.0)

            print(f"    💧 {self.water_sim.status_str()}")
            return True

        elif action_name in ("STOP_POUR", "STOP"):
            # Return to upright above beaker
            mujoco.mj_kinematics(m, d)
            beaker_pos = d.site_xpos[beaker_site].copy()
            q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 1.5)
            return True

        elif action_name == "RETURN_HOME":
            mujoco.mj_kinematics(m, d)
            beaker_pos = d.site_xpos[beaker_site].copy()
            q_above = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q_above, -0.1, 1.5)
            self.move_to(self.q_home, -0.1, 2.0)
            return True

        elif action_name == "DONE":
            return True

        else:
            print(f"    ⚠ Unknown action: {action_name}")
            return False


# ─── Main Loop ────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("=" * 60)
        print("ERROR: GROQ_API_KEY environment variable not set!")
        print("Get a free key at: https://console.groq.com")
        print("Then run: export GROQ_API_KEY='your-key-here'")
        print("=" * 60)
        return

    client = Groq(api_key=api_key)
    water_sim = WaterSimulator()
    seg_renderer = mujoco.Renderer(m, 480, 640)
    gazebo_bridge = GazeboUDPBridge()

    print("=" * 60)
    print("🤖 Llama 3 Gazebo Agent (Digital Twin)")
    print(f"   Model: {GROQ_MODEL}")
    print(f"   Target: Pour ~{TARGET_FILL_ML}ml into {BEAKER_CAPACITY_ML}ml beaker")
    print(f"   Bottle has: {BOTTLE_WATER_ML}ml")
    print("=" * 60)

    # Initialize
    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0

    # Let sim settle
    for _ in range(500):
        mujoco.mj_step(m, d)

    conversation_history = []
    phase = "start"
    step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        controller = MotionController(viewer, water_sim, gazebo_bridge, seg_renderer)

        while step < MAX_STEPS and viewer.is_running():
            step += 1
            print(f"\n{'─'*50}")
            print(f"Step {step}/{MAX_STEPS}")

            # Build observation with Vision
            obs = build_observation(water_sim, phase, controller.gripper_val > 0.5, seg_renderer)
            print(f"  📊 Water: {water_sim.status_str()}")

            # Query Llama 3
            print(f"  🧠 Querying Llama 3...")
            action_str = query_llama(client, obs, conversation_history)

            if not action_str:
                print("  ⚠ No response, retrying...")
                continue

            action_name, param = parse_action(action_str)
            param_str = f" {param}" if param else ""
            print(f"  🎯 Action: {action_name}{param_str}")

            # Safety check: force stop if near capacity
            if water_sim.beaker_ml >= 140:
                print("  🛑 SAFETY: Beaker near capacity! Forcing STOP_POUR")
                action_name = "STOP_POUR"
                param = None

            # Execute
            success = controller.execute(action_name, param)

            # Update phase
            if action_name == "MOVE_ABOVE_BOTTLE":
                phase = "approaching"
            elif action_name in ("LOWER_TO_BOTTLE", "LOWER_TO"):
                phase = "at_bottle"
            elif action_name in ("CLOSE_GRIPPER", "CLOSE"):
                phase = "grasped"
            elif action_name in ("LIFT_BOTTLE", "LIFT"):
                phase = "lifted"
            elif action_name in ("MOVE_TO_BEAKER", "MOVE_TO"):
                phase = "above_beaker"
            elif action_name in ("TILT_POUR", "TILT"):
                phase = "pouring"
            elif action_name in ("STOP_POUR", "STOP"):
                phase = "stopped_pour"
            elif action_name == "RETURN_HOME":
                phase = "returning"
            elif action_name == "DONE":
                print(f"\n{'='*60}")
                print(f"✅ TASK COMPLETE!")
                print(f"   Beaker: {water_sim.beaker_ml:.1f}/{BEAKER_CAPACITY_ML:.0f}ml")
                print(f"   Target was: {TARGET_FILL_ML}ml")
                overspill = "YES ⚠" if water_sim.beaker_ml > BEAKER_CAPACITY_ML else "NO ✅"
                print(f"   Overspill: {overspill}")
                print(f"{'='*60}")
                break

        # Keep viewer open 
        print("\n🎬 Simulation complete. Close the viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)


if __name__ == "__main__":
    main()
