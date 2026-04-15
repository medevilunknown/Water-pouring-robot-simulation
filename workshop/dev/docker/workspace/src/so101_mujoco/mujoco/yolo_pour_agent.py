"""
YOLOv8 + Llama 3 Robot Pouring Agent
=======================================
Extends the Llama 3 agent with live Computer Vision Object Detection via YOLOv8.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import json
import math
from PIL import Image
import threading
from groq import Groq
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
BEAKER_CAPACITY_ML = 150.0
TARGET_FILL_ML = 100.0
BOTTLE_WATER_ML = 120.0
POUR_RATE_ML_PER_SEC = 25.0
MAX_STEPS = 30

# ─── MuJoCo Setup ───────────────────────────────────────────────
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

# Init YOLO model (downloads on first run)
print("Loading YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt")

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
    def __init__(self):
        self.bottle_ml = BOTTLE_WATER_ML
        self.beaker_ml = 0.0
        self.beaker_capacity = BEAKER_CAPACITY_ML
        self.is_pouring = False
    
    def update(self, tilt_angle_deg, dt):
        if tilt_angle_deg > 30 and self.bottle_ml > 0:
            flow_factor = min(1.0, (tilt_angle_deg - 30) / 60.0)
            flow = POUR_RATE_ML_PER_SEC * flow_factor * dt
            actual_flow = min(flow, self.bottle_ml)
            self.bottle_ml -= actual_flow
            self.beaker_ml += actual_flow
            self.is_pouring = True
        else:
            self.is_pouring = False

# ─── State Observer & CV ─────────────────────────────────────────
def get_bottle_tilt_deg():
    bottle_qposadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")]
    quat = d.qpos[bottle_qposadr+3:bottle_qposadr+7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    cos_angle = np.clip(np.dot(rot[:, 2], [0, 0, 1]), -1, 1)
    return math.degrees(math.acos(cos_angle))

def detect_objects(renderer):
    """Render frame and return annotated YOLO image + detected labels."""
    renderer.update_scene(d, camera="external_camera")
    pixels = renderer.render()
    
    # YOLO inference (COCO model detects cup/bottle)
    results = yolo_model(pixels, verbose=False)[0]
    annotated_frame = results.plot()
    
    detections = []
    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        # x1, y1, x2, y2 = box.xyxy[0].tolist() # Optional coordinates
        detections.append({"class": cls_name, "confidence": round(conf, 2)})
        
    return annotated_frame, detections

def build_observation(water_sim, phase, gripper_open, detections):
    mujoco.mj_kinematics(m, d)
    grip_pos = d.site_xpos[grasp_site]
    bottle_pos = d.site_xpos[bottle_site]
    beaker_pos = d.site_xpos[beaker_site]
    tilt = get_bottle_tilt_deg()
    
    obs = {
        "phase": phase,
        "vision_detected_objects": [d['class'] for d in detections],
        "gripper": "open" if gripper_open else "closed",
        "bottle_tilt_deg": round(tilt, 1),
        "water": {
            "bottle_ml": round(water_sim.bottle_ml, 1),
            "beaker_ml": round(water_sim.beaker_ml, 1),
        }
    }
    return json.dumps(obs, indent=2)

# ─── LLM Agent ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You control a robot arm. Pick water bottle, move to beaker, pour ~100ml, and stop.
Wait for beaker_ml to reach 90-100ml then STOP_POUR immediately. Do NOT overspill.

Actions:
  MOVE_ABOVE_BOTTLE
  LOWER_TO_BOTTLE
  CLOSE_GRIPPER
  LIFT_BOTTLE
  MOVE_TO_BEAKER
  TILT_POUR 45
  STOP_POUR
  RETURN_HOME
  DONE
"""

def query_llama(client, observation, conversation_history):
    conversation_history.append({"role": "user", "content": observation})
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:],
            temperature=0.1, max_tokens=50,
        )
        action = response.choices[0].message.content.strip().split("\n")[0]
        conversation_history.append({"role": "assistant", "content": action})
        return action
    except Exception as e:
        print(f"  ⚠ LLM Error: {e}")
        return None

def parse_action(action_str):
    if not action_str: return None, None
    parts = action_str.strip().split()
    name = parts[0].upper()
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

# ─── Main Loop ────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY env via https://console.groq.com")
        return

    client = Groq(api_key=api_key)
    water_sim = WaterSimulator()
    renderer = mujoco.Renderer(m, 480, 640)
    
    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0

    print("🤖 YOLOv8 + Llama 3 Agent Starting...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        phase = "start"
        conversation_history = []
        gripper_val = 1.0
        
        def do_motion(target_q, new_gripper_val, duration):
            nonlocal gripper_val
            start_q = d.qpos[qpos_ids[:5]].copy()
            start_time = d.time
            while d.time - start_time < duration:
                alpha = (d.time - start_time) / duration
                alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
                d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
                d.ctrl[act_ids[5]] = new_gripper_val
                
                tilt = get_bottle_tilt_deg()
                water_sim.update(tilt, m.opt.timestep)
                mujoco.mj_step(m, d)
                viewer.sync()
                
                # Visual update every ~50 ticks
                if math.floor(d.time / m.opt.timestep) % 50 == 0:
                    frm, _ = detect_objects(renderer)
                    # YOLO plot returns BGR format, convert to RGB for PIL
                    Image.fromarray(frm[..., ::-1]).save("yolo_detection.jpg")
                    
                time.sleep(m.opt.timestep)
            gripper_val = new_gripper_val

        while step < MAX_STEPS and viewer.is_running():
            step += 1
            print(f"\nStep {step}/{MAX_STEPS}")
            
            # YOLO detect
            frame, detections = detect_objects(renderer)
            Image.fromarray(frame[..., ::-1]).save("yolo_detection.jpg")
            
            obs = build_observation(water_sim, phase, gripper_val > 0.5, detections)
            print(f"  👁 Vision Detections: {[dt['class'] for dt in detections]}")
            print(f"  🧠 Llama queried...")
            action_str = query_llama(client, obs, conversation_history)
            
            if not action_str: continue
            action_name, param = parse_action(action_str)
            print(f"  🎯 Action: {action_name}")

            bottle_pos = d.site_xpos[bottle_site].copy()
            beaker_pos = d.site_xpos[beaker_site].copy()

            if action_name == "MOVE_ABOVE_BOTTLE":
                q = solve_ik(bottle_pos + np.array([0, 0, 0.08]), np.array([0, 0, -1]))
                do_motion(q, 1.0, 2.0)
                phase = "approaching"
            elif action_name in ("LOWER_TO_BOTTLE", "LOWER_TO"):
                q = solve_ik(bottle_pos + np.array([0, 0, -0.01]), np.array([0, 0, -1]))
                do_motion(q, 1.0, 1.5)
                phase = "at_bottle"
            elif action_name in ("CLOSE_GRIPPER", "CLOSE"):
                q_cur = d.qpos[qpos_ids[:5]].copy()
                do_motion(q_cur, 0.0, 1.0)
                do_motion(q_cur, -0.1, 0.5)
                phase = "grasped"
            elif action_name in ("LIFT_BOTTLE", "LIFT"):
                q = solve_ik(bottle_pos + np.array([0, 0, 0.15]), np.array([0, 0, -1]))
                do_motion(q, -0.1, 1.5)
                phase = "lifted"
            elif action_name in ("MOVE_TO_BEAKER", "MOVE_TO"):
                q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
                do_motion(q, -0.1, 2.0)
                phase = "above_beaker"
            elif action_name in ("TILT_POUR", "TILT"):
                q_cur = d.qpos[qpos_ids[:5]].copy()
                q_tilt = q_cur.copy()
                q_tilt[4] += math.radians(45.0) * 1.5
                do_motion(q_tilt, -0.1, 1.5)
                do_motion(q_tilt, -0.1, 2.0)
                phase = "pouring"
            elif action_name in ("STOP_POUR", "STOP"):
                q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
                do_motion(q, -0.1, 1.5)
                phase = "stopped_pour"
            elif action_name == "RETURN_HOME":
                q_cur = d.qpos[qpos_ids[:5]].copy()
                do_motion(q_cur, -0.1, 1.0)
                do_motion(q_home, -0.1, 2.0)
                phase = "returning"
            elif action_name == "DONE":
                break

        print("🎬 Done! Close the MuJoCo viewer to exit.")
        while viewer.is_running():
            frm, _ = detect_objects(renderer)
            Image.fromarray(frm[..., ::-1]).save("yolo_detection.jpg")
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
