"""
Llama 3 Self-Learning Robot Pouring Agent
=========================================
Uses Groq API (Llama 3.3-70b) and MuJoCo to learn how to pour optimally.
The LLM evaluates itself after every episode and generates a Reflection Rule.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import json
import math
import random
from groq import Groq

# ─── Configuration ───────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
BEAKER_CAPACITY_ML = 150.0
TARGET_FILL_ML = 100.0          
BOTTLE_WATER_ML = 120.0         
POUR_RATE_ML_PER_SEC = 25.0     
MAX_STEPS = 30                  
MAX_EPISODES = 5
KNOWLEDGE_FILE = "knowledge.json"

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
    def __init__(self, bottle_ml=BOTTLE_WATER_ML, beaker_capacity=BEAKER_CAPACITY_ML):
        self.bottle_ml = bottle_ml
        self.beaker_ml = 0.0
        self.beaker_capacity = beaker_capacity
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
                f"Pouring: {'YES' if self.is_pouring else 'no'}")

# ─── State Observer ──────────────────────────────────────────────
def get_bottle_tilt_deg():
    bottle_qposadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")]
    quat = d.qpos[bottle_qposadr+3:bottle_qposadr+7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    z_axis = rot[:, 2]
    cos_angle = np.clip(np.dot(z_axis, [0, 0, 1]), -1, 1)
    return math.degrees(math.acos(cos_angle))

def get_object_bbox(renderer, obj_name):
    geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
    if geom_id == -1: return None
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera="external_camera")
    seg = renderer.render()
    mask = (seg[:, :, 0] == 5) & (seg[:, :, 1] == geom_id)
    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0: return None
    return [int(np.min(x_idx)), int(np.min(y_idx)), int(np.max(x_idx)), int(np.max(y_idx))]

def build_observation(water_sim, phase, gripper_open, renderer=None):
    mujoco.mj_kinematics(m, d)
    grip_pos = d.site_xpos[grasp_site]
    bottle_pos = d.site_xpos[bottle_site]
    beaker_pos = d.site_xpos[beaker_site]
    joint_pos = [d.qpos[qpos_ids[i]] for i in range(6)]
    tilt = get_bottle_tilt_deg()
    
    detections = {}
    if renderer:
        bottle_box = get_object_bbox(renderer, "bottle_visual_shell")
        beaker_box = get_object_bbox(renderer, "target_visual_shell")
        if bottle_box: detections["bottle"] = f"bbox {bottle_box}"
        if beaker_box: detections["beaker"] = f"bbox {beaker_box}"

    obs = {
        "phase": phase,
        "vision_object_detection": detections,
        "gripper_pos": [round(x, 4) for x in grip_pos],
        "bottle_pos": [round(x, 4) for x in bottle_pos],
        "beaker_pos": [round(x, 4) for x in beaker_pos],
        "bottle_tilt_deg": round(tilt, 1),
        "water": {
            "bottle_ml": round(water_sim.bottle_ml, 1),
            "beaker_ml": round(water_sim.beaker_ml, 1),
            "target_fill_ml": TARGET_FILL_ML,
            "is_overspilling": water_sim.is_overspilling,
        }
    }
    return json.dumps(obs, indent=2)

# ─── REFLECTION & LEARNING MEMORY ─────────────────────────────────
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_knowledge(reflections):
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(reflections, f, indent=2)

def generate_reflection(client, episode_history, score, beaker_ml):
    print("  🧠 Generating Reflection on past episode...")
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in episode_history if msg['role'] == "assistant"])
    prompt = f"""You just completed an episode of robot pouring simulation.
Target: 100ml. You poured: {beaker_ml:.1f}ml. Score type: {score}.
Action sequence summary:
{history_text}

Provide ONE short, concise rule (max 2 sentences) to improve performance in the next episode. Focus heavily on when to STOP pouring, or whether the TILT angle was too aggressive.
Respond with ONLY the actionable rule."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "You are formulating self-improvement rules for a robot."}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Reflection Error: {e}")
        return "Always monitor beaker_ml and STOP_POUR at 90ml."

# ─── LLM Agent ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot arm controller controlling a SO-101 6-DOF arm.
Task: Pick up water bottle, move above beaker, pour water to ~100ml without overspilling (cap: 150ml).
You receive JSON state. Output EXACTLY ONE action.

Available actions:
  MOVE_ABOVE_BOTTLE
  LOWER_TO_BOTTLE
  CLOSE_GRIPPER
  LIFT_BOTTLE
  MOVE_TO_BEAKER
  TILT_POUR <angle>
  STOP_POUR
  RETURN_HOME
  DONE

Tips: Pick up bottle sequence → MOVE_TO_BEAKER → TILT_POUR 45 to start → watch beaker_ml → STOP_POUR when hitting ~90ml.
CRITICAL: Output ONLY the action, nothing else."""

def query_llama(client, observation, conversation_history, rules=""):
    appended_prompt = SYSTEM_PROMPT
    if list(rules):
        appended_prompt += "\n\nPAST LESSONS (Apply strictly!):\n" + "\n".join(f"- {r}" for r in rules)

    conversation_history.append({"role": "user", "content": f"Current state:\n{observation}\n\nAction?"})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": appended_prompt}] + conversation_history[-10:],
            temperature=0.1, max_tokens=50,
        )
        action = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": action})
        return action
    except:
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

# ─── Motion Controller ──────────────────────────────────────────
class MotionController:
    def __init__(self, viewer, water_sim):
        self.viewer = viewer
        self.water_sim = water_sim
        self.gripper_val = 1.0
        self.q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])

    def move_to(self, target_q, gripper_val, duration=1.0): # Faster execution for RL Loop
        start_q = d.qpos[qpos_ids[:5]].copy()
        start_time = d.time
        while d.time - start_time < duration:
            alpha = (d.time - start_time) / duration
            alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
            d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
            d.ctrl[act_ids[5]] = gripper_val
            tilt = get_bottle_tilt_deg()
            self.water_sim.update(tilt, m.opt.timestep)
            mujoco.mj_step(m, d)
            self.viewer.sync()
            time.sleep(m.opt.timestep / 4.0) # Speed up simulation render sleep
        self.gripper_val = gripper_val

    def wait(self, duration=1.0):
        start = d.time
        while d.time - start < duration:
            tilt = get_bottle_tilt_deg()
            self.water_sim.update(tilt, m.opt.timestep)
            mujoco.mj_step(m, d)
            self.viewer.sync()
            time.sleep(m.opt.timestep / 4.0)

    def execute(self, action_name, param=None):
        mujoco.mj_kinematics(m, d)
        bottle_pos = d.site_xpos[bottle_site].copy()
        beaker_pos = d.site_xpos[beaker_site].copy()

        if action_name == "MOVE_ABOVE_BOTTLE":
            q = solve_ik(bottle_pos + np.array([0, 0, 0.08]), np.array([0, 0, -1]))
            self.move_to(q, 1.0, 1.0)
        elif action_name in ("LOWER_TO_BOTTLE", "LOWER_TO"):
            q = solve_ik(bottle_pos + np.array([0, 0, -0.01]), np.array([0, 0, -1]))
            self.move_to(q, 1.0, 1.0)
        elif action_name in ("CLOSE_GRIPPER", "CLOSE"):
            q_cur = d.qpos[qpos_ids[:5]].copy()
            self.move_to(q_cur, 0.0, 1.0)
            self.wait(0.2)
            self.gripper_val = -0.1
        elif action_name in ("LIFT_BOTTLE", "LIFT"):
            mujoco.mj_kinematics(m, d)
            bottle_pos = d.site_xpos[bottle_site].copy()
            q = solve_ik(bottle_pos + np.array([0, 0, 0.15]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 1.0)
        elif action_name in ("MOVE_TO_BEAKER", "MOVE_TO"):
            q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 1.5)
        elif action_name in ("TILT_POUR", "TILT"):
            a_deg = 45.0
            if param:
                try: a_deg = float(param)
                except: a_deg = 45.0
            a_deg = min(90.0, max(10.0, a_deg))
            q_cur = d.qpos[qpos_ids[:5]].copy()
            q_tilt = q_cur.copy()
            q_tilt[4] += math.radians(a_deg) * 1.5
            self.move_to(q_tilt, -0.1, 1.0)
            self.wait(1.5)
            print(f"    💧 {self.water_sim.status_str()}")
        elif action_name in ("STOP_POUR", "STOP"):
            mujoco.mj_kinematics(m, d)
            beaker_pos = d.site_xpos[beaker_site].copy()
            q = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q, -0.1, 1.0)
        elif action_name == "RETURN_HOME":
            mujoco.mj_kinematics(m, d)
            beaker_pos = d.site_xpos[beaker_site].copy()
            q_above = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))
            self.move_to(q_above, -0.1, 1.0)
            self.move_to(self.q_home, -0.1, 1.5)
        elif action_name == "DONE":
            pass
        return True

# ─── Episodic Loop ────────────────────────────────────────────────
def run_episode(ep_num, client, viewer, seg_renderer, knowledge):
    print(f"\n{'='*70}")
    print(f"🎬 BEGIN EPISODE {ep_num}/{MAX_EPISODES} 🎬")
    print(f"Loading {len(knowledge)} lessons from memory...")
    print(f"{'='*70}")

    # Randomize starting state safely
    mujoco.mj_resetData(m, d)
    
    # Safe bounds for table placement
    bx = random.uniform(0.15, 0.22)
    by = random.uniform(-0.15, -0.05)
    gx = random.uniform(0.23, 0.30)
    gy = random.uniform(0.05, 0.15)
    
    b_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
    g_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_glass")
    
    b_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")]
    g_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_glass_joint")]
    
    d.qpos[b_qadr:b_qadr+3] = [bx, by, 0]
    d.qpos[g_qadr:g_qadr+3] = [gx, gy, 0]

    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0
    for _ in range(200): mujoco.mj_step(m, d)

    water_sim = WaterSimulator()
    controller = MotionController(viewer, water_sim)
    conversation_history = []
    phase = "start"
    step = 0

    while step < MAX_STEPS and viewer.is_running():
        step += 1
        print(f"\n  [Ep {ep_num} | Step {step}/{MAX_STEPS}] Water: {water_sim.beaker_ml:.1f}/150ml")
        obs = build_observation(water_sim, phase, controller.gripper_val > 0.5, seg_renderer)
        action_str = query_llama(client, obs, conversation_history, rules=knowledge)
        if not action_str: continue

        action_name, param = parse_action(action_str)
        print(f"  🎯 Action: {action_name} {param or ''}")

        if water_sim.beaker_ml >= 140:
            print("  🛑 SAFETY: Forcing STOP_POUR")
            action_name = "STOP_POUR"; param = None

        controller.execute(action_name, param)

        if action_name == "DONE": break
        elif action_name == "MOVE_ABOVE_BOTTLE": phase = "approaching"
        elif action_name in ("CLOSE_GRIPPER", "CLOSE"): phase = "grasped"
        elif action_name in ("MOVE_TO_BEAKER", "MOVE_TO"): phase = "above_beaker"
        elif action_name in ("TILT_POUR", "TILT"): phase = "pouring"
        elif action_name in ("STOP_POUR", "STOP"): phase = "stopped_pour"

    # Evaluation
    score = "FAILED"
    if 90.0 <= water_sim.beaker_ml <= 115.0: score = "SUCCESS (Safe State)"
    elif water_sim.beaker_ml > 115.0: score = "FAILED (Overspill)"
    elif water_sim.beaker_ml < 90.0: score = "FAILED (Undertarget)"

    print(f"\n  📊 EPISODE {ep_num} RESULT: {score} | End Volume: {water_sim.beaker_ml:.1f}ml")
    
    reflection = generate_reflection(client, conversation_history, score, water_sim.beaker_ml)
    if reflection:
        print(f"  💡 New Knowledge: {reflection}")
        knowledge.append(reflection)
        save_knowledge(knowledge)

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set!")
        return

    client = Groq(api_key=api_key)
    knowledge = load_knowledge()
    seg_renderer = mujoco.Renderer(m, 480, 640)

    print("🤖 Starting Self-Adapting RL Agent")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for ep in range(1, MAX_EPISODES + 1):
            if not viewer.is_running(): break
            run_episode(ep, client, viewer, seg_renderer, knowledge)
            
        print("\n🎬 All Episodic Training Complete. Close viewer to exit.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
