import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import json
import math
from groq import Groq
from vision_module import VisionModule

# ─── Configuration ───────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
BOTTLE_WATER_ML = 120.0
BEAKER_CAPACITY_ML = 150.0
POUR_RATE_ML_PER_SEC = 25.0
MAX_STEPS = 40
KNOWLEDGE_FILE = "knowledge.json"

# ─── SO-101 Agent ────────────────────────────────────────────────
class AutonomousPourAgent:
    def __init__(self, xml_path="scene.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.vision = VisionModule(self.m, self.d)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.act_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names]
        self.qpos_ids = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]
        self.dof_ids = [self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]
        
        self.grasp_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.beaker_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")
        
        self.bottle_ml = BOTTLE_WATER_ML
        self.beaker_ml = 0.0
        self.knowledge = self.load_knowledge()

    def load_knowledge(self):
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, "r") as f: return json.load(f)
        return []

    def save_knowledge(self, reflections):
        with open(KNOWLEDGE_FILE, "w") as f: json.dump(reflections, f, indent=2)

    def solve_ik_side_grasp(self, target_pos, fingers_dir):
        """4DOF IK constraint to keep wrist camera upright and fingers horizontal."""
        q = self.d.qpos.copy()
        # DOF selection: shoulder_pan (0), shoulder_lift (1), elbow_flex (2), wrist_flex (3)
        # We lock wrist_roll (4) to -1.57 (horizontal) to avoid camera collision
        arm_dofs = self.dof_ids[:4]
        arm_qpos = self.qpos_ids[:4]
        
        self.d.qpos[self.qpos_ids[4]] = -1.57 # Lock wrist_roll for camera clearance
        
        for _ in range(300):
            mujoco.mj_kinematics(self.m, self.d)
            pos = self.d.site_xpos[self.grasp_site]
            err_pos = target_pos - pos
            
            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.grasp_site)
            
            Jp = jacp[:, arm_dofs]
            delta_q = Jp.T @ err_pos * 2.0
            
            # Orientation: Finger axis (Site Z) should match fingers_dir
            mat = self.d.site_xmat[self.grasp_site].reshape(3, 3)
            site_z = mat[:, 2] # The 'fingers' axis
            err_rot = np.cross(site_z, fingers_dir)
            
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 0.5
            
            q[arm_qpos] += delta_q
            self.d.qpos[:] = q
            
        return q[self.qpos_ids[:5]]

    def move_smooth(self, target_q, gripper_val, duration=2.0, viewer=None):
        start_q = self.d.qpos[self.qpos_ids[:5]].copy()
        start_time = self.d.time
        while self.d.time - start_time < duration:
            alpha = (self.d.time - start_time) / duration
            alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
            self.d.ctrl[self.act_ids[:5]] = start_q + alpha * (target_q - start_q)
            self.d.ctrl[self.act_ids[5]] = gripper_val
            
            # Water simulation
            tilt = self.get_tilt_deg()
            if tilt > 30 and self.bottle_ml > 0:
                flow = POUR_RATE_ML_PER_SEC * min(1.0, (tilt-30)/60.0) * self.m.opt.timestep
                self.bottle_ml -= flow
                self.beaker_ml += flow

            mujoco.mj_step(self.m, self.d)
            if viewer: viewer.sync()
            time.sleep(self.m.opt.timestep)

    def get_tilt_deg(self):
        b_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
        b_qadr = self.m.jnt_qposadr[self.m.body_jntadr[b_id]]
        quat = self.d.qpos[b_qadr+3:b_qadr+7]
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return math.degrees(math.acos(np.clip(np.dot(rot[:, 2], [0, 0, 1]), -1, 1)))

    def query_llama(self, obs, conversation_history):
        system_prompt = """You are a robot controller. Output ONLY the action name from the list below. Do NOT explain. Do NOT use markdown.

Available Actions:
- MOVE_TO_BOTTLE
- GRASP_BOTTLE
- VERIFY_GRASP
- LIFT_BOTTLE
- MOVE_TO_BEAKER
- TILT_POUR <angle>
- STOP_POUR
- DONE

Example Response:
MOVE_TO_BOTTLE

Current "Lessons Learned" from past attempts: """ + (self.knowledge[-1] if self.knowledge else "None")

        user_msg = f"Current state: {json.dumps(obs)}. Respond with ONLY an action name."
        conversation_history.append({"role": "user", "content": user_msg})
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + conversation_history[-10:],
                temperature=0.0, max_tokens=20,
            )
            action = response.choices[0].message.content.strip().split("\n")[0].split(".")[0].strip().upper()
            conversation_history.append({"role": "assistant", "content": action})
            return action
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def parse_action(self, action_str):
        if not action_str: return None, None
        parts = action_str.strip().split()
        name = parts[0].upper()
        param = parts[1] if len(parts) > 1 else None
        return name, param

    def run_episode(self, ep_num):
        print(f"\n--- Episode {ep_num} Start ---")
        mujoco.mj_resetData(self.m, self.d)
        self.beaker_ml = 0.0
        self.bottle_ml = BOTTLE_WATER_ML
        
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            conversation_history = []
            phase = "idle"
            gripper_val = 1.0
            
            for step in range(MAX_STEPS):
                # 1. PERCEIVE
                vision_data = self.vision.detect_3d()
                bottle_pos = vision_data.get('bottle', self.d.site_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")])
                beaker_pos = self.d.site_xpos[self.beaker_site]
                
                grip_q = self.d.qpos[self.qpos_ids[5]]
                obs = {
                    "phase": phase,
                    "bottle_xyz": [round(x, 3) for x in bottle_pos.tolist()],
                    "beaker_ml": round(self.beaker_ml, 1),
                    "tilt": round(self.get_tilt_deg(), 1),
                    "gripper": "open" if gripper_val > 0.5 else ("closed" if grip_q < 0.1 else "holding_obj")
                }
                
                # 2. PLAN
                action_str = self.query_llama(obs, conversation_history)
                action_name, param = self.parse_action(action_str)
                print(f"  🎯 LLM Action: {action_name} {param or ''}")
                
                # 3. ACT
                if action_name == "MOVE_TO_BOTTLE":
                    # Multi-stage approach to avoid tilting the bottle
                    # 1. Move to "safe" height above approach point
                    approach_offset = np.array([0, -0.15, 0.05]) 
                    safe_target = bottle_pos + approach_offset
                    q_safe = self.solve_ik_side_grasp(safe_target, [0, -1, 0])
                    self.move_smooth(q_safe, 1.0, 1.5, viewer)
                    
                    # 2. Descend to approach height and move closer
                    final_approach = bottle_pos + np.array([0, -0.075, 0])
                    q_final = self.solve_ik_side_grasp(final_approach, [0, -1, 0])
                    self.move_smooth(q_final, 1.0, 1.5, viewer)
                    phase = "at_bottle"
                
                elif action_name == "GRASP_BOTTLE":
                    # Final horizontal approach before closing
                    target_ik = bottle_pos + np.array([0, -0.015, 0])
                    q = self.solve_ik_side_grasp(target_ik, [0, -1, 0])
                    self.move_smooth(q, 1.0, 1.0, viewer)
                    
                    # Close gripper
                    q_cur = self.d.qpos[self.qpos_ids[:5]].copy()
                    self.move_smooth(q_cur, -0.4, 0.8, viewer)
                    gripper_val = -0.4
                    phase = "grasp_attempted"
                
                elif action_name == "VERIFY_GRASP":
                    grip_q = self.d.qpos[self.qpos_ids[5]]
                    if grip_q > 0.1: # Significant resistance
                        print("✅ Grasp Verified: Bottle caught!")
                        phase = "grasped"
                    else:
                        print("❌ Grasp Failed: Jaws closed empty.")
                        phase = "idle"
                
                elif action_name == "LIFT_BOTTLE":
                    if phase != "grasped":
                        print("⚠️ Cannot lift: Not grasped.")
                    else:
                        q_cur = self.d.qpos[self.qpos_ids[:5]].copy()
                        q_cur[1] -= 0.2 
                        self.move_smooth(q_cur, -0.4, 1.5, viewer)
                        phase = "lifted"
                elif action_name == "MOVE_TO_BEAKER":
                    q = self.solve_ik_side_grasp(beaker_pos + [0, 0, 0.2], [0, -1, 0])
                    self.move_smooth(q, -0.4, 2.0, viewer)
                    phase = "above_beaker"
                elif action_name == "TILT_POUR":
                    angle = float(param or 45)
                    q_cur = self.d.qpos[self.qpos_ids[:5]].copy()
                    q_cur[4] += math.radians(angle) * 1.5
                    self.move_smooth(q_cur, -0.4, 2.0, viewer)
                    phase = "pouring"
                elif action_name == "STOP_POUR":
                    q = self.solve_ik_side_grasp(beaker_pos + [0, 0, 0.2], [0, -1, 0])
                    self.move_smooth(q, -0.4, 1.5, viewer)
                    phase = "stopped"
                elif action_name == "DONE":
                    break
                
                if not viewer.is_running(): break

            # 4. REFLECT
            score = "SUCCESS" if 90 < self.beaker_ml < 110 else "FAILED"
            print(f"Episode Final Volume: {self.beaker_ml:.1f}ml - {score}")
            
            # Simple reflection logic
            reflection = f"Episode {ep_num}: "
            if self.beaker_ml < 90: reflection += "Poured too little. Tilt more or longer."
            elif self.beaker_ml > 110: reflection += "Poured too much. Stop earlier."
            else: reflection += "Perfect pour. Maintain strategy."
            
            self.knowledge.append(reflection)
            self.save_knowledge(self.knowledge)

if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("Please export GROQ_API_KEY")
    else:
        agent = AutonomousPourAgent()
        agent.run_episode(1)
