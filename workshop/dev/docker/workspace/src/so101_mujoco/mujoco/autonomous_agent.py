"""
Autonomous SO-101 Pick-and-Pour Agent  (v10 – uses proven pour_pipeline IK)
=============================================================================
Deterministic state machine:
  1. DETECT  – YOLOv8 + depth → 3D bottle/beaker positions
  2. APPROACH– Multi-stage IK (proven 4-DOF from pour_pipeline)
  3. GRASP   – Close + VERIFY (qpos > 0.15 = holding bottle)
  4. LIFT    – Raise bottle
  5. MOVE    – Transport above beaker
  6. POUR    – Tilt wrist, emit particles, track volume
  7. REFLECT – LLM evaluates & stores lesson in knowledge.json
"""

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
GROQ_MODEL     = "llama-3.3-70b-versatile"
BOTTLE_ML      = 120.0
TARGET_ML      = 100.0
N_PARTICLES    = 60
KNOWLEDGE_FILE = "knowledge.json"

# Adaptive grasp trials: (Y_offset, Z_height_shift)
TRIALS = [
    ( 0.12,  0.01),   # Side +Y
    ( 0.12,  0.03),   # High side +Y
    (-0.12,  0.01),   # Side -Y
    (-0.12,  0.03),   # High side -Y
    ( 0.09, -0.01),   # Low side +Y
]

# ═══════════════════════════════════════════════════════════════════
class AutonomousPourAgent:

    def __init__(self, xml_path="scene.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.vision = VisionModule(self.m, self.d)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Joint bookkeeping
        jnames = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
        self.act  = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in jnames]
        self.qpos = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jnames]
        self.dof  = [self.m.jnt_dofadr [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jnames]

        self.jnt_range = np.array([
            self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in jnames[:5]
        ])

        self.grip_site   = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.bottle_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
        self.neck_site   = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "bottle_neck")
        self.beaker_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")
        self.bottle_fill = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_water_fill")
        self.target_fill = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "target_water_fill")

        # Water particles
        self.wp_qpos = []
        self.wp_qvel = []
        for i in range(N_PARTICLES):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"wpj_{i:02d}")
            self.wp_qpos.append(self.m.jnt_qposadr[jid])
            self.wp_qvel.append(self.m.jnt_dofadr[jid])

        self.bottle_ml = BOTTLE_ML
        self.glass_ml  = 0.0
        self.wp_next   = 0
        self.knowledge = self._load_kb()

    # ── knowledge base ────────────────────────────────────────────
    def _load_kb(self):
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE) as f:
                return json.load(f)
        return []

    def _save_kb(self):
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    # ── IK (proven 4-DOF from pour_pipeline, 500 iterations) ──────
    def solve_ik(self, target_pos, fingers_dir=None):
        """4-DOF IK targeting gripperframe with wrist_roll locked."""
        q = self.d.qpos.copy()
        arm_dofs = self.dof[:4]
        arm_qpos = self.qpos[:4]

        for _ in range(500):
            q[self.qpos[4]] = -1.57
            self.d.qpos[:] = q

            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)

            pos = self.d.site_xpos[self.grip_site]
            mat = self.d.site_xmat[self.grip_site].reshape(3, 3)

            err_pos = target_pos - pos

            jacp = np.zeros((3, self.m.nv))
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.grip_site)

            Jp = jacp[:, arm_dofs]
            dq = Jp.T @ err_pos * 4.0   # proven gain

            curr_fingers = mat[:, 2]
            if fingers_dir is not None:
                err_rot = np.cross(curr_fingers, fingers_dir)
                Jr = jacr[:, arm_dofs]
                dq += Jr.T @ err_rot * 1.5   # proven gain
            else:
                horizontal_err = np.array([0, 0, -curr_fingers[2]])
                Jr = jacr[:, arm_dofs]
                dq += Jr.T @ horizontal_err * 1.0

            q[arm_qpos] += dq

            # Clamp to joint limits
            for i, jid in enumerate(arm_qpos):
                lo, hi = self.jnt_range[i]
                if lo != hi:
                    q[jid] = np.clip(q[jid], lo, hi)
            self.d.qpos[:] = q

        return q[self.qpos[:5]]

    # ── smooth motion ─────────────────────────────────────────────
    def move(self, target_q, grip, dur, viewer):
        start_q = self.d.qpos[self.qpos[:5]].copy()
        t0 = self.d.time
        while self.d.time - t0 < dur:
            a = 0.5 - 0.5 * math.cos((self.d.time - t0) / dur * math.pi)
            self.d.ctrl[self.act[:5]] = start_q + a * (target_q - start_q)
            self.d.ctrl[self.act[5]] = grip
            mujoco.mj_step(self.m, self.d)
            if viewer:
                viewer.sync()
            time.sleep(self.m.opt.timestep)

    # ── water emitter ─────────────────────────────────────────────
    def emit_water(self, target_pos):
        if self.wp_next >= N_PARTICLES or self.bottle_ml <= 0:
            return
        neck = self.d.site_xpos[self.neck_site].copy()
        adr = self.wp_qpos[self.wp_next]
        self.d.qpos[adr:adr+3] = neck + np.random.uniform(-0.003, 0.003, 3)
        self.d.qpos[adr+3:adr+7] = [1, 0, 0, 0]

        direction = target_pos - neck
        direction[2] = -0.3
        direction /= np.linalg.norm(direction)

        vadr = self.wp_qvel[self.wp_next]
        self.d.qvel[vadr:vadr+3] = direction * (0.4 + np.random.uniform(-0.05, 0.05))
        self.d.qvel[vadr+3:vadr+6] = 0

        ml_per = BOTTLE_ML / N_PARTICLES
        self.wp_next += 1
        self.bottle_ml -= ml_per
        self.glass_ml  += ml_per

        bf = max(0, self.bottle_ml / BOTTLE_ML)
        gf = min(1, self.glass_ml / 150.0)
        self.m.geom_size[self.bottle_fill][1] = 0.038 * bf
        self.m.geom_size[self.target_fill][1] = 0.004 + 0.031 * gf
        self.m.geom_pos[self.target_fill][2]  = 0.006 + 0.031 * gf / 2

    # ══════════════════════════════════════════════════════════════
    #  MAIN EPISODE
    # ══════════════════════════════════════════════════════════════
    def run_episode(self, ep_num=1):
        print(f"\n{'='*60}")
        print(f"  SO-101 Autonomous Agent – Episode {ep_num}")
        print(f"{'='*60}")

        # Home pose
        q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
        self.d.qpos[self.qpos[:5]] = q_home
        self.d.ctrl[self.act[:5]]  = q_home
        self.d.ctrl[self.act[5]]   = 1.0
        self.bottle_ml = BOTTLE_ML
        self.glass_ml  = 0.0
        self.wp_next   = 0

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # Let physics settle
            for _ in range(100):
                mujoco.mj_step(self.m, self.d)
            viewer.sync()

            init_qpos = self.d.qpos.copy()
            init_qvel = self.d.qvel.copy()

            # ─── PHASE 1: DETECT ──────────────────────────────────
            print("\n🔍 Phase 1: DETECT – YOLOv8 + depth projection…")
            vision = self.vision.detect_3d()
            bottle_pos = vision.get(
                "bottle",
                self.d.site_xpos[self.bottle_site].copy()
            )
            beaker_pos = self.d.site_xpos[self.beaker_site].copy()
            print(f"   Bottle  @ {np.round(bottle_pos, 3)}")
            print(f"   Beaker  @ {np.round(beaker_pos, 3)}")

            # ─── PHASE 2+3: APPROACH + GRASP (adaptive trials) ───
            grasped = False
            for trial_idx, (y_off, z_shift) in enumerate(TRIALS):
                if trial_idx > 0:
                    print(f"\n  🔄 RESETTING for trial {trial_idx+1}…")
                    self.d.qpos[:] = init_qpos
                    self.d.qvel[:] = init_qvel
                    mujoco.mj_forward(self.m, self.d)
                    self.move(q_home, 1.0, 1.5, viewer)

                # Check bottle still upright
                mujoco.mj_kinematics(self.m, self.d)
                bottle_pos = self.d.site_xpos[self.bottle_site].copy()
                if bottle_pos[2] < 0.06:
                    print("  ❌ Bottle knocked over.")
                    continue

                print(f"\n🤖 Phase 2: APPROACH (trial {trial_idx+1}/{len(TRIALS)}, "
                      f"Y_off={y_off}, Z_shift={z_shift})")

                # ── PLAN ──
                # gripperframe is at FINGERTIPS. To center bottle in jaws,
                # the site must be 6cm PAST the bottle in the finger direction.
                fingers_dir = np.array([0, -1 if y_off > 0 else 1, 0])
                deep_offset = fingers_dir * 0.06

                target = bottle_pos.copy()
                target[2] += z_shift
                site_grasp = target + deep_offset

                q_pre   = self.solve_ik(site_grasp + fingers_dir * -0.12 + np.array([0, 0, 0.05]),
                                         fingers_dir)
                q_grasp = self.solve_ik(site_grasp, fingers_dir)
                q_lift  = self.solve_ik(site_grasp + np.array([0, 0, 0.15]), fingers_dir)
                q_above = self.solve_ik(beaker_pos + np.array([-0.05, 0.0, 0.20]),
                                         np.array([1, 0, 0]))
                print("  ✅ IK solutions computed")

                # ── APPROACH ──
                print("  → Pre-grasp waypoint…")
                self.move(q_pre, 1.0, 2.0, viewer)

                print(f"  → Reaching bottle (fingers → {fingers_dir})…")
                self.move(q_grasp, 1.0, 2.0, viewer)

                # ── GRASP ──
                print("\n🖐  Phase 3: GRASP")
                print("  → Closing gripper…")
                self.move(q_grasp, -0.4, 1.5, viewer)

                # Stabilize
                t0 = self.d.time
                while self.d.time - t0 < 1.0:
                    self.d.ctrl[self.act[5]] = -0.4
                    mujoco.mj_step(self.m, self.d)
                    viewer.sync()

                q_grip = self.d.qpos[self.qpos[5]]

                # ── VERIFY ──
                if q_grip < 0.15:
                    print(f"  ❌ Grasp FAILED (grip width = {q_grip:.3f})")
                    self.move(q_pre, 1.0, 1.0, viewer)
                    continue
                else:
                    print(f"  ✅ GRASP VERIFIED (grip width = {q_grip:.3f})")
                    grasped = True
                    break

            if not grasped:
                print("\n⛔ All grasp trials exhausted. Aborting.")
                self._reflect(ep_num, "GRASP_FAIL")
                return

            # ─── PHASE 4: LIFT ────────────────────────────────────
            print("\n⬆️  Phase 4: LIFT")
            self.move(q_lift, -0.4, 1.5, viewer)

            # ─── PHASE 5: MOVE TO BEAKER ──────────────────────────
            print("\n➡️  Phase 5: TRANSPORT to beaker")
            self.move(q_above, -0.4, 2.5, viewer)

            # ─── PHASE 6: POUR ────────────────────────────────────
            print(f"\n💧 Phase 6: POUR – target {TARGET_ML:.0f} ml")
            q_tilt = q_above.copy()
            q_tilt[4] += 1.8   # ~100° tilt
            print("  → Tilting…")
            self.move(q_tilt, -0.4, 2.0, viewer)

            # Hold tilt and emit particles
            t0 = self.d.time
            last_emit = 0
            while self.d.time - t0 < 6.0:
                self.d.ctrl[self.act[:5]] = q_tilt
                self.d.ctrl[self.act[5]] = -0.4

                if self.d.time - last_emit > 0.08 and self.bottle_ml > 0:
                    self.emit_water(self.d.site_xpos[self.beaker_site].copy())
                    last_emit = self.d.time

                if int((self.d.time - t0) * 10) % 10 == 0:
                    print(f"\r  Pouring: {self.glass_ml:.0f}ml in glass", end="", flush=True)

                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                time.sleep(self.m.opt.timestep)

            print(f"\n  Final volume: {self.glass_ml:.1f} ml")

            # Return upright
            print("\n🏠 Phase 6b: RETURN")
            self.move(q_above, -0.1, 1.5, viewer)
            self.move(q_home, 1.0, 2.0, viewer)

            # ─── PHASE 7: REFLECT ─────────────────────────────────
            self._reflect(ep_num, "POUR_DONE")

            print("\n✅ Episode complete – close viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                time.sleep(m.opt.timestep if hasattr(m, 'opt') else 0.01)

    # ── LLM Reflection ────────────────────────────────────────────
    def _reflect(self, ep_num, outcome):
        err = abs(self.glass_ml - TARGET_ML)
        score = "SUCCESS" if err < 15 else ("OVERFILL" if self.glass_ml > TARGET_ML else "UNDERFILL")
        print(f"\n📝 Phase 7: REFLECT – {score} (glass={self.glass_ml:.1f}ml, error={err:.1f}ml)")

        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content":
                     "You are a robotics coach. Analyze the episode and give ONE short rule to improve."},
                    {"role": "user", "content":
                     f"Episode {ep_num}: outcome={outcome}, glass={self.glass_ml:.1f}ml, "
                     f"target={TARGET_ML}ml, score={score}. "
                     f"Previous lessons: {self.knowledge[-3:] if self.knowledge else 'None'}"}
                ],
                temperature=0.3, max_tokens=60,
            )
            lesson = resp.choices[0].message.content.strip()
        except Exception as e:
            lesson = f"Episode {ep_num}: {score} ({self.glass_ml:.1f}ml). LLM error: {e}"

        print(f"   Lesson: {lesson}")
        self.knowledge.append(lesson)
        self._save_kb()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("⚠️  Please: export GROQ_API_KEY=your_key")
    else:
        agent = AutonomousPourAgent()
        agent.run_episode(1)
