"""
SO-101 Full Pour Pipeline (Resilient Adaptive Retry)
====================================================
Monitors the bottle state. If the gripper misses the bottle,
it resets and attempts a slightly different approach path.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# ─── Model ────────────────────────────────────────────────────
MODEL_PATH = "scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# ─── Joint mapping ───────────────────────────────────────────
SO101_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]
act_ids  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in SO101_JOINT_NAMES]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in SO101_JOINT_NAMES]
dof_ids  = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in SO101_JOINT_NAMES]

jnt_range = np.array([
    m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in SO101_JOINT_NAMES[:5]
])

# ─── Key IDs ─────────────────────────────────────────────────
grasp_site_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
bottle_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
bottle_neck_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bottle_neck")
target_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")
bottle_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_visual_shell")
bottle_fill_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_water_fill")
target_fill_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_water_fill")

# Water particles
N_PARTICLES = 60
wp_qposadr = []
wp_qveladr = []
for i in range(N_PARTICLES):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"wpj_{i:02d}")
    wp_qposadr.append(m.jnt_qposadr[jid])
    wp_qveladr.append(m.jnt_dofadr[jid])

# ─── IK Solver ──────────────────────────────────────────────
def solve_ik(target_pos, target_dir=None):
    """
    Improved 4-DOF IK Solver targeting the site 'gripperframe' directly.
    target_dir: The global vector the fingers (Local Z) should point towards.
    """
    q = d.qpos.copy()
    arm_dofs = dof_ids[:4]
    arm_qpos = qpos_ids[:4]

    for _ in range(500):
        # Lock wrist_roll to -1.57 (Camera Up)
        q[qpos_ids[4]] = -1.57
        d.qpos[:] = q
        
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        
        pos = d.site_xpos[grasp_site_id]
        mat = d.site_xmat[grasp_site_id].reshape(3, 3)
        
        err_pos = target_pos - pos
        
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, grasp_site_id)
        
        # 1. Positional goal
        Jp = jacp[:, arm_dofs]
        delta_q = Jp.T @ err_pos * 4.0
        
        # 2. Orientation goal: Keep fingers (Z) horizontal and pointed at target_dir
        curr_fingers = mat[:, 2]
        if target_dir is not None:
            err_rot = np.cross(curr_fingers, target_dir)
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 1.5
        else:
            # Just keep it horizontal
            horizontal_err = np.array([0, 0, -curr_fingers[2]])
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ horizontal_err * 1.0

        q[arm_qpos] += delta_q
        
        for i, jid in enumerate(arm_qpos):
            lo, hi = jnt_range[i]
            if lo != hi: q[jid] = np.clip(q[jid], lo, hi)
        d.qpos[:] = q
        
    return q[qpos_ids[:5]]

# ─── Motion ─────────────────────────────────────────────────
def move_to(target_q, gripper_val, viewer, duration=2.0):
    start_q = d.qpos[qpos_ids[:5]].copy()
    t0 = d.time
    while d.time - t0 < duration:
        alpha = (d.time - t0) / duration
        alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
        d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
        d.ctrl[act_ids[5]] = gripper_val
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)

# ─── Water Emitter ──────────────────────────────────────────
class WaterEmitter:
    def __init__(self):
        self.next = 0
        self.last_t = 0
        self.interval = 0.08
        self.bottle_ml = 120.0
        self.glass_ml = 0.0
        self.ml_per = 120.0 / N_PARTICLES
        
    def update(self, target_pos, t):
        if t - self.last_t < self.interval: return
        if self.next >= N_PARTICLES or self.bottle_ml <= 0: return
        
        i = self.next
        neck_pos = d.site_xpos[bottle_neck_id].copy()
        adr = wp_qposadr[i]
        d.qpos[adr:adr+3] = neck_pos + np.random.uniform(-0.003, 0.003, 3)
        d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
        
        direction = target_pos - neck_pos
        direction[2] = -0.3
        direction /= np.linalg.norm(direction)
        
        vadr = wp_qveladr[i]
        d.qvel[vadr:vadr+3] = direction * (0.4 + np.random.uniform(-0.05, 0.05))
        d.qvel[vadr+3:vadr+6] = 0
        
        self.next += 1
        self.bottle_ml -= self.ml_per
        self.glass_ml += self.ml_per
        self.last_t = t
        
        bf = max(0, self.bottle_ml / 120.0)
        gf = min(1, self.glass_ml / 150.0)
        m.geom_size[bottle_fill_id][1] = 0.038 * bf
        m.geom_size[target_fill_id][1] = 0.004 + 0.031 * gf
        m.geom_pos[target_fill_id][2] = 0.006 + 0.031 * gf / 2

# ─── Main Pipeline ──────────────────────────────────────────
def main():
    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0

    print("=" * 60)
    print("  SO-101 Pour Pipeline (Robust 4DOF Ver 3.0)")
    print("=" * 60)

    mj_reset = False
    trial_idx = 0
    
    # Adaptive trials: (Y_offset, Z_height_shift)
    TRIALS = [
        ( 0.12,  0.01),   # Trial 1: Side +Y
        ( 0.12,  0.03),   # Trial 2: High side +Y
        (-0.12,  0.01),   # Trial 3: Side -Y
        (-0.12,  0.03),   # Trial 4: High side -Y
        ( 0.09, -0.01),   # Trial 5: Low side +Y
    ]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        for _ in range(100): mujoco.mj_step(m, d)
        viewer.sync()

        init_qpos = d.qpos.copy()
        init_qvel = d.qvel.copy()

        while trial_idx < len(TRIALS):
            if mj_reset:
                print(f"\n  🔄 RESETTING ENV FOR TRIAL {trial_idx+1}...")
                d.qpos[:] = init_qpos
                d.qvel[:] = init_qvel
                mujoco.mj_forward(m, d)
                move_to(q_home, 1.0, viewer, 1.5)
                mj_reset = False

            y_off, z_shift = TRIALS[trial_idx]
            
            print(f"\n━━ Stage 1 · SENSE (Trial {trial_idx+1}) ━━")
            mujoco.mj_kinematics(m, d)
            bottle_pos = d.site_xpos[bottle_site_id].copy()
            glass_pos  = d.site_xpos[target_site_id].copy()
            
            # Check if bottle is upright
            if bottle_pos[2] < 0.06:
                print("  ❌ Bottle knocked over. Resetting.")
                mj_reset = True
                trial_idx += 1
                continue

            print("\n━━ Stage 2 · PLAN ━━")
            # The site 'gripperframe' is at the tips.
            # To center the bottle in the jaws, we move the site PAST the bottle.
            # Fingers are along site Z.
            fingers_dir = np.array([0, -1 if y_off > 0 else 1, 0])
            deep_offset = fingers_dir * 0.06 # go 6cm past center
            
            target_pos = bottle_pos.copy()
            target_pos[2] += z_shift
            
            # site_target = bottle_center + offset_to_tips + deep_offset
            # If fingers point to -Y, site should be at Y = bottle_Y - 0.06
            site_grasp = target_pos + deep_offset
            
            q_pre    = solve_ik(site_grasp + fingers_dir * -0.12 + np.array([0,0,0.05]), fingers_dir)
            q_grasp  = solve_ik(site_grasp, fingers_dir)
            q_lift   = solve_ik(site_grasp + np.array([0, 0, 0.15]), fingers_dir)
            q_above  = solve_ik(glass_pos + np.array([-0.05, 0.0, 0.20]), np.array([1,0,0]))
            print("  ✅ 4DOF IK computed (Wait for reach)")

            print("\n━━ Stage 3 · GRASP ATTEMPT ━━")
            print("  → Pre-grasp...")
            move_to(q_pre, 1.0, viewer, 2.0)
            
            print(f"  → Reaching bottle... (Fingers: {fingers_dir})")
            move_to(q_grasp, 1.0, viewer, 2.0)
            
            print(f"  → Closing gripper...")
            move_to(q_grasp, -0.4, viewer, 1.5) # Close harder
            
            # Wait to stabilize
            t0 = d.time
            while d.time - t0 < 1.0:
                d.ctrl[act_ids[5]] = -0.4
                mujoco.mj_step(m, d)
                viewer.sync()
            
            q_grip = d.qpos[qpos_ids[5]]
            
            # Width verification
            if q_grip < 0.15:
                print(f"  ❌ Missed bottle (grip width={q_grip:.3f})")
                move_to(q_pre, 1.0, viewer, 1.0)
                mj_reset = True
                trial_idx += 1
                continue
            
            print(f"  ✅ GRASP VERIFIED (width={q_grip:.3f})")
            break

        if trial_idx >= len(TRIALS):
            print("\n  ❌ All trials failed.")
            return

        print("\n━━ Stage 4 · LIFT ━━")
        move_to(q_lift, -0.4, viewer, 1.5)
        
        print("\n━━ Stage 5 · TRANSPORT ━━")
        move_to(q_above, -0.4, viewer, 2.5)

        print("\n━━ Stage 6 · POUR ━━")
        q_tilt = q_above.copy()
        q_tilt[4] += 1.8 
        print("  → Tilting...")
        move_to(q_tilt, -0.4, viewer, 2.0)
        
        emitter = WaterEmitter()
        t0 = d.time
        while d.time - t0 < 6.0:
            d.ctrl[act_ids[:5]] = q_tilt
            d.ctrl[act_ids[5]] = -0.4
            emitter.update(d.site_xpos[target_site_id].copy(), d.time)
            
            if int((d.time - t0) * 10) % 10 == 0:
                print(f"\r  Pouring: {emitter.glass_ml:.0f}ml in glass", end="", flush=True)

            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)
            
        print("\n  ✅ Pour complete")

        print("\n━━ Stage 7 · RETURN ━━")
        move_to(q_above, -0.1, viewer, 1.5)
        move_to(q_home, 1.0, viewer, 2.0)

        print("\n" + "=" * 60)
        print("  ✅ TASK COMPLETELY SUCCESSFUL")
        print("=" * 60)


        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
