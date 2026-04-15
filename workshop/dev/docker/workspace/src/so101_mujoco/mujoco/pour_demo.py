import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
dof_ids = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]

grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

def solve_ik(target_pos, target_z_axis=None):
    # Simple Jacobian transpose IK
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
            # z-axis of the gripper frame points forward out of the jaw
            z_axis = mat[:, 2] 
            err_rot = np.cross(z_axis, target_z_axis)
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 1.0
            
        q[arm_qpos] += delta_q
        d.qpos[:] = q
    
    return q[arm_qpos]

# Waypoints
# 1. Home
q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])

mujoco.mj_kinematics(m, d)
bottle_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
beaker_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")

bottle_pos = d.site_xpos[bottle_site].copy()
beaker_pos = d.site_xpos[beaker_site].copy()

# Precalculate IK for waypoints
# Note: gripper approaches from above, so z-axis is [0, 0, -1]
q_pre_grasp = solve_ik(bottle_pos + np.array([0, 0, 0.08]), np.array([0, 0, -1]))
q_grasp = solve_ik(bottle_pos + np.array([0, 0, -0.01]), np.array([0, 0, -1]))

q_lift = solve_ik(bottle_pos + np.array([0, 0, 0.15]), np.array([0, 0, -1]))

q_above_beaker = solve_ik(beaker_pos + np.array([0, 0, 0.20]), np.array([0, 0, -1]))

# Tilted pour orientation 
q_pour = solve_ik(beaker_pos + np.array([-0.05, 0, 0.15]), np.array([0.7, 0, -0.7]))

print("Starting simulation sequence...")

with mujoco.viewer.launch_passive(m, d) as viewer:
    def move_to(target_q, gripper_val, duration=2.0):
        start_q = d.qpos[qpos_ids[:5]].copy()
        start_time = d.time
        while d.time - start_time < duration:
            alpha = (d.time - start_time) / duration
            alpha = 0.5 - 0.5 * np.cos(alpha * np.pi) # smooth interpolation
            
            d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
            d.ctrl[act_ids[5]] = gripper_val
            
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)
            
    # Reset
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0 # open

    # 1. Move to pre-grasp
    print("Moving above bottle...")
    move_to(q_pre_grasp, 1.0, 2.0)
    
    # 2. Go down and grasp
    print("Grasping...")
    move_to(q_grasp, 1.0, 1.5)
    move_to(q_grasp, 0.0, 1.0) # Close gripper
    
    # Give some time for particles to settle and grip to tighten
    start = d.time
    while d.time - start < 1.0:
        d.ctrl[act_ids[5]] = 0.0
        mujoco.mj_step(m, d)
        viewer.sync()
    
    # 3. Lift
    print("Lifting...")
    move_to(q_lift, -0.1, 1.5)
    
    # 4. Move above beaker
    print("Moving to beaker...")
    move_to(q_above_beaker, -0.1, 2.0)
    
    # 5. Pour
    print("Pouring...")
    # Rotate the wrist roll to tilt the bottle
    q_tilt = q_above_beaker.copy()
    q_tilt[4] += 2.0 # twist wrist roll
    
    move_to(q_tilt, -0.1, 2.0)
    
    # Hold to let water pour
    start = d.time
    while d.time - start < 3.0:
        mujoco.mj_step(m, d)
        viewer.sync()
        
    print("Done! Returning...")
    move_to(q_above_beaker, -0.1, 1.5)
    move_to(q_home, -0.1, 2.0)
    
    # Keep viewer open
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)
