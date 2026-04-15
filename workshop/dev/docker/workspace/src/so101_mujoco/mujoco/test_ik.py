import mujoco
import numpy as np
import time

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

# Find joint and actuator indices
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
actuator_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]

grasp_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
bottle_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bottle_grasp")
target_glass_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_glass_site")

print("Actuator IDs:", actuator_ids)
print("QPos IDs:", qpos_ids)
print("Grasp Site ID:", grasp_site_id)
print("Bottle Site ID:", bottle_site_id)

def solve_ik(target_pos, target_z_axis=None):
    # Simple Jacobian transpose IK
    q = d.qpos.copy()
    for _ in range(200):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        
        pos = d.site_xpos[grasp_site_id]
        err_pos = target_pos - pos
        
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, grasp_site_id)
        
        # Only use the arm joints for IK
        J = jacp[:, qpos_ids[:5]]
        
        delta_q = J.T @ err_pos * 2.0
        
        if target_z_axis is not None:
            mat = d.site_xmat[grasp_site_id].reshape(3, 3)
            z_axis = mat[:, 2]
            err_rot = np.cross(z_axis, target_z_axis)
            Jr = jacr[:, qpos_ids[:5]]
            delta_q += Jr.T @ err_rot * 0.5
            
        q[qpos_ids[:5]] += delta_q
        d.qpos[:] = q
    
    return q[qpos_ids[:5]]

# 1. Approach bottle
mujoco.mj_kinematics(m, d)
bottle_pos = d.site_xpos[bottle_site_id].copy()
pre_grasp = bottle_pos + np.array([0, 0, 0.1])
q_pre = solve_ik(pre_grasp, target_z_axis=np.array([0, 0, -1]))
print("Pre-grasp angles:", q_pre)
