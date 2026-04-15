import mujoco
import mujoco.viewer
import numpy as np
import time

# ─── Configuration ───────────────────────────────────────────────
MODEL_PATH = "scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
dof_ids = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]

grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_visual_shell")
jaw1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box")
jaw2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_box")

# ─── Vision Module ───────────────────────────────────────────────
def get_bottle_pos_vision(renderer):
    """
    Scans all available cameras to find and localize the bottle's main cylinder.
    """
    cameras = ["external_camera", "hand_view", "world_view"]
    
    for cam in cameras:
        # 1. Get Segmentation
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera=cam)
        seg = renderer.render().astype(int)
        
        # Check if ID exists in any channel
        mask = (seg == target_id)
        if np.any(mask):
            print(f"  👁 Vision: Bottle detected in camera '{cam}'")
            
            # 2. Get Depth
            renderer.disable_segmentation_rendering()
            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera=cam)
            depth = renderer.render()
            
            # Mask centroid
            mask_2d = np.any(mask, axis=-1) if mask.ndim == 3 else mask
            pixels = np.argwhere(mask_2d)
            v_mean, u_mean = pixels.mean(axis=0).astype(int)
            z_dist = depth[v_mean, u_mean]
            
            # 3. Project to World Coordinates
            cam_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            cam_pos = d.cam_xpos[cam_idx]
            cam_mat = d.cam_xmat[cam_idx].reshape(3, 3)
            
            fovy = m.cam_fovy[cam_idx]
            h, w = mask_2d.shape
            f = 0.5 * h / np.tan(np.radians(fovy) / 2.0)
            
            # Ray math
            dx = (u_mean - w / 2.0) / f
            dy = (h / 2.0 - v_mean) / f
            ray_cam = np.array([dx, dy, -1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = cam_mat @ ray_cam
            
            # 🚀 CORRECTION: target center by adding radius (0.031)
            world_pos = cam_pos + ray_world * (z_dist + 0.031)
            
            print(f"  👁 Vision Recognized Shape: Cylinder at {world_pos} (corrected to center)")
            renderer.disable_depth_rendering()
            return world_pos
            
    print("  👁 Vision Error: Bottle not detected in any camera!")
    return None

# ─── IK Controller ───────────────────────────────────────────────
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
            z_axis = mat[:, 2] # Gripper forward axis
            err_rot = np.cross(z_axis, target_z_axis)
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 1.5
            
        q[arm_qpos] += delta_q
        d.qpos[:] = q
    return q[arm_qpos]

# ─── Motion Helper ───────────────────────────────────────────────
def move_to(target_q, gripper_val, viewer, duration=2.0):
    start_q = d.qpos[qpos_ids[:5]].copy()
    start_time = d.time
    while d.time - start_time < duration:
        alpha = (d.time - start_time) / duration
        alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)
        d.ctrl[act_ids[:5]] = start_q + alpha * (target_q - start_q)
        d.ctrl[act_ids[5]] = gripper_val
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)

# ─── Contact Utility ──────────────────────────────────────────────
def check_contacts(g1, g2):
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 == g1 and c.geom2 == g2) or (c.geom1 == g2 and c.geom2 == g1):
            return True
    return False

# ─── Main Execution ──────────────────────────────────────────────
def main():
    renderer = mujoco.Renderer(m, 480, 640)
    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    
    # Reset
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0

    print("🚀 Starting Vision-Based Side Grasp Demo...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 1. Vision Phase
        print("Scanning environment...")
        for _ in range(50):
            mujoco.mj_step(m, d)
        viewer.sync()
        
        bottle_pos = get_bottle_pos_vision(renderer)
        if bottle_pos is None: return

        # 2. Planning Phase
        print("Planning side-approach trajectories...")
        target_z_axis = np.array([0, 1, 0]) # Forward along Y
        
        # Approaches from -Y side
        q_pre_grasp = solve_ik(bottle_pos + np.array([0, -0.15, 0]), target_z_axis)
        q_grasp = solve_ik(bottle_pos, target_z_axis)
        q_lift = solve_ik(bottle_pos + np.array([0, 0, 0.20]), target_z_axis)

        # 3. Execution Phase
        print("Step 1: Moving to side approach position...")
        move_to(q_pre_grasp, 1.0, viewer, 2.5)
        
        print("Step 2: Approaching for side grasp...")
        move_to(q_grasp, 1.0, viewer, 2.0)
        
        print("Step 3: Closing gripper on bottle...")
        move_to(q_grasp, 0.0, viewer, 1.5)
        
        # Verify Contacts
        print("Verifying contacts...")
        start = d.time
        while d.time - start < 1.0:
            d.ctrl[act_ids[5]] = -0.05
            mujoco.mj_step(m, d)
            viewer.sync()
            
        c1 = check_contacts(jaw1_id, target_id)
        c2 = check_contacts(jaw2_id, target_id)
        if c1 and c2:
            print("  ✅ Firm grasp confirmed!")
        else:
            print(f"  ⚠️ Contact Warning: Jaw1={c1}, Jaw2={c2}")

        print("Step 4: Lifting bottle...")
        move_to(q_lift, -0.05, viewer, 2.5)

        print("✅ Demo Complete! Inspect the hold in the viewer.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
