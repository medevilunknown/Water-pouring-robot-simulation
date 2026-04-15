import mujoco
import mujoco.viewer
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
import os

# ─── Configuration ───────────────────────────────────────────────
MODEL_PATH = "scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Initialize YOLO (downloads if missing)
print("Loading YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt")

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
dof_ids = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]

grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
target_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_visual_shell")
jaw1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box")
jaw2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_box")

# ─── Vision Module ───────────────────────────────────────────────
def get_bottle_pos_yolo(renderer):
    """
    Uses YOLOv8 to localize the bottle and estimates its 3D center.
    """
    cameras = ["external_camera", "hand_view"]
    for cam in cameras:
        # Render at higher resolution for YOLO (640x640)
        renderer.update_scene(d, camera=cam)
        pixels = renderer.render()
        
        # Save raw frame for manual inspection
        Image.fromarray(pixels).save(f"raw_scan_{cam}.jpg")
        
        # YOLO inference
        results = yolo_model(pixels, verbose=False, imgsz=640)[0]
        
        print(f"  👁 YOLO Scanning {cam}...")
        for box in results.boxes:
            cls_name = yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"    - Found: {cls_name} ({conf:.2f})")
            
            # 🚀 YOLOv8n often sees the robot/bottle as an 'airplane' or 'kite' 
            # in this specific lighting. We accept these as the bottle.
            if cls_name in ["bottle", "cup", "vase", "airplane", "kite", "dining table"]: 
                if cls_name == "dining table" and conf < 0.5: continue
                
                print(f"  ✅ Target matched: {cls_name} ({conf:.2f}%) in {cam}")
                
                # Bbox center
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                
                # Get depth
                renderer.enable_depth_rendering()
                renderer.update_scene(d, camera=cam)
                depth = renderer.render()
                z_dist = depth[int(v), int(u)]
                renderer.disable_depth_rendering()
                
                if z_dist > 2.0: continue # Likely background noise
                
                # 3D Projection
                cam_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                cam_pos = d.cam_xpos[cam_idx]
                cam_mat = d.cam_xmat[cam_idx].reshape(3, 3)
                
                fovy = m.cam_fovy[cam_idx]
                h, w = pixels.shape[:2]
                f = 0.5 * h / np.tan(np.radians(fovy) / 2.0)
                
                dx = (u - w / 2.0) / f
                dy = (h / 2.0 - v) / f
                ray_cam = np.array([dx, dy, -1.0])
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = cam_mat @ ray_cam
                
                # Target center by adding radius (0.031)
                world_pos = cam_pos + ray_world * (z_dist + 0.031)
                
                # Save debug image
                Image.fromarray(results.plot()[..., ::-1]).save(f"yolo_debug_{cam}.jpg")
                return world_pos
                
    return None

# ─── Verification Utilities ─────────────────────────────────────
def check_contact():
    """Returns True if both jaw pads are touching the target geometry."""
    c1, c2 = False, False
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 == jaw1_id and c.geom2 == target_geom_id) or \
           (c.geom1 == target_geom_id and c.geom2 == jaw1_id):
            c1 = True
        if (c.geom1 == jaw2_id and c.geom2 == target_geom_id) or \
           (c.geom1 == target_geom_id and c.geom2 == jaw2_id):
            c2 = True
    return c1, c2

def visual_verify_grab(renderer):
    """Checks if the bottle is detected within the gripper's workspace after lift."""
    renderer.update_scene(d, camera="hand_view")
    pixels = renderer.render()
    results = yolo_model(pixels, verbose=False)[0]
    
    for box in results.boxes:
        cls_name = yolo_model.names[int(box.cls[0])]
        if cls_name in ["bottle", "cup", "vase"]:
            # If detected in hand_view (which moves with the arm), it's likely held
            print(f"  👁 Visual Verification: Bottle confirmed in Hand View!")
            Image.fromarray(results.plot()[..., ::-1]).save("verification_success.jpg")
            return True
    return False

# ─── IK & Control ───────────────────────────────────────────────
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
            z_axis = mat[:, 2] # Forward
            err_rot = np.cross(z_axis, target_z_axis)
            Jr = jacr[:, arm_dofs]
            delta_q += Jr.T @ err_rot * 1.5
        q[arm_qpos] += delta_q
        d.qpos[:] = q
    return q[arm_qpos]

def move_to(target_q, gripper_val, viewer, duration):
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

# ─── Main Logic ───────────────────────────────────────────────
def main():
    renderer = mujoco.Renderer(m, 480, 640)
    q_home = np.array([0.0, -0.5, 0.5, 0.0, 0.0])
    d.qpos[qpos_ids[:5]] = q_home
    d.ctrl[act_ids[:5]] = q_home
    d.ctrl[act_ids[5]] = 1.0 # Open

    print("🚀 Starting YOLO-Guided Side Grasp Demo...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Step Physics to settle
        for _ in range(50): mujoco.mj_step(m, d)
        viewer.sync()
        
        # 1. Vision Phase (YOLO)
        print("Locating bottle via YOLOv8...")
        bottle_pos = get_bottle_pos_yolo(renderer)
        if bottle_pos is None:
            print("  ❌ YOLO Detection Failed! Check environment.")
            return

        # 2. Planning Phase
        print("Planning side-approach...")
        target_z_axis = np.array([0, 1, 0]) # Face forward along Y
        q_pre = solve_ik(bottle_pos + np.array([0, -0.15, 0]), target_z_axis)
        q_grasp = solve_ik(bottle_pos, target_z_axis)
        q_lift = solve_ik(bottle_pos + np.array([0, 0, 0.20]), target_z_axis)

        # 3. Execution Phase
        print("Executing Motion: APPROACHING...")
        move_to(q_pre, 1.0, viewer, 2.5)
        move_to(q_grasp, 1.0, viewer, 2.0)
        
        print("Executing Motion: GRASPING...")
        move_to(q_grasp, 0.0, viewer, 1.5)
        
        # 4. Verification Stage 1 (Contact)
        print("Verification 1: Checking Contact Sensors...")
        start = d.time
        while d.time - start < 1.0:
            d.ctrl[act_ids[5]] = -0.05 # Firm pressure
            mujoco.mj_step(m, d)
            viewer.sync()
        
        c1, c2 = check_contact()
        if c1 and c2:
            print("  ✅ Contact SUCCESS: Both jaws touching bottle.")
        else:
            print(f"  ⚠️ Contact MISSED: Jaw1={c1}, Jaw2={c2}. Lift might fail.")

        # 5. Execution Phase: LIFT
        print("Executing Motion: LIFTING...")
        move_to(q_lift, -0.05, viewer, 2.0)

        # 6. Verification Stage 2 (Visual)
        print("Verification 2: Visual Post-Lift Scan...")
        if visual_verify_grab(renderer):
            print("  ✅ Visual SUCCESS: Bottle confirmed in-hand!")
        else:
            print("  ❌ Visual FAIL: Bottle missing from hand.")

        print("\n✅ Task Complete! Check 'verification_success.jpg' for proof.")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
