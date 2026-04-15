import mujoco
import numpy as np
import time

def run_diagnostics():
    print("Loading MuJoCo Scene...")
    try:
        model = mujoco.MjModel.from_xml_path('src/so101_mujoco/mujoco/scene.xml')
        data = mujoco.MjData(model)
        print("\n\u2714\ufe0f Model Loaded Successfully!")
    except Exception as e:
        print(f"\n\u274c FAILED TO LOAD MODEL: {e}")
        return

    # Check Geoms
    try:
        g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box1")
        g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_box_geom")
    except Exception:
        print("\n\u274c FAILED TO FIND GEOMS 'fixed_jaw_box1' and 'red_box_geom'. Are they spelled correctly?")
        return

    if g1 == -1 or g2 == -1:
        print("\n\u274c GEOM IDs returned -1. They don't exist in the loaded model.")
        return

    print("\n--- Physical Properties ---")
    def props(name, gid):
        print(f"{name} -> contype: {model.geom_contype[gid]}, conaffinity: {model.geom_conaffinity[gid]}")

    props("Gripper Box 1", g1)
    props("Red Box", g2)

    collide = (model.geom_contype[g1] & model.geom_conaffinity[g2]) or (model.geom_contype[g2] & model.geom_conaffinity[g1])
    print(f"\nMathematical Collision Engine Prediction: {'YES' if collide else 'NO'}")

    print("\n--- Physics Execution Test ---")
    
    # Move the red box to 0,0,0 and the robot gripper to 0,0,0
    print("Dropping Red Box onto the robot...")
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    box_joint_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")]
    
    mujoco.mj_forward(model, data)
    gripper_pos = data.geom_xpos[g1]
    
    # Teleport box explicitly
    data.qpos[box_joint_addr:box_joint_addr+3] = gripper_pos
    
    contacts = 0
    # Step physics a few frames to see if contact resolves
    for _ in range(10):
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 == g1 and c.geom2 == g2) or (c.geom1 == g2 and c.geom2 == g1):
                contacts += 1

    if contacts > 0:
        print("\n\u2714\ufe0f SUCCESS! MUJOCO IS PHYSICALLY REGISTERING COLLISION HITS BETWEEN THE GRIPPER AND THE BOX!")
        print("    If it's not working in Teleop, your Grip value is likely not closing far enough to physically touch the 3.5cm box!")
    else:
        print("\n\u274c FAILED. MuJoCo refuses to compute contacts between them even when forced to overlap.")
        
if __name__ == "__main__":
    run_diagnostics()
