import mujoco
import numpy as np
from PIL import Image

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 640)

# Settle physics
for _ in range(100):
    mujoco.mj_step(m, d)

cameras = ["external_camera", "world_view", "hand_view"]
bottle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bottle_visual_shell")

print(f"Targeting bottle_visual_shell (ID: {bottle_id})")

for cam in cameras:
    print(f"\nScanning: {cam}")
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render()
    
    unique_ids = np.unique(seg)
    print(f"  Geoms seen: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in unique_ids if i >= 0]}")
    
    mask = (seg == bottle_id)
    if np.any(mask):
        print(f"  ✅ FOUND BOTTLE in {cam}")
        # Save a debug image of the mask
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"debug_mask_{cam}.png")
    else:
        print(f"  ❌ Bottle not visible in {cam}")

    # Save RGB for sanity
    r.disable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(f"debug_rgb_{cam}.png")
