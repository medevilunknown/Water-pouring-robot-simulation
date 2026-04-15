import mujoco
import mujoco.viewer
import time

print("Creating empty MuJoCo model...")
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
data = mujoco.MjData(model)

print("Launching GUI. Close the window to pass the test.")
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 5:
            mujoco.mj_step(model, data)
            viewer.sync()
    print("PASS: MuJoCo 3D rendering successful!")
except Exception as e:
    print(f"FAIL: {e}")