from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

print("Instantiating Feetech Motor Bus...")
try:
    # We use a dummy port and a minimal motor config. We expect a specific serial timeout/connection error, 
    # NOT a ModuleNotFoundError or dependency crash.
    dummy_motor = Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)
    motors = {"dummy": dummy_motor}
    bus = FeetechMotorsBus(port="/dev/ttyUSB99", motors=motors)
    bus.connect()
except Exception as e:
    error_str = str(e).lower()
    if "could not open port" in error_str or "no such file" in error_str or "could not connect on port" in error_str:
        print("PASS: Feetech SDK loaded perfectly. (Correctly failed to find dummy port).")
    elif "modulenotfound" in error_str:
        print(f"FAIL: Missing dependencies for Feetech extra. {e}")
    else:
        print(f"WARNING: Unexpected error: {e}")