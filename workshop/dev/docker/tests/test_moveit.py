import os
import sys

import rclpy

# Prefer the newer MoveIt 2 Python binding if available, but fall back to moveit_py (older).
# In ROS Humble the supported Python binding is packaged as ros-humble-pymoveit2.
BINDING_NAME = None
BINDINGS_AVAILABLE = False
IMPORT_ERROR = None

try:
    # moveit_py (older MoveIt 2 python bindings)
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    BINDING_NAME = "moveit_py"
    BINDINGS_AVAILABLE = True
except ImportError as e_moveit_py:
    try:
        # newer ROS2 binding (pymoveit2)
        import pymoveit2
        from pymoveit2 import MoveIt2
        BINDING_NAME = "pymoveit2"
        BINDINGS_AVAILABLE = True
    except ImportError as e_pymoveit2:
        IMPORT_ERROR = e_moveit_py
        BINDINGS_AVAILABLE = False

PYTHON_INFO = f"Python: {sys.executable} ({sys.version.splitlines()[0]})"
ROS_INFO = f"ROS_DISTRO={os.environ.get('ROS_DISTRO')} AMENT_PREFIX_PATH={os.environ.get('AMENT_PREFIX_PATH')}"

def main():
    print("Initializing MoveIt 2 Python bindings...")
    
    if not BINDINGS_AVAILABLE:
        print(f"FAIL: MoveIt 2 binding error: {IMPORT_ERROR}")
        print(f"  {PYTHON_INFO}")
        print(f"  {ROS_INFO}")
        print("Ensure you're running with the system Python used by ROS 2 (e.g. /usr/bin/python3) and that ROS 2 is sourced:")
        print("  source /opt/ros/humble/setup.bash")
        print("If you are inside a venv, deactivate it or run the test with /usr/bin/python3.")
        sys.exit(1)
        
    # We only need to confirm that the Python MoveIt bindings import correctly.
    # Full MoveIt usage would require a running ROS 2 graph and robot model.
    print(f"PASS: Python MoveIt 2 bindings are importable (binding={BINDING_NAME})")

if __name__ == '__main__':
    main()