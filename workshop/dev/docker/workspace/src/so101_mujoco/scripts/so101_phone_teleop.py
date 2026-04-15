#!/usr/bin/env python3
import os
import time
import socket
import json
import numpy as np
import threading
from scipy.spatial.transform import Rotation

try:
    import ikpy.chain
    from teleop import Teleop
except ImportError:
    print("Dependencies missing! Please run: pip install teleop ikpy scipy")
    exit(1)

class SimpleWebXRPhone:
    """Standalone WebXR interface for ANY Phone (iOS or Android) via the browser"""
    def __init__(self):
        self.camera_offset = np.array([0.0, -0.02, 0.04])
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = {}
        self._lock = threading.Lock()
        
        self._calib_pos = np.zeros(3)
        self._calib_rot_inv = Rotation.identity()
        self.enabled = False
        self._gripper_angle = 1.74  # Fully open

    def connect(self):
        print("Starting WebXR Server...")
        self._teleop = Teleop()
        
        # MONKEY-PATCH: Disable local SSL so WebXR Viewer doesn't block the WebSocket handshake on self-signed certs!
        # The external Pinggy tunnel will handle the real SSL for us.
        self._teleop.ssl_keyfile = None
        self._teleop.ssl_certfile = None
        
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        
        print("\n=======================================================")
        print("1. Open the 'https://xxx.pinggy-free.link' URL in WebXR Viewer.")
        print("2. Agree to camera permissions.")
        print("3. Point your phone forward, TOUCH AND DRAG the screen to calibrate.")
        print("=======================================================\n")
        
        while True:
            with self._lock:
                msg = self._latest_message
                p = self._latest_pose
                
            if msg.get("move", False) and p is not None:
                rot = Rotation.from_matrix(p[:3, :3])
                raw_pos = p[:3, 3] - rot.apply(self.camera_offset)
                
                self._calib_pos = raw_pos.copy()
                self._calib_rot_inv = rot.inv()
                self.enabled = False
                print("\n✅ Phone Calibrated successfully!")
                break
            time.sleep(0.01)

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        with self._lock:
            self._latest_pose = pose
            self._latest_message = message

    def get_action(self):
        with self._lock:
            p = self._latest_pose
            msg = self._latest_message
            
        if p is None: return None
        
        rot = Rotation.from_matrix(p[:3, :3])
        raw_pos = p[:3, 3] - rot.apply(self.camera_offset)
        
        just_enabled = False
        enable = bool(msg.get("move", False))
        if enable and not self.enabled:
            # Re-zero position when you lift your finger and touch again (clutching)
            self._calib_pos = raw_pos.copy()
            just_enabled = True
            
        self.enabled = enable
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        
        if msg.get("reservedButtonA", False):
            self._gripper_angle = max(0.0, self._gripper_angle - 0.1)
        elif msg.get("reservedButtonB", False):
            self._gripper_angle = min(1.74, self._gripper_angle + 0.1)
            
        return {
            'phone.enabled': enable,
            'phone.just_enabled': just_enabled,
            'phone.pos': pos_cal,
            'gripper_a3': self._gripper_angle
        }

class PhoneTeleopNode:
    def __init__(self):
        print("Setting up Teleop Node (WebXR Tunnel Mode)")
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mujoco_bridge_addr = ("127.0.0.1", 9876)

        self.phone = SimpleWebXRPhone()
        
        urdf_path = "src/so101_description/urdf/so101.urdf"
        if not os.path.exists(urdf_path):
            print(f"URDF not found at {urdf_path}! Cannot run IK.")
            return

        self.ik_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, 
            active_links_mask=[False, True, True, True, True, True, False]
        )
        print("IK Chain loaded!")
        
        self.phone.connect()
        self.prev_joints = [0.0] * len(self.ik_chain.links)
        
        # Teleop improvements
        self.base_robot_pos = np.array([0.2, 0.0, 0.15]) # Safe resting pose above table
        self.smoothed_target = np.array([0.2, 0.0, 0.15])
        self.alpha = 0.15 # EMA Smoothing factor (lower = smoother but slightly laggy)

    def run(self):
        print("\n🚀 Teleop stream started! Touch & drag on your phone screen to control the arm.\n")
        while True:
            t0 = time.time()
            self.step()
            elapsed = time.time() - t0
            time.sleep(max(0, 0.02 - elapsed))

    def step(self):
        action = self.phone.get_action()
        if not action or not action.get('phone.enabled', False):
            return 
            
        if action.get('phone.just_enabled', False):
            # Clutching: Freeze the robot's current position to be the new center!
            self.base_robot_pos = self.smoothed_target.copy()
            
        pos = action['phone.pos'] 
        
        # Precision scale factor (1.0 means 10cm on phone = 10cm on robot)
        scale_factor = 1.0 
        
        # Proper Axis Mapping (Phone ARKit -> Robot URDF)
        # ARKit: X=Right, Y=Up, Z=Backwards.   Robot: X=Forward, Y=Left, Z=Up
        robot_offset = np.array([-pos[2], -pos[0], pos[1]]) * scale_factor
        
        raw_target = self.base_robot_pos + robot_offset
        
        # Exponential Moving Average (EMA) to completely eliminate hand jitter
        self.smoothed_target = self.alpha * raw_target + (1.0 - self.alpha) * self.smoothed_target

        ik_joints = self.ik_chain.inverse_kinematics(target_position=self.smoothed_target, initial_position=self.prev_joints)
        self.prev_joints = ik_joints
        
        target_joint_values = [ik_joints[1], ik_joints[2], ik_joints[3], ik_joints[4], ik_joints[5]]
        gripper_val = action['gripper_a3']

        payload = target_joint_values + [gripper_val]
        self.sock.sendto(json.dumps(payload).encode('utf-8'), self.mujoco_bridge_addr)

def main():
    node = PhoneTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        print("Teleop stopped.")

if __name__ == '__main__':
    main()
