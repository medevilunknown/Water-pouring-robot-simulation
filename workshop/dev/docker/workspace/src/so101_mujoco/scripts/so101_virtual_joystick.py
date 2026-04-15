#!/usr/bin/env python3
import os
import time
import json
import socket
import threading
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import ikpy.chain
except ImportError:
    print("ikpy is not installed. Please run: pip install ikpy")
    exit(1)

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SO-101 Virtual PS Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js"></script>
    <style>
        body { margin: 0; padding: 0; background-color: #111; overflow: hidden; color: white; font-family: -apple-system, sans-serif; user-select: none; -webkit-user-select: none; }
        #left_zone { position: absolute; left: 0; top: 0; width: 50%; height: 100%; border-right: 2px solid #333; }
        #right_zone { position: absolute; right: 0; top: 0; width: 50%; height: 100%; }
        #gripper_btn {
            position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
            padding: 15px 40px; font-size: 20px; font-weight: bold;
            background: #ff4444; color: white; border: none; border-radius: 12px;
            box-shadow: 0 4px #aa0000; cursor: pointer; z-index: 100;
        }
        #gripper_btn:active { box-shadow: 0 0 #aa0000; transform: translate(-50%, 4px); }
        .label { position: absolute; top: 50%; width: 100%; text-align: center; color: #555; pointer-events: none; transform: translateY(-50%); font-size: 1.2rem; font-weight: bold; opacity: 0.5;}
    </style>
</head>
<body>
    <button id="gripper_btn">GRIPPER: OPEN</button>
    <div id="left_zone"><div class="label">Forward / Back<br><br>Left / Right</div></div>
    <div id="right_zone"><div class="label">Up / Down</div></div>

    <script>
        // Use the current phone URL to establish WebSocket
        let ws_protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
        let ws = new WebSocket(ws_protocol + window.location.host + "/ws");
        
        let axes = { x: 0.0, y: 0.0, z: 0.0 };
        let gripper_open = true;

        document.getElementById('gripper_btn').addEventListener('touchstart', (e) => {
            e.preventDefault();
            gripper_open = !gripper_open;
            let btn = document.getElementById('gripper_btn');
            btn.innerText = gripper_open ? "GRIPPER: OPEN" : "GRIPPER: CLOSED";
            btn.style.background = gripper_open ? "#ff4444" : "#44ff44";
            btn.style.boxShadow = gripper_open ? "0 4px #aa0000" : "0 4px #00aa00";
        });

        // Left Joystick configs
        let managerLeft = nipplejs.create({
            zone: document.getElementById('left_zone'),
            mode: 'dynamic', color: 'cyan', size: 150
        });

        // Right Joystick configs
        let managerRight = nipplejs.create({
            zone: document.getElementById('right_zone'),
            mode: 'dynamic', color: 'orange', size: 150
        });

        // Left joystick maps to X (Forward/Backward) and Y (Left/Right)
        managerLeft.on('move', (evt, data) => {
            if(data.vector) {
                axes.x = data.vector.y;  // Up vector pushes X forward
                axes.y = -data.vector.x; // Right vector pushes Y negatively (MuJoCo left is positive)
            }
        });
        managerLeft.on('end', () => { axes.x = 0; axes.y = 0; });

        // Right joystick maps to Z (Up/Down)
        managerRight.on('move', (evt, data) => {
            if(data.vector) {
                axes.z = data.vector.y;  // Up vector pushes Z UP
            }
        });
        managerRight.on('end', () => { axes.z = 0; });

        // Stream constantly at ~50Hz
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ 
                    x: axes.x, 
                    y: axes.y, 
                    z: axes.z, 
                    gripper: gripper_open ? 1.74 : 0.0 
                }));
            }
        }, 20); 
    </script>
</body>
</html>
"""

app = FastAPI()

class RobotController:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mujoco_bridge_addr = ("127.0.0.1", 9876)
        
        urdf_path = "src/so101_description/urdf/so101.urdf"
        if not os.path.exists(urdf_path):
            print(f"ERROR: URDF not found at {urdf_path}")
            exit(1)

        self.ik_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path, active_links_mask=[False, True, True, True, True, True, False]
        )
        self.prev_joints = [0.0] * len(self.ik_chain.links)
        
        # Starting position of the gripper
        self.target_pos = np.array([0.20, 0.0, 0.15])
        
        # How fast the robot moves when the stick is fully pushed (meters per tick)
        self.speed = 0.005
        
        self.axes = {"x": 0.0, "y": 0.0, "z": 0.0, "gripper": 1.74}
        self.running = True
        
        self.loop_thread = threading.Thread(target=self._ik_loop, daemon=True)
        self.loop_thread.start()

    def update_axes(self, payload):
        self.axes["x"] = float(payload.get("x", 0.0))
        self.axes["y"] = float(payload.get("y", 0.0))
        self.axes["z"] = float(payload.get("z", 0.0))
        self.axes["gripper"] = float(payload.get("gripper", 1.74))

    def _ik_loop(self):
        print("IK Physics Loop Started...")
        while self.running:
            # Velocity control: Add joystick vectors to the robot's current position
            self.target_pos[0] += self.axes["x"] * self.speed
            self.target_pos[1] += self.axes["y"] * self.speed
            self.target_pos[2] += self.axes["z"] * self.speed
            
            # Simple boundary box to prevent the arm from flying into infinity
            self.target_pos[0] = np.clip(self.target_pos[0], 0.05, 0.40) # Fwd/Back limits
            self.target_pos[1] = np.clip(self.target_pos[1], -0.35, 0.35) # Left/Right limits
            self.target_pos[2] = np.clip(self.target_pos[2], -0.05, 0.45) # Up/Down limits

            ik_joints = self.ik_chain.inverse_kinematics(target_position=self.target_pos, initial_position=self.prev_joints)
            self.prev_joints = ik_joints
            
            target_joint_values = [ik_joints[1], ik_joints[2], ik_joints[3], ik_joints[4], ik_joints[5]]
            
            msg = target_joint_values + [self.axes["gripper"]]
            self.sock.sendto(json.dumps(msg).encode('utf-8'), self.mujoco_bridge_addr)
            
            time.sleep(0.02) # 50hz physics update

robot = RobotController()

@app.get("/")
def get_home():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📲 Mobile Phone Connected to Virtual Joystick!")
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            robot.update_axes(cmd)
    except WebSocketDisconnect:
        print("📲 Mobile Phone Disconnected! Pausing robot.")
        # Auto-stop robot if phone screen turns off
        robot.update_axes({"x": 0.0, "y": 0.0, "z": 0.0, "gripper": robot.axes["gripper"]})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("🎮 VIRTUAL PLAYSTATION CONTROLLER LAUNCHED! 🎮")
    print("Open this exact link on your phone:")
    print("http://YOUR_LAPTOP_IP:8000")
    print("\n(e.g., http://192.168.1.5:8000)")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
