#!/usr/bin/env python3
import time
import json
import socket
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SO-101 Button Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <style>
        body { margin: 0; padding: 20px; background-color: #111; color: white; font-family: -apple-system, sans-serif; user-select: none; -webkit-user-select: none; text-align: center; overflow: hidden; touch-action: none;}
        .row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #222; padding: 15px; border-radius: 12px;}
        .title { font-size: 16px; font-weight: bold; width: 35%; text-align: left; }
        .btn-group { display: flex; gap: 10px; width: 65%; justify-content: flex-end;}
        .btn { 
            padding: 18px 10px; font-size: 18px; font-weight: bold; 
            background: #444; color: white; border: none; border-radius: 8px; 
            box-shadow: 0 4px #222; cursor: pointer; flex: 1; touch-action: manipulation;
        }
        .btn:active { background: #666; box-shadow: 0 0 #222; transform: translateY(4px); }
    </style>
</head>
<body>
    <h2 style="margin-bottom:25px;">Robot Joint Controls</h2>
    <div id="controls"></div>

    <script>
        let ws_protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
        let ws = new WebSocket(ws_protocol + window.location.host + "/ws");
        
        const joints = [
            {id: 0, name: "Shoulder Pan", neg: "Left", pos: "Right"},
            {id: 1, name: "Shoulder Lift", neg: "Down", pos: "Up"},
            {id: 2, name: "Elbow Flex", neg: "Down", pos: "Up"},
            {id: 3, name: "Wrist Flex", neg: "Down", pos: "Up"},
            {id: 4, name: "Wrist Roll", neg: "CCW", pos: "CW"},
            {id: 5, name: "Gripper", neg: "Close", pos: "Open"}
        ];

        let activeCommand = { id: -1, dir: 0 };

        function render() {
            let container = document.getElementById("controls");
            joints.forEach(j => {
                let row = document.createElement("div");
                row.className = "row";
                row.innerHTML = `
                    <div class="title">${j.name}</div>
                    <div class="btn-group">
                        <button class="btn" ontouchstart="startMove(${j.id}, -1)" onmousedown="startMove(${j.id}, -1)" ontouchend="stopMove()" onmouseup="stopMove()" onmouseleave="stopMove()">${j.neg}</button>
                        <button class="btn" ontouchstart="startMove(${j.id}, 1)" onmousedown="startMove(${j.id}, 1)" ontouchend="stopMove()" onmouseup="stopMove()" onmouseleave="stopMove()">${j.pos}</button>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        function startMove(id, dir) { 
            activeCommand = { id: id, dir: dir }; 
            // Vibrate phone slightly for physical feedback if supported
            if (navigator.vibrate) navigator.vibrate(15);
        }
        function stopMove() { activeCommand = { id: -1, dir: 0 }; }

        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN && activeCommand.id !== -1) {
                ws.send(JSON.stringify(activeCommand));
            }
        }, 30); // 33Hz updates

        window.onload = render;
        
        // Prevent accidental scrolling while touching buttons
        document.body.addEventListener('touchmove', function(e) { e.preventDefault(); }, { passive: false });
    </script>
</body>
</html>
"""

app = FastAPI()

class RobotController:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mujoco_bridge_addr = ("127.0.0.1", 9876)
        
        # Absolute robot joint angles
        # ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        self.joints = [0.0, -0.5, 0.5, 0.0, 0.0, 1.74]
        self.speed = 0.015
        
        self.active_id = -1
        self.active_dir = 0
        self.running = True
        
        self.loop_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.loop_thread.start()

    def update_command(self, payload):
        self.active_id = int(payload.get("id", -1))
        self.active_dir = float(payload.get("dir", 0))

    def _physics_loop(self):
        print("Joint Loop Started...")
        while self.running:
            if self.active_id != -1:
                # Modifiers for different joint sensitivities
                mod = 2.0 if self.active_id == 5 else 1.0
                self.joints[self.active_id] += self.active_dir * (self.speed * mod)
                
                # Simple boundary boxes
                self.joints[5] = max(0.0, min(1.74, self.joints[5])) # gripper fully open = 1.74
                
                # Transmit over our zero-latency UDP backbone
                self.sock.sendto(json.dumps(self.joints).encode('utf-8'), self.mujoco_bridge_addr)
            
            time.sleep(0.02) # 50hz

robot = RobotController()

@app.get("/")
def get_home():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📲 Mobile Phone Connected to Button Controller!")
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            robot.update_command(cmd)
    except WebSocketDisconnect:
        print("📲 Mobile Phone Disconnected!")
        robot.update_command({"id": -1, "dir": 0})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("🔘 INDIVIDUAL JOINT BUTTON CONTROLLER LAUNCHED! 🔘")
    print("Open this exact link on your phone (or laptop!):")
    print("http://YOUR_LAPTOP_IP:8000")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
