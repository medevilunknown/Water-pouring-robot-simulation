#!/usr/bin/python3
"""
SO-101 Simulation Live Dashboard
==================================
Self-contained Flask + ROS 2 web dashboard.
Shows live water state, joint positions, agent commands, and detections.
Auto-refreshes at 2Hz via fetch().
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from flask import Flask, jsonify

app = Flask(__name__)

# ── Shared state (thread-safe enough for reads) ──────────────
state = {
    "water": {"bottle_ml": 120.0, "beaker_ml": 0.0, "beaker_capacity_ml": 150.0,
              "beaker_fill_pct": 0.0, "total_poured_ml": 0.0,
              "is_pouring": False, "is_overspilling": False},
    "joints": {},
    "detections": [],
    "commands": [],
    "episode": 0,
    "agent_status": "Waiting for first episode...",
    "connected": True,
}


# ── HTML served inline ───────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SO-101 Pouring Simulation — Live</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#060612;color:#d0d0e8;min-height:100vh}
.hdr{background:linear-gradient(135deg,#10103a 0%,#08081e 100%);padding:18px 28px;border-bottom:1px solid #1e1e50;display:flex;align-items:center;gap:14px}
.hdr h1{font-size:22px;font-weight:700;background:linear-gradient(90deg,#00d4ff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.live{width:9px;height:9px;border-radius:50%;background:#22c55e;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 6px #22c55e}50%{opacity:.4;box-shadow:0 0 2px #22c55e}}
.hdr .tag{margin-left:auto;padding:5px 14px;border-radius:16px;font-size:12px;font-weight:600;letter-spacing:.5px}
.tag.ok{background:#22c55e18;color:#4ade80;border:1px solid #22c55e30}
.tag.warn{background:#f59e0b18;color:#fbbf24;border:1px solid #f59e0b30}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:22px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:linear-gradient(160deg,#0d0d28 0%,#09091c 100%);border:1px solid #1c1c48;border-radius:14px;padding:20px;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#7c3aed40,transparent)}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#7c3aed;margin-bottom:14px;font-weight:600}
.row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #12122e}
.row:last-child{border-bottom:none}
.row .lbl{color:#6b6b9e;font-size:13px}
.row .val{font-size:17px;font-weight:600;font-family:'JetBrains Mono',monospace}
.val.cyan{color:#22d3ee}.val.green{color:#4ade80}.val.amber{color:#fbbf24}.val.red{color:#f87171}
.bar-wrap{width:100%;height:10px;background:#12122e;border-radius:5px;margin-top:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;transition:width .4s ease;background:linear-gradient(90deg,#06b6d4,#7c3aed)}
.bar-fill.hot{background:linear-gradient(90deg,#f59e0b,#ef4444)}
.jbar{display:flex;align-items:center;gap:8px;margin:5px 0}
.jbar .nm{width:95px;font-size:11px;color:#6b6b9e;font-family:'JetBrains Mono',monospace}
.jbar .bg{flex:1;height:7px;background:#12122e;border-radius:4px;overflow:hidden}
.jbar .fl{height:100%;border-radius:4px;background:linear-gradient(90deg,#06b6d4,#7c3aed);transition:width .3s ease}
.jbar .vl{width:55px;text-align:right;font-size:11px;color:#8b8bb8;font-family:'JetBrains Mono',monospace}
.wide{grid-column:span 2}
@media(max-width:900px){.wide{grid-column:span 1}}
.log{max-height:320px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:12px}
.log::-webkit-scrollbar{width:4px}.log::-webkit-scrollbar-thumb{background:#2a2a5a;border-radius:2px}
.le{padding:6px 10px;margin:3px 0;border-radius:6px;background:#08081a;border-left:3px solid #7c3aed;line-height:1.4}
.le .t{color:#4b4b7a;font-size:10px}.le .c{color:#22d3ee}
.det-item{padding:8px 12px;margin:4px 0;border-radius:8px;background:#0a0a1e;border:1px solid #16163a}
.det-item .nm{color:#a78bfa;font-weight:600;font-size:13px}.det-item .pos{color:#6b6b9e;font-size:11px;margin-top:2px}
.arm-vis{text-align:center;padding:10px 0}
.arm-svg{width:180px;height:220px}
.ep-badge{display:inline-block;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;background:#7c3aed20;color:#a78bfa;border:1px solid #7c3aed30;margin-bottom:10px}
</style>
</head>
<body>
<div class="hdr">
 <span class="live"></span>
 <h1>SO-101 Pouring Simulation</h1>
 <span class="tag ok" id="conn-tag">● CONNECTED</span>
</div>
<div class="grid">

 <!-- Robot Arm SVG -->
 <div class="card">
  <h2>🤖 Robot Arm</h2>
  <div class="arm-vis">
   <svg class="arm-svg" viewBox="0 0 180 220" xmlns="http://www.w3.org/2000/svg">
    <!-- Base -->
    <rect x="55" y="190" width="70" height="25" rx="4" fill="#1e1e50" stroke="#7c3aed" stroke-width="1.5"/>
    <text x="90" y="207" text-anchor="middle" fill="#6b6b9e" font-size="8" font-family="Inter">BASE</text>
    <!-- Shoulder -->
    <line x1="90" y1="190" x2="90" y2="155" stroke="#7c3aed" stroke-width="3" stroke-linecap="round"/>
    <circle cx="90" cy="155" r="6" fill="#0d0d28" stroke="#22d3ee" stroke-width="2"/>
    <!-- Upper Arm -->
    <line id="upper-arm" x1="90" y1="155" x2="70" y2="110" stroke="#22d3ee" stroke-width="3" stroke-linecap="round"/>
    <circle cx="70" cy="110" r="5" fill="#0d0d28" stroke="#a78bfa" stroke-width="2"/>
    <!-- Forearm -->
    <line id="forearm" x1="70" y1="110" x2="90" y2="70" stroke="#a78bfa" stroke-width="3" stroke-linecap="round"/>
    <circle cx="90" cy="70" r="5" fill="#0d0d28" stroke="#f472b6" stroke-width="2"/>
    <!-- Wrist -->
    <line id="wrist" x1="90" y1="70" x2="95" y2="45" stroke="#f472b6" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Gripper -->
    <line id="grip-l" x1="85" y1="45" x2="78" y2="30" stroke="#4ade80" stroke-width="2" stroke-linecap="round"/>
    <line id="grip-r" x1="105" y1="45" x2="112" y2="30" stroke="#4ade80" stroke-width="2" stroke-linecap="round"/>
    <circle cx="95" cy="45" r="4" fill="#0d0d28" stroke="#4ade80" stroke-width="1.5"/>
    <!-- Labels -->
    <text x="20" y="160" fill="#3b3b6e" font-size="7" font-family="Inter">shoulder</text>
    <text x="25" y="112" fill="#3b3b6e" font-size="7" font-family="Inter">elbow</text>
    <text x="105" y="68" fill="#3b3b6e" font-size="7" font-family="Inter">wrist</text>
    <text x="115" y="38" fill="#3b3b6e" font-size="7" font-family="Inter">gripper</text>
   </svg>
  </div>
  <div style="text-align:center">
   <span class="ep-badge" id="ep-badge">EPISODE --</span>
  </div>
  <div class="row"><span class="lbl">Status</span><span class="val cyan" id="agent-status">Initializing</span></div>
 </div>

 <!-- Water State -->
 <div class="card">
  <h2>💧 Water Simulation</h2>
  <div class="row"><span class="lbl">Bottle Remaining</span><span class="val cyan" id="bottle-ml">120.0 ml</span></div>
  <div class="row"><span class="lbl">Beaker Level</span><span class="val green" id="beaker-ml">0.0 ml</span></div>
  <div class="row"><span class="lbl">Fill Percentage</span><span class="val" id="fill-pct">0.0%</span></div>
  <div class="bar-wrap"><div class="bar-fill" id="fill-bar" style="width:0%"></div></div>
  <div style="height:12px"></div>
  <div class="row"><span class="lbl">Total Poured</span><span class="val cyan" id="total-poured">0.0 ml</span></div>
  <div class="row"><span class="lbl">Pouring Now?</span><span class="val" id="pouring">No</span></div>
  <div class="row"><span class="lbl">Overspill?</span><span class="val green" id="overspill">No</span></div>
 </div>

 <!-- Detections -->
 <div class="card">
  <h2>🎯 3D Object Detections</h2>
  <div class="row"><span class="lbl">Objects Found</span><span class="val cyan" id="det-count">0</span></div>
  <div id="det-list" style="margin-top:8px"></div>
 </div>

 <!-- Joints -->
 <div class="card">
  <h2>🦾 Joint Positions (rad)</h2>
  <div id="joints"></div>
 </div>

 <!-- Command Log -->
 <div class="card wide">
  <h2>📋 Agent Command Log</h2>
  <div class="log" id="cmd-log">
   <div class="le"><span class="t">--:--:--</span> <span class="c">Waiting for first episode...</span></div>
  </div>
 </div>
</div>

<script>
const JOINT_NAMES=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];
let prevCmdLen=0;
function up(){
 fetch('/api/state').then(r=>r.json()).then(d=>{
  // Water
  const w=d.water||{};
  el('bottle-ml').textContent=(w.bottle_ml??120).toFixed(1)+' ml';
  el('beaker-ml').textContent=(w.beaker_ml??0).toFixed(1)+' ml';
  const pct=w.beaker_fill_pct??0;
  el('fill-pct').textContent=pct.toFixed(1)+'%';
  el('fill-pct').className='val '+(pct>90?'red':pct>50?'amber':'cyan');
  const bar=el('fill-bar');
  bar.style.width=Math.min(pct,100)+'%';
  bar.className='bar-fill'+(pct>85?' hot':'');
  el('total-poured').textContent=(w.total_poured_ml??0).toFixed(1)+' ml';
  el('pouring').textContent=w.is_pouring?'🌊 YES':'No';
  el('pouring').className='val '+(w.is_pouring?'cyan':'');
  el('overspill').textContent=w.is_overspilling?'⚠️ OVERFLOW':'Safe';
  el('overspill').className='val '+(w.is_overspilling?'red':'green');

  // Detections
  const dets=d.detections||[];
  el('det-count').textContent=dets.length;
  el('det-list').innerHTML=dets.map(x=>{
   const p=x.pose||x.world_position||[];
   return `<div class="det-item"><div class="nm">${x.name||'?'}</div><div class="pos">${x.shape||'obj'} — r=${(x.radius||0).toFixed(3)} h=${(x.height||0).toFixed(3)}<br>pos: [${p.slice(0,3).map(v=>v.toFixed(3)).join(', ')}]</div></div>`;
  }).join('');

  // Joints
  let jh='';
  JOINT_NAMES.forEach(n=>{
   const v=(d.joints||{})[n]||0;
   const p=((v+3.14)/6.28)*100;
   jh+=`<div class="jbar"><span class="nm">${n}</span><div class="bg"><div class="fl" style="width:${Math.max(0,Math.min(100,p))}%"></div></div><span class="vl">${v.toFixed(4)}</span></div>`;
  });
  el('joints').innerHTML=jh;

  // Commands
  const cmds=d.commands||[];
  if(cmds.length!==prevCmdLen){
   prevCmdLen=cmds.length;
   const lg=el('cmd-log');
   lg.innerHTML=cmds.slice(-30).map(c=>
    `<div class="le"><span class="t">${c.time}</span> <span class="c">${c.cmd}</span></div>`
   ).join('');
   lg.scrollTop=lg.scrollHeight;
  }

  // Episode & Status
  el('ep-badge').textContent='EPISODE '+(d.episode||'--');
  el('agent-status').textContent=d.agent_status||'Running';

  // Connection tag
  el('conn-tag').textContent='● CONNECTED';
  el('conn-tag').className='tag ok';
 }).catch(()=>{
  el('conn-tag').textContent='● DISCONNECTED';
  el('conn-tag').className='tag warn';
 });
}
function el(id){return document.getElementById(id)}
setInterval(up,500);
up();
</script>
</body>
</html>"""


class DashboardNode(Node):
    def __init__(self):
        super().__init__('web_dashboard')
        self.create_subscription(String, '/water_state', self._water_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(String, '/detected_objects_3d', self._det_cb, 10)
        self.create_subscription(String, '/agent/command', self._cmd_cb, 10)
        self.get_logger().info('Web Dashboard ROS subscriptions active')

    def _water_cb(self, msg):
        try:
            state["water"] = json.loads(msg.data)
        except Exception:
            pass

    def _joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                state["joints"][name] = msg.position[i]

    def _det_cb(self, msg):
        try:
            state["detections"] = json.loads(msg.data)
        except Exception:
            pass

    def _cmd_cb(self, msg):
        cmd_text = msg.data.strip()
        state["commands"].append({
            "time": time.strftime("%H:%M:%S"),
            "cmd": cmd_text
        })
        if len(state["commands"]) > 100:
            state["commands"] = state["commands"][-100:]

        # Track episode from RESET commands
        if 'EPISODE' in cmd_text.upper() or 'RESET' in cmd_text.upper():
            state["episode"] = state.get("episode", 0) + 1
            state["agent_status"] = f"Running Episode {state['episode']}"
        elif 'MOVE_ABOVE' in cmd_text.upper():
            state["agent_status"] = "Approaching bottle"
        elif 'CLOSE_GRIPPER' in cmd_text.upper():
            state["agent_status"] = "Grasping bottle"
        elif 'LIFT' in cmd_text.upper():
            state["agent_status"] = "Lifting bottle"
        elif 'MOVE_TO_BEAKER' in cmd_text.upper():
            state["agent_status"] = "Moving to beaker"
        elif 'TILT' in cmd_text.upper():
            state["agent_status"] = "🌊 POURING"
        elif 'STOP_POUR' in cmd_text.upper():
            state["agent_status"] = "Stopping pour"
        elif 'RETURN_HOME' in cmd_text.upper():
            state["agent_status"] = "Returning home"
        elif 'DONE' in cmd_text.upper():
            state["agent_status"] = "Episode complete ✓"


@app.route('/')
def index():
    return DASHBOARD_HTML

@app.route('/api/state')
def api_state():
    return jsonify(state)


def run_flask():
    app.run(host='0.0.0.0', port=5555, debug=False, use_reloader=False)


def main():
    rclpy.init()
    node = DashboardNode()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    node.get_logger().info('Dashboard serving on http://0.0.0.0:5555')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
