#!/usr/bin/env python3
"""
LLM Pour Agent Node (Self-Adapting RL Loop)
===========================================
Subscribes to vision detections and water state, queries Llama 3 for the next action.
Evaluates its performance at the end of each episode and generates self-improvement rules.
"""

import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
BEAKER_CAPACITY_ML = 150.0
TARGET_FILL_ML = 100.0
BOTTLE_WATER_ML = 120.0
MAX_STEPS = 30
MAX_EPISODES = 5
KNOWLEDGE_FILE = "gazebo_knowledge.json"

# ── Memory ───────────────────────────────────────────────────
def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_knowledge(reflections):
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(reflections, f, indent=2)

# ── LLM System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot arm controller for a SO-101 6-DOF robotic arm in Gazebo.

Task: Pick up a water bottle, move it above a beaker, pour water to ~100ml without overspilling (cap: 150ml).
You receive JSON state. Output EXACTLY ONE action.

Available actions:
  MOVE_ABOVE_BOTTLE
  LOWER_TO_BOTTLE
  CLOSE_GRIPPER
  LIFT_BOTTLE
  MOVE_TO_BEAKER
  TILT_POUR <angle>
  STOP_POUR
  RETURN_HOME
  DONE

Tips: Pick up bottle sequence → MOVE_TO_BEAKER → TILT_POUR 45 to start → watch beaker_ml → STOP_POUR when hitting ~90ml.
CRITICAL: Output ONLY the action, nothing else."""


class LlmPourAgentNode(Node):
    def __init__(self):
        super().__init__('llm_pour_agent_node')
        self.get_logger().info('Initialising LLM RL Pour Agent...')

        self.declare_parameter('groq_api_key', '')
        self.declare_parameter('agent_rate_hz', 0.5)
        self.declare_parameter('auto_mode', True)

        api_key = self.get_parameter('groq_api_key').value
        if not api_key:
            api_key = os.environ.get('GROQ_API_KEY', '')
        agent_rate = self.get_parameter('agent_rate_hz').value
        self.auto_mode = self.get_parameter('auto_mode').value

        self.client = None
        if api_key and GROQ_AVAILABLE:
            self.client = Groq(api_key=api_key)
            self.get_logger().info(f'Groq client initialised (model: {GROQ_MODEL}) ✓')
        else:
            self.get_logger().warn('LLM unavailable — Agent will stall.')

        # ── RL State ─────────────────────────────────────────────
        self.episode = 1
        self.knowledge = load_knowledge()
        self.get_logger().info(f'Loaded {len(self.knowledge)} past lessons from memory.')

        # ── Episode State ─────────────────────────────────────────────
        self._reset_episode_state()
        
        self.latest_detections = []
        self.water_state = {
            'bottle_ml': BOTTLE_WATER_ML,
            'beaker_ml': 0.0,
            'beaker_capacity_ml': BEAKER_CAPACITY_ML,
            'beaker_fill_pct': 0.0,
            'is_pouring': False,
            'is_overspilling': False,
        }
        self.action_acknowledged = True

        self.create_subscription(String, '/detected_objects', self._detections_cb, 10)
        self.create_subscription(String, '/water_state', self._water_cb, 10)
        self.create_subscription(String, '/agent/action_ack', self._ack_cb, 10)
        self.cmd_pub = self.create_publisher(String, '/agent/command', 10)

        period = 1.0 / agent_rate
        self.create_timer(period, self._decision_loop)
        self.get_logger().info(f'Agent loop at {agent_rate} Hz — ready Episode 1')

    def _reset_episode_state(self):
        self.phase = 'start'
        self.step = 0
        self.conversation_history = []
        self.gripper_open = True
        self.task_done = False
        self.last_action = None

    def _detections_cb(self, msg: String):
        try: self.latest_detections = json.loads(msg.data)
        except json.JSONDecodeError: pass

    def _water_cb(self, msg: String):
        try: self.water_state = json.loads(msg.data)
        except json.JSONDecodeError: pass

    def _ack_cb(self, msg: String):
        self.action_acknowledged = True
        ack_data = msg.data
        self.get_logger().info(f'Action acknowledged: {ack_data}')
        
        # If we just finished resetting the physical episode, increment and restart!
        try:
            ack_json = json.loads(ack_data)
            if ack_json.get('command') == 'RESET_EPISODE':
                self.episode += 1
                if self.episode > MAX_EPISODES:
                    self.get_logger().info('🏁 ALL EPISODES COMPLETED! SHUTTING DOWN AGENT.')
                    self.task_done = True
                else:
                    self.get_logger().info(f'\n\n{"="*60}\n🎬 STARTING EPISODE {self.episode}/{MAX_EPISODES} 🎬\n{"="*60}')
                    self._reset_episode_state()
        except:
            pass

    def _build_observation(self) -> str:
        objects = {}
        for det in self.latest_detections:
            name = det.get('name', det.get('class', 'unknown'))
            pos = det.get('world_position', [0, 0, 0])
            objects[name] = {'position': pos, 'confidence': det.get('confidence', 0)}
        obs = {
            'phase': self.phase, 'step': self.step,
            'gripper': 'open' if self.gripper_open else 'closed',
            'detected_objects': objects,
            'water': {
                'bottle_ml': round(self.water_state.get('bottle_ml', BOTTLE_WATER_ML), 1),
                'beaker_ml': round(self.water_state.get('beaker_ml', 0.0), 1),
                'beaker_capacity_ml': BEAKER_CAPACITY_ML,
                'target_fill_ml': TARGET_FILL_ML,
                'is_overspilling': self.water_state.get('is_overspilling', False),
            },
        }
        return json.dumps(obs, indent=2)

    def _query_llm(self, observation: str) -> str:
        self.conversation_history.append({
            'role': 'user', 'content': f'Current state:\n{observation}\n\nWhat action should I take?'
        })
        appended_prompt = SYSTEM_PROMPT
        if self.knowledge:
            appended_prompt += "\n\nPAST LESSONS (Apply strictly!):\n" + "\n".join(f"- {r}" for r in self.knowledge)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{'role': 'system', 'content': appended_prompt}] + self.conversation_history[-10:],
                temperature=0.1, max_tokens=50,
            )
            action = response.choices[0].message.content.strip().split('\n')[0]
            self.conversation_history.append({'role': 'assistant', 'content': action})
            return action
        except Exception as e:
            self.get_logger().error(f'LLM query failed: {e}')
            return None

    def _evaluate_and_reflect(self, beaker_ml):
        score = "FAILED"
        if 90.0 <= beaker_ml <= 115.0: score = "SUCCESS (Safe State)"
        elif beaker_ml > 115.0: score = "FAILED (Overspill)"
        elif beaker_ml < 90.0: score = "FAILED (Undertarget)"

        self.get_logger().info(f"\n📊 EPISODE {self.episode} RESULT: {score} | End Volume: {beaker_ml:.1f}ml")
        self.get_logger().info("🧠 Generating Reflection...")
        
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history if msg['role'] == "assistant"])
        prompt = f"""You completed an episode of pouring simulation.
Target: 100ml. You poured: {beaker_ml:.1f}ml. Score: {score}.
Action sequence:
{history_text}
Provide ONE concise rule (max 2 sentences) to improve performance next time. Focus on when to STOP pouring. Respond ONLY with the rule."""

        try:
            res = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": "You formulate robot self-improvement rules."}, {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=60,
            )
            rule = res.choices[0].message.content.strip()
            self.get_logger().info(f"💡 New Knowledge: {rule}")
            self.knowledge.append(rule)
            save_knowledge(self.knowledge)
        except Exception as e:
            self.get_logger().error(f"Reflection error: {e}")

    @staticmethod
    def _parse_action(action_str: str):
        if not action_str: return None, None
        parts = action_str.strip().split()
        name = parts[0].upper()
        two_word = {'MOVE_ABOVE', 'MOVE_TO', 'LOWER_TO', 'CLOSE_GRIPPER', 'LIFT_BOTTLE', 'STOP_POUR', 'RETURN_HOME', 'TILT_POUR'}
        three_word = {'MOVE_ABOVE_BOTTLE', 'LOWER_TO_BOTTLE', 'MOVE_TO_BEAKER'}
        if len(parts) >= 3:
            candidate = '_'.join(p.upper() for p in parts[:3])
            if candidate in three_word: return candidate, parts[3] if len(parts) > 3 else None
        if len(parts) >= 2:
            candidate = '_'.join(p.upper() for p in parts[:2])
            if candidate in two_word: return candidate, parts[2] if len(parts) > 2 else None
        return name, parts[1] if len(parts) > 1 else None

    def _update_phase(self, action_name: str):
        phase_map = {
            'MOVE_ABOVE_BOTTLE': 'approaching', 'LOWER_TO_BOTTLE': 'at_bottle', 'CLOSE_GRIPPER': 'grasped',
            'LIFT_BOTTLE': 'lifted', 'MOVE_TO_BEAKER': 'above_beaker', 'TILT_POUR': 'pouring',
            'STOP_POUR': 'stopped_pour', 'RETURN_HOME': 'returning', 'DONE': 'done', 'RESET_EPISODE': 'resetting'
        }
        if action_name in phase_map: self.phase = phase_map[action_name]
        if action_name in ('CLOSE_GRIPPER', 'CLOSE'): self.gripper_open = False
        elif action_name in ('OPEN_GRIPPER', 'OPEN'): self.gripper_open = True

    def _publish_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.action_acknowledged = False
        self.last_action = cmd

    def _decision_loop(self):
        if self.task_done or not self.action_acknowledged or self.phase == 'resetting':
            return

        self.step += 1
        beaker_ml = self.water_state.get('beaker_ml', 0.0)

        if self.step > MAX_STEPS:
            self.get_logger().warn('Max steps reached — forcing episode end')
            self._evaluate_and_reflect(beaker_ml)
            self._publish_command('RESET_EPISODE')
            self._update_phase('RESET_EPISODE')
            return

        self.get_logger().info(f'\n{"─"*50}')
        self.get_logger().info(f'Ep {self.episode} | Step {self.step}/{MAX_STEPS} | Phase: {self.phase}')

        obs = self._build_observation()

        if beaker_ml >= 140.0 and self.phase == 'pouring':
            self.get_logger().warn('🛑 SAFETY: Beaker near capacity!')
            self._publish_command('STOP_POUR')
            self._update_phase('STOP_POUR')
            return

        action_str = self._query_llm(obs) if self.client else 'DONE'
        if not action_str: return

        action_name, param = self._parse_action(action_str)
        if not action_name: return

        param_str = f' {param}' if param else ''
        self.get_logger().info(f'🎯 Action: {action_name}{param_str}')

        if action_name == 'DONE':
            self._evaluate_and_reflect(beaker_ml)
            self._publish_command('RESET_EPISODE')
            self._update_phase('RESET_EPISODE')
        else:
            cmd = action_name + param_str
            self._publish_command(cmd)
            self._update_phase(action_name)

def main(args=None):
    rclpy.init(args=args)
    node = LlmPourAgentNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
