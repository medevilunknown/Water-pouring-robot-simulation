#!/usr/bin/python3
"""
Self-Training Agent Node (Robust Edition)
==========================================
Orchestrates the episode loop. Works without LLM by using scripted fallback.
Resets Gazebo, triggers motion execution, evaluates, saves safe states, and repeats.
If Ollama is available, uses it for diagnosis and tuning.
"""

import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

try:
    from local_llm_client import LocalLLMClient
    LLM_IMPORT_OK = True
except ImportError:
    LLM_IMPORT_OK = False

try:
    from episode_logger import EpisodeLogger
    LOGGER_IMPORT_OK = True
except ImportError:
    LOGGER_IMPORT_OK = False


class SelfTrainingAgent(Node):
    def __init__(self):
        super().__init__('self_training_agent')
        self.get_logger().info('Initializing Self-Training Agent (Robust Edition)...')

        # LLM client (optional)
        self.llm = None
        if LLM_IMPORT_OK:
            try:
                self.llm = LocalLLMClient(model='llama3.2')
                self.get_logger().info('LLM client loaded (may use scripted fallback if Ollama is unavailable)')
            except Exception as e:
                self.get_logger().warn(f'LLM client init failed: {e}')

        # Episode logger
        self.logger = None
        if LOGGER_IMPORT_OK:
            self.logger = EpisodeLogger('/tmp/pour_agent_logs')
            self.get_logger().info('Episode logger ready at /tmp/pour_agent_logs')

        self.max_episodes = 50
        self.current_episode = 0
        self.in_episode = False
        self.waiting_for_ack = False
        self.current_cmd_index = 0

        self.safe_state_config = None

        self.current_config = {
            "approach_offset_z": 0.10,
            "grip_position": -0.10,
            "pour_tilt_deg": 45.0,
            "pour_duration_sec": 3.0
        }

        self.detections = []
        self.water_state = {}

        # Subscriptions
        self.create_subscription(String, '/detected_objects_3d', self._det_cb, 10)
        self.create_subscription(String, '/water_state', self._water_cb, 10)
        self.create_subscription(String, '/agent/action_ack', self._ack_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(String, '/agent/command', 10)
        self.config_pub = self.create_publisher(String, '/agent/config', 10)

        # Startup delay: wait 10s for all other nodes to be ready
        self.get_logger().info('Waiting 10s for simulation to settle...')
        self.startup_timer = self.create_timer(10.0, self._start_first_episode)

    def _det_cb(self, msg):
        try:
            self.detections = json.loads(msg.data)
        except Exception:
            pass

    def _water_cb(self, msg):
        try:
            self.water_state = json.loads(msg.data)
        except Exception:
            pass

    def _ack_cb(self, msg):
        try:
            ack = json.loads(msg.data)
            cmd = ack.get('command', '')
            success = ack.get('success', False)
            self.get_logger().info(f'ACK received: {cmd} -> {"OK" if success else "FAIL"}')
        except Exception:
            pass

    def _start_first_episode(self):
        """Called once after startup delay."""
        self.startup_timer.cancel()
        self.get_logger().info('Startup delay complete. Beginning episode loop.')
        # Run episode loop in a daemon thread so time.sleep doesn't block ROS callbacks
        self._episode_thread = threading.Thread(target=self._episode_thread_loop, daemon=True)
        self._episode_thread.start()

    def _episode_thread_loop(self):
        """Background thread that runs episodes sequentially."""
        while rclpy.ok() and self.current_episode < self.max_episodes:
            self._run_episode_loop()
            time.sleep(5.0)  # Wait between episodes

    def _run_episode_loop(self):
        if self.in_episode or self.current_episode >= self.max_episodes:
            return

        self.in_episode = True
        self.current_episode += 1

        sep = '=' * 50
        self.get_logger().info(f'\n{sep}\nSTARTING EPISODE {self.current_episode}\n{sep}')

        # 1. Reset
        self._reset_simulation()

        # Publish current config to executor
        cfg_msg = String()
        cfg_msg.data = json.dumps(self.current_config)
        self.config_pub.publish(cfg_msg)
        self.get_logger().info(f'Published config: {self.current_config}')

        if self.logger:
            self.logger.start_episode(self.current_episode, self.current_config)

        # 2. Check detections (use mock if none received yet)
        if not self.detections:
            self.get_logger().warn('No detections from vision yet — using default object positions.')
            self.detections = [
                {"name": "water_bottle", "shape": "cylinder", "radius": 0.031, "height": 0.128,
                 "pose": [-0.27, -0.10, 0.839, 0, 0, 0]},
                {"name": "target_beaker", "shape": "cylinder", "radius": 0.030, "height": 0.070,
                 "pose": [-0.30, 0.12, 0.810, 0, 0, 0]}
            ]
            # Publish these defaults to the dashboard/vision topic so the UI shows them
            det_pub_tmp = self.create_publisher(String, '/detected_objects_3d', 10)
            det_msg = String()
            det_msg.data = json.dumps(self.detections)
            det_pub_tmp.publish(det_msg)

        if self.logger:
            self.logger.log_detection("all", self.detections)

        self.get_logger().info(f'Detections: {len(self.detections)} objects')
        for d in self.detections:
            self.get_logger().info(f'  - {d.get("name", "?")} at {d.get("pose", "?")}')

        # 3. Plan (skip LLM if unavailable — always PROCEED in scripted mode)
        plan = 'PROCEED'
        if self.llm:
            try:
                plan = self.llm.plan_grasp(self.detections, self.current_config)
                self.get_logger().info(f'LLM plan decision: {plan}')
            except Exception as e:
                self.get_logger().warn(f'LLM planning failed ({e}), proceeding with scripted plan.')
                plan = 'PROCEED'

        if plan == 'ABORT':
            self.get_logger().warn('Plan aborted.')
            if self.logger:
                self.logger.log_failure('planning', 'llm_abort')
            self.in_episode = False
            return

        # 4. Execute sequence
        self.get_logger().info('Executing pouring sequence...')
        result_status = self._execute_sequence()

        # 5. Evaluate
        water_poured = self.water_state.get('beaker_ml', 0.0)
        beaker_pct = self.water_state.get('beaker_fill_pct', 0.0)

        if result_status == "SUCCESS" and water_poured > 20.0:
            self.get_logger().info(
                f'EPISODE {self.current_episode} SUCCESS! '
                f'Poured {water_poured:.1f}ml ({beaker_pct:.1f}% full)')
            if self.logger:
                self.logger.log_success(water_poured)
            if self.safe_state_config is None:
                self.safe_state_config = self.current_config.copy()
                self.get_logger().info('Saved SAFE STATE config.')
        else:
            self.get_logger().warn(
                f'EPISODE {self.current_episode} FAILED. '
                f'Water: {water_poured:.1f}ml, Status: {result_status}')
            if self.logger:
                self.logger.log_failure('execution', result_status)

            # Diagnose via LLM if available
            if self.llm and self.logger:
                try:
                    self.get_logger().info('Diagnosing failure with LLM...')
                    adjustments = self.llm.diagnose_failure(self.logger.current_episode)
                    self.get_logger().info(f'LLM Adjustments: {adjustments}')
                    self.logger.current_episode["llm_suggested_config"] = adjustments
                    for k, v in adjustments.items():
                        if k in self.current_config:
                            self.current_config[k] = v
                except Exception as e:
                    self.get_logger().warn(f'LLM diagnosis failed: {e}')

        if self.logger:
            self.logger.save()
            self.get_logger().info(f'Episode {self.current_episode} log saved.')

        self.in_episode = False

    def _reset_simulation(self):
        self.get_logger().info("Resetting simulation...")
        msg = String()
        msg.data = "RESET_EPISODE"
        self.cmd_pub.publish(msg)
        time.sleep(5.0)
        self.get_logger().info("Simulation reset complete.")

    def _execute_sequence(self):
        """Send the planned sequence of commands to the executor."""
        cmds = [
            "MOVE_ABOVE_BOTTLE",
            "LOWER_TO_BOTTLE",
            "CLOSE_GRIPPER",
            "LIFT_BOTTLE",
            "MOVE_TO_BEAKER",
            f"TILT_POUR {self.current_config['pour_tilt_deg']}",
            "STOP_POUR",
            "RETURN_HOME",
            "DONE"
        ]

        for i, cmd in enumerate(cmds):
            phase = cmd.split()[0]
            if self.logger:
                self.logger.log_phase(phase)
            msg = String()
            msg.data = cmd
            self.cmd_pub.publish(msg)
            self.get_logger().info(f'  [{i+1}/{len(cmds)}] Sent: {cmd}')
            # Wait between commands to allow execution
            wait_time = 6.0 if 'MOVE' in cmd or 'LIFT' in cmd else 4.0
            time.sleep(wait_time)

            # Registration Gate: Validate if it was actually picked!
            if phase == "LIFT_BOTTLE":
                bottle_detected = False
                for obj in self.detections:
                    if 'bottle' in obj.get('name', '').lower():
                        z_height = float(obj.get('pose', [0, 0, 0])[2])
                        bottle_detected = True
                        if z_height < 0.85:
                            self.get_logger().error(f"Grasp missed! Bottle Z-height is {z_height:.2f} (still on table!)")
                            return "GRASP_FAILED"
                        else:
                            self.get_logger().info(f"Registered Pick Success! Bottle securely lifted to {z_height:.2f}")
                if not bottle_detected:
                    self.get_logger().error("Bottle lost from vision entirely!")
                    return "GRASP_FAILED"

        return "SUCCESS"


def main(args=None):
    rclpy.init(args=args)
    node = SelfTrainingAgent()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
