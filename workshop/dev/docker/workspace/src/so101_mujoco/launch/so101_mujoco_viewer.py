#!/usr/bin/python3

import argparse
import threading
import time

import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class So101MujocoViewer(Node):
    def __init__(self, model_path: str):
        super().__init__('so101_mujoco_viewer')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper',
        ]

        self._joint_qpos_addr = {}
        for name in self.joint_names:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id >= 0:
                self._joint_qpos_addr[name] = self.model.jnt_qposadr[j_id]

        self._lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.get_logger().info(f'Viewer model loaded: {model_path}')

    def _on_joint_state(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._joint_qpos_addr:
                    self.data.qpos[self._joint_qpos_addr[name]] = float(pos)
            mujoco.mj_forward(self.model, self.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to MuJoCo scene.xml')
    args = parser.parse_args()

    rclpy.init()
    node = So101MujocoViewer(args.model)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            try:
                with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
                    node.get_logger().info('MuJoCo viewer started.')
                    while rclpy.ok() and viewer.is_running():
                        with node._lock:
                            viewer.sync()
                        time.sleep(0.01)
            except Exception as exc:
                node.get_logger().warn(f'Viewer crashed ({exc}). Restarting in 1s...')
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
