import socket
import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading
import sys

class So101GazeboBridge(Node):
    def __init__(self):
        super().__init__('so101_gazebo_bridge')
        
        # IMPORTANT: Force use_sim_time so timestamps match Gazebo
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # ROS 2 Publishers for Gazebo Controllers
        self.arm_pub = self.create_publisher(
            JointTrajectory, 
            '/arm_controller/joint_trajectory', 
            10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, 
            '/gripper_controller/joint_trajectory', 
            10
        )
        
        # Joint Names mapping
        self.arm_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        self.gripper_joints = ['gripper']
        
        # UDP Setup
        self.udp_ip = "0.0.0.0" # Listen on all interfaces inside container
        self.udp_port = 9876
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.udp_ip, self.udp_port))
            self.sock.settimeout(1.0)
            self.get_logger().info(f"Successfully bound to UDP {self.udp_ip}:{self.udp_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to bind to UDP port {self.udp_port}: {e}")
            sys.exit(1)
        
        self.packet_count = 0
        self.running = True
        self.thread = threading.Thread(target=self.udp_listener)
        self.thread.daemon = True
        self.thread.start()
        
        self.get_logger().info("------------------------------------------------")
        self.get_logger().info("GZ BRIDGE IS READY. WAITING FOR TELEOP PACKETS...")
        self.get_logger().info("------------------------------------------------")

    def udp_listener(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.packet_count += 1
                if self.packet_count % 10 == 0:
                    self.get_logger().info(f"Received {self.packet_count} packets so far (Latest from {addr})")
                
                msg = json.loads(data.decode())
                self.process_command(msg)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f"UDP Packet Error: {e}")

    def process_command(self, cmd_dict):
        now = self.get_clock().now().to_msg()
        
        # 1. Handle Arm Joints
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = now
        arm_msg.joint_names = self.arm_joints
        
        point = JointTrajectoryPoint()
        positions = []
        found_arm = False
        
        for j in self.arm_joints:
            if j in cmd_dict:
                positions.append(float(cmd_dict[j]))
                found_arm = True
            else:
                positions.append(0.0) 
        
        if found_arm:
            point.positions = positions
            point.time_from_start.nanosec = 100000000 # 100ms
            arm_msg.points.append(point)
            self.arm_pub.publish(arm_msg)

        # 2. Handle Gripper
        if 'gripper' in cmd_dict:
            grip_msg = JointTrajectory()
            grip_msg.header.stamp = now
            grip_msg.joint_names = self.gripper_joints
            g_point = JointTrajectoryPoint()
            g_point.positions = [float(cmd_dict['gripper'])]
            g_point.time_from_start.nanosec = 100000000
            grip_msg.points.append(g_point)
            self.gripper_pub.publish(grip_msg)

    def stop(self):
        self.running = False
        self.sock.close()

def main(args=None):
    rclpy.init(args=args)
    node = So101GazeboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
