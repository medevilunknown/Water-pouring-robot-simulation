#!/usr/bin/python3
"""
Motion Executor Node
====================
Receives high-level commands from the LLM agent on /agent/command and
translates them into MoveIt2 service calls + gripper trajectory commands.

Also runs the water flow simulation: reads the wrist_roll joint angle
to determine bottle tilt, computes flow rate, and publishes /water_state.

Uses the existing MoveIt services:
  /create_traj        (PoseReq)       — move to a Cartesian pose
  /move_to_joint_states (JointReq)    — move to joint configuration
  /rotate_effector    (RotateEffector) — rotate wrist by delta angle
  /pick_object        (PickObject)    — grasp workflow
"""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Quaternion
from builtin_interfaces.msg import Duration

# MoveIt service interfaces from so101_unified_bringup
try:
    from so101_unified_bringup.srv import PoseReq, JointReq, RotateEffector
    MOVEIT_SERVICES = True
except ImportError:
    MOVEIT_SERVICES = False

# ── Configuration ─────────────────────────────────────────────
BEAKER_CAPACITY_ML = 150.0
BOTTLE_WATER_ML = 120.0
POUR_RATE_ML_PER_SEC = 25.0

# Robot spawn pose in world frame
ROBOT_BASE_X = -0.55
ROBOT_BASE_Y = 0.0
ROBOT_BASE_Z = 0.7774

# Object positions (world frame) — used as defaults, overridden by vision
BOTTLE_WORLD = [-0.27, -0.10, 0.839]
BEAKER_WORLD = [-0.30, 0.12, 0.810]

# Home joint configuration
HOME_JOINTS = [0.0, -0.5, 0.5, 0.0, 0.0]

# Joint names
ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll']
GRIPPER_JOINT = 'gripper'


def world_to_robot(world_pos):
    """Convert world-frame position to robot base_link frame."""
    return [
        world_pos[0] - ROBOT_BASE_X,
        world_pos[1] - ROBOT_BASE_Y,
        world_pos[2] - ROBOT_BASE_Z,
    ]


def make_pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Create a geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


class WaterSimulator:
    """Software-simulated water flow based on bottle tilt angle."""

    def __init__(self, bottle_ml=BOTTLE_WATER_ML, beaker_cap=BEAKER_CAPACITY_ML):
        self.bottle_ml = bottle_ml
        self.beaker_ml = 0.0
        self.beaker_capacity = beaker_cap
        self.is_pouring = False
        self.total_poured = 0.0

    def update(self, tilt_angle_deg, dt):
        """Update water levels based on bottle tilt angle."""
        if tilt_angle_deg > 30 and self.bottle_ml > 0:
            flow_factor = min(1.0, (tilt_angle_deg - 30) / 60.0)
            flow = POUR_RATE_ML_PER_SEC * flow_factor * dt
            actual_flow = min(flow, self.bottle_ml)
            self.bottle_ml -= actual_flow
            self.beaker_ml += actual_flow
            self.total_poured += actual_flow
            self.is_pouring = True
        else:
            self.is_pouring = False
        return self.beaker_ml

    @property
    def beaker_pct(self):
        return (self.beaker_ml / self.beaker_capacity) * 100

    @property
    def is_overspilling(self):
        return self.beaker_ml >= self.beaker_capacity

    def to_dict(self):
        return {
            'bottle_ml': round(self.bottle_ml, 1),
            'beaker_ml': round(self.beaker_ml, 1),
            'beaker_capacity_ml': self.beaker_capacity,
            'beaker_fill_pct': round(self.beaker_pct, 1),
            'total_poured_ml': round(self.total_poured, 1),
            'is_pouring': self.is_pouring,
            'is_overspilling': self.is_overspilling,
        }


class MotionExecutorNode(Node):
    """Translates high-level agent commands into MoveIt motions."""

    def __init__(self):
        super().__init__('motion_executor_node')
        self.get_logger().info('Initialising Motion Executor Node...')

        # Force sim time
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.cb_group = ReentrantCallbackGroup()

        # ── State ─────────────────────────────────────────────
        self.current_joint_positions = {}
        self.water_sim = WaterSimulator()
        self.bottle_pos = BOTTLE_WORLD[:]
        self.beaker_pos = BEAKER_WORLD[:]
        self.executing = False
        self.gripper_closed = False
        self.accumulated_tilt = 0.0  # Track total tilt applied
        
        # Agent config that can be dynamically updated (tuned)
        self.config = {
            "approach_offset_z": 0.10,
            "grip_position": -0.10,
            "pour_tilt_deg": 75.0,  # Explicitly increased default tilt per user request
            "pour_duration_sec": 3.0
        }

        # ── Subscriptions ─────────────────────────────────────
        self.create_subscription(
            String, '/agent/command', self._command_cb, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            String, '/agent/config', self._config_cb, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            String, '/detected_objects', self._detections_cb, 10,
            callback_group=self.cb_group
        )

        # ── Publishers ────────────────────────────────────────
        self.ack_pub = self.create_publisher(String, '/agent/action_ack', 10)
        self.water_pub = self.create_publisher(String, '/water_state', 10)
        self.arm_traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )
        self.gripper_traj_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10
        )

        # ── MoveIt Service Clients ────────────────────────────
        if MOVEIT_SERVICES:
            self.pose_client = self.create_client(
                PoseReq, '/create_traj', callback_group=self.cb_group
            )
            self.joint_client = self.create_client(
                JointReq, '/move_to_joint_states', callback_group=self.cb_group
            )
            self.rotate_client = self.create_client(
                RotateEffector, '/rotate_effector', callback_group=self.cb_group
            )
            self.get_logger().info('MoveIt service clients created ✓')
        else:
            self.pose_client = None
            self.joint_client = None
            self.rotate_client = None
            self.get_logger().warn(
                'MoveIt services unavailable — using direct trajectory publishing'
            )

        # ── Water simulation timer (10 Hz) ────────────────────
        self.create_timer(0.1, self._water_sim_loop, callback_group=self.cb_group)

        self.get_logger().info('Motion Executor ready ✓')

    # ── Callbacks ─────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def _detections_cb(self, msg: String):
        """Update object positions from vision."""
        try:
            dets = json.loads(msg.data)
            for det in dets:
                name = det.get('name', '')
                pos = det.get('world_position')
                if pos:
                    if name == 'water_bottle':
                        self.bottle_pos = pos
                    elif name == 'target_beaker':
                        self.beaker_pos = pos
        except json.JSONDecodeError:
            pass

    def _config_cb(self, msg: String):
        """Update executing configuration parameters from the agent."""
        try:
            new_config = json.loads(msg.data)
            self.config.update(new_config)
            self.get_logger().info(f'Configuration updated: {self.config}')
        except json.JSONDecodeError:
            self.get_logger().warn('Received invalid JSON configuration.')

    def _command_cb(self, msg: String):
        """Handle incoming agent command."""
        cmd = msg.data.strip()
        self.get_logger().info(f'Received command: {cmd}')

        # Execute in a thread to avoid blocking the callback
        thread = threading.Thread(target=self._execute_command, args=(cmd,))
        thread.daemon = True
        thread.start()

    # ── Water simulation ──────────────────────────────────────

    def _get_bottle_tilt_deg(self) -> float:
        """Estimate bottle tilt from wrist_roll joint angle."""
        wrist_roll = self.current_joint_positions.get('wrist_roll', 0.0)
        # When the gripper is closed and tilted, the bottle tilts with it
        if self.gripper_closed:
            return abs(math.degrees(wrist_roll))
        return 0.0

    def _water_sim_loop(self):
        """Update water simulation and publish state."""
        tilt = self._get_bottle_tilt_deg()
        self.water_sim.update(tilt, 0.1)  # 0.1s per tick at 10Hz

        msg = String()
        msg.data = json.dumps(self.water_sim.to_dict())
        self.water_pub.publish(msg)

    # ── Motion execution ──────────────────────────────────────

    def _execute_command(self, cmd: str):
        """Execute a high-level command."""
        if self.executing:
            self.get_logger().warn('Already executing — ignoring command')
            return

        self.executing = True
        success = False

        try:
            parts = cmd.strip().split()
            action = parts[0]
            param = parts[1] if len(parts) > 1 else None

            # Handle multi-word action names
            if len(parts) >= 3:
                three_word = '_'.join(parts[:3]).upper()
                if three_word in ('MOVE_ABOVE_BOTTLE', 'LOWER_TO_BOTTLE', 'MOVE_TO_BEAKER'):
                    action = three_word
                    param = parts[3] if len(parts) > 3 else None
            if len(parts) >= 2:
                two_word = '_'.join(parts[:2]).upper()
                if two_word in ('TILT_POUR', 'STOP_POUR', 'RETURN_HOME',
                               'CLOSE_GRIPPER', 'LIFT_BOTTLE', 'MOVE_ABOVE',
                               'LOWER_TO', 'MOVE_TO'):
                    action = two_word
                    param = parts[2] if len(parts) > 2 else None

            action = action.upper()
            self.get_logger().info(f'Executing: {action} (param={param})')

            if action == 'MOVE_ABOVE_BOTTLE':
                success = self._move_above_bottle()

            elif action in ('LOWER_TO_BOTTLE', 'LOWER_TO'):
                success = self._lower_to_bottle()

            elif action in ('CLOSE_GRIPPER', 'CLOSE'):
                success = self._close_gripper()

            elif action in ('LIFT_BOTTLE', 'LIFT'):
                success = self._lift_bottle()

            elif action in ('MOVE_TO_BEAKER', 'MOVE_TO'):
                success = self._move_to_beaker()

            elif action in ('TILT_POUR', 'TILT'):
                angle = 45.0
                if param:
                    try:
                        angle = float(param)
                    except ValueError:
                        angle = 45.0
                success = self._tilt_pour(angle)

            elif action in ('STOP_POUR', 'STOP'):
                success = self._stop_pour()

            elif action == 'RETURN_HOME':
                success = self._return_home()

            elif action == 'DONE':
                success = True
                self.get_logger().info('Task DONE — waiting for next instructions')
                
            elif action == 'RESET_EPISODE':
                success = self._reset_episode()

            else:
                self.get_logger().warn(f'Unknown action: {action}')

        except Exception as e:
            self.get_logger().error(f'Execution error: {e}')
        finally:
            self.executing = False

        # Acknowledge
        ack = String()
        ack.data = json.dumps({
            'command': cmd,
            'success': success,
            'water': self.water_sim.to_dict(),
        })
        self.ack_pub.publish(ack)

    # ── Individual actions ────────────────────────────────────

    def _move_above_bottle(self) -> bool:
        """Move gripper above the bottle (pre-grasp)."""
        pos = world_to_robot(self.bottle_pos)
        pos[2] += self.config.get("approach_offset_z", 0.10)  # Use configurable offset
        # Gripper pointing down: quat for -Z orientation
        pose = make_pose(pos[0], pos[1], pos[2],
                         qx=0.0, qy=0.7071, qz=0.0, qw=0.7071)
        return self._call_pose_service(pose)

    def _lower_to_bottle(self) -> bool:
        """Lower gripper to bottle grasp position."""
        pos = world_to_robot(self.bottle_pos)
        pos[2] += 0.02  # Assumes gripper grabs slightly above center
        pose = make_pose(pos[0], pos[1], pos[2],
                         qx=0.0, qy=0.7071, qz=0.0, qw=0.7071)
        return self._call_pose_service(pose)

    def _close_gripper(self) -> bool:
        """Close the gripper with tuned position."""
        target_grip = self.config.get("grip_position", -0.1)
        self._send_gripper_trajectory(target_grip)
        self.gripper_closed = True
        
        # User explicitly requested to adjust speed/stability to ensure picking works
        self.get_logger().info('Stabilizing gripper payload before lifting...')
        time.sleep(3.0)  # Wait for gripper physics to aggressively secure the bottle block
        return True

    def _open_gripper(self) -> bool:
        """Open the gripper."""
        self._send_gripper_trajectory(1.2)
        self.gripper_closed = False
        time.sleep(1.0)
        return True

    def _lift_bottle(self) -> bool:
        """Lift the bottle from the table."""
        pos = world_to_robot(self.bottle_pos)
        pos[2] += 0.18  # Lift 18cm above original position
        pose = make_pose(pos[0], pos[1], pos[2],
                         qx=0.0, qy=0.7071, qz=0.0, qw=0.7071)
        return self._call_pose_service(pose)

    def _move_to_beaker(self) -> bool:
        """Move the held bottle above the beaker."""
        pos = world_to_robot(self.beaker_pos)
        pos[2] += 0.20  # 20cm above beaker
        pose = make_pose(pos[0], pos[1], pos[2],
                         qx=0.0, qy=0.7071, qz=0.0, qw=0.7071)
        return self._call_pose_service(pose)

    def _tilt_pour(self, angle_deg: float) -> bool:
        """Tilt the bottle by rotating the wrist."""
        angle_deg = min(90.0, max(10.0, angle_deg))
        angle_rad = math.radians(angle_deg)

        # Use rotate_effector service or direct trajectory
        if self.rotate_client and self.rotate_client.wait_for_service(timeout_sec=2.0):
            req = RotateEffector.Request()
            req.rotation_angle = angle_rad
            future = self.rotate_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.result() is not None:
                self.accumulated_tilt += angle_rad
                # Hold pour position for 3 seconds to let water flow
                time.sleep(3.0)
                return future.result().success
        else:
            # Direct trajectory fallback
            self.accumulated_tilt += angle_rad
            current = list(self.current_joint_positions.get(j, 0.0)
                           for j in ARM_JOINTS)
            current[4] += angle_rad  # wrist_roll
            self._send_arm_trajectory(current, duration_sec=2.0)
            time.sleep(3.0)  # Hold pour
            return True

        return False

    def _stop_pour(self) -> bool:
        """Return bottle to upright by reversing accumulated tilt."""
        if abs(self.accumulated_tilt) > 0.01:
            if self.rotate_client and self.rotate_client.wait_for_service(timeout_sec=2.0):
                req = RotateEffector.Request()
                req.rotation_angle = -self.accumulated_tilt
                future = self.rotate_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
                self.accumulated_tilt = 0.0
                if future.result() is not None:
                    return future.result().success
            else:
                # Direct trajectory fallback
                current = list(self.current_joint_positions.get(j, 0.0)
                               for j in ARM_JOINTS)
                current[4] -= self.accumulated_tilt
                self._send_arm_trajectory(current, duration_sec=2.0)
                self.accumulated_tilt = 0.0
                time.sleep(2.0)
                return True
        return True

    def _return_home(self) -> bool:
        """Return arm to home configuration and open gripper."""
        # First open gripper to release bottle
        self._open_gripper()
        time.sleep(0.5)

        # Move to home joints
        if self.joint_client and self.joint_client.wait_for_service(timeout_sec=2.0):
            req = JointReq.Request()
            req.joints = JointState()
            req.joints.name = ARM_JOINTS
            req.joints.position = HOME_JOINTS
            future = self.joint_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
            if future.result() is not None:
                return future.result().success
        else:
            self._send_arm_trajectory(HOME_JOINTS, duration_sec=3.0)
            time.sleep(3.0)
            return True
        return False

    def _reset_episode(self) -> bool:
        """Return home and reset water simulator state for next RL iteration."""
        self.get_logger().info('Resetting environment for next Episode...')
        self.water_sim = WaterSimulator()
        self._return_home()
        time.sleep(2.0)
        return True

    # ── Service call helpers ──────────────────────────────────

    def _call_pose_service(self, pose: Pose, constraint=False) -> bool:
        """Call the /create_traj MoveIt service."""
        if self.pose_client and self.pose_client.wait_for_service(timeout_sec=5.0):
            req = PoseReq.Request()
            req.object_pose = pose
            req.constraint = constraint
            req.type = 'pour'

            self.get_logger().info(
                f'  MoveIt target: ({pose.position.x:.3f}, '
                f'{pose.position.y:.3f}, {pose.position.z:.3f})'
            )

            future = self.pose_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            if future.result() is not None:
                if future.result().success:
                    self.get_logger().info('  MoveIt motion succeeded ✓')
                else:
                    self.get_logger().warn('  MoveIt motion failed ✗')
                return future.result().success
            else:
                self.get_logger().error('  MoveIt service call timed out')
        else:
            # Fallback: compute approximate joint targets and send directly
            self.get_logger().warn('  MoveIt unavailable — using direct joint trajectory')
            self._fallback_move_to_pose(pose)
            return True
        return False

    def _fallback_move_to_pose(self, pose: Pose):
        """Simple joint-space fallback when MoveIt is unavailable."""
        # Use a basic IK approximation for the SO-101
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # Shoulder pan: atan2(y, x)
        pan = math.atan2(y, x)

        # Distance in XY plane
        r = math.sqrt(x*x + y*y)

        # Simple 2-link planar IK for shoulder_lift and elbow_flex
        L1 = 0.11257  # upper arm length
        L2 = 0.1349   # lower arm length

        d = math.sqrt(r*r + z*z)
        d = min(d, L1 + L2 - 0.01)  # Clamp to reachable

        cos_elbow = (d*d - L1*L1 - L2*L2) / (2 * L1 * L2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow = math.acos(cos_elbow)

        # Shoulder lift
        alpha = math.atan2(z, r)
        beta = math.atan2(L2 * math.sin(elbow), L1 + L2 * cos_elbow)
        shoulder_lift = alpha + beta

        joints = [pan, -shoulder_lift, elbow - math.pi/2, 0.0, 0.0]
        # Clamp all joints to safe ranges
        limits = [1.91, 1.74, 1.69, 1.65, 2.74]
        for i in range(5):
            joints[i] = max(-limits[i], min(limits[i], joints[i]))

        self._send_arm_trajectory(joints, duration_sec=3.0)
        time.sleep(3.0)

    # ── Direct trajectory publishers ──────────────────────────

    def _send_arm_trajectory(self, positions, duration_sec=2.0):
        """Publish a direct arm joint trajectory."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9)
        )
        msg.points.append(point)
        self.arm_traj_pub.publish(msg)
        self.get_logger().info(f'  Arm trajectory published: {[f"{p:.3f}" for p in positions]}')

    def _send_gripper_trajectory(self, position, duration_sec=1.0):
        """Publish a gripper trajectory command."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [GRIPPER_JOINT]

        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9)
        )
        msg.points.append(point)
        self.gripper_traj_pub.publish(msg)
        self.get_logger().info(f'  Gripper → {position:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = MotionExecutorNode()
    try:
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
