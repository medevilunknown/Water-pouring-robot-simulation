import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    mujoco_share  = get_package_share_directory('so101_mujoco')
    mujoco_prefix = get_package_prefix('so101_mujoco')
    unified_share = get_package_share_directory('so101_unified_bringup')

    mujoco_scene         = os.path.join(mujoco_share, 'mujoco', 'scene.xml')
    mujoco_bridge_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'so101_mujoco_bridge.py')
    mujoco_viewer_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'so101_mujoco_viewer.py')
    moveit_rviz_config   = os.path.join(unified_share, 'config', 'moveit.rviz')

    moveit_config = MoveItConfigsBuilder(
        'so101_new_calib', package_name='so101_moveit_config'
    ).to_moveit_configs()

    # ── nodes ────────────────────────────────────────────────────────────────

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[moveit_config.robot_description, {'use_sim_time': False}],
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': False},
        ],
        arguments=['--ros-args', '--log-level', 'warn'],
    )

    moveit_server = Node(
        package='so101_unified_bringup',
        executable='moveit_server',
        name='moveit_server',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {
                'use_sim_time': False,
                'move_group_name':        LaunchConfiguration('move_group_name'),
                'collision_object_frame': LaunchConfiguration('collision_object_frame'),
                'base_frame_id':          LaunchConfiguration('base_frame_id'),
                'end_effector_frame_id':  LaunchConfiguration('end_effector_frame_id'),
                'wrist_roll_joint_name':  LaunchConfiguration('wrist_roll_joint_name'),
            },
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', moveit_rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': False},
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    mujoco_bridge = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', mujoco_bridge_script,
            '--model',        LaunchConfiguration('mujoco_scene'),
            '--startup-pose', LaunchConfiguration('startup_pose'),
        ],
        output='screen',
    )

    mujoco_viewer = ExecuteProcess(
        cmd=['/usr/bin/python3', mujoco_viewer_script, '--model', LaunchConfiguration('mujoco_scene')],
        condition=IfCondition(LaunchConfiguration('show_viewer')),
        output='screen',
    )

    # ── launch description ───────────────────────────────────────────────────

    return LaunchDescription([
        DeclareLaunchArgument('mujoco_scene',   default_value=mujoco_scene),
        DeclareLaunchArgument('startup_pose',   default_value='home',
                              description='Initial arm pose: home or upright'),
        DeclareLaunchArgument('show_viewer',    default_value='true',
                              description='Launch MuJoCo GUI viewer (true|false)'),
        DeclareLaunchArgument('rviz',           default_value='true',
                              description='Launch RViz with MoveIt config (true|false)'),
        DeclareLaunchArgument('move_group_name',        default_value='arm'),
        DeclareLaunchArgument('collision_object_frame', default_value='world'),
        DeclareLaunchArgument('base_frame_id',          default_value='base_link'),
        DeclareLaunchArgument('end_effector_frame_id',  default_value='gripper_link'),
        DeclareLaunchArgument('wrist_roll_joint_name',  default_value='wrist_roll'),

        # MuJoCo bridge starts immediately (physics + camera + action servers)
        mujoco_bridge,
        mujoco_viewer,
        robot_state_publisher,
        # MoveIt needs RSP + joint_states first
        TimerAction(period=2.0,  actions=[move_group]),
        # moveit_server needs move_group ready
        TimerAction(period=5.0,  actions=[moveit_server]),
        # RViz last
        TimerAction(period=7.0,  actions=[rviz]),
    ])
