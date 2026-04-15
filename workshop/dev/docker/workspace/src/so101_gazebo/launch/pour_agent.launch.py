"""
Pour Agent Launch File
======================
Launches the full Gazebo + MoveIt + LLM pouring pipeline:

  1. so101_unified_bringup/main.launch.py  (Gazebo + MoveIt + controllers + camera bridge)
  2. vision_detector_node.py                (YOLO object detection)
  3. llm_pour_agent_node.py                 (Llama 3 decision-making)
  4. motion_executor_node.py                (MoveIt motion execution + water sim)

Usage (inside Docker container):
  export GROQ_API_KEY="your-key"
  ros2 launch so101_gazebo pour_agent.launch.py

Without Groq key (uses scripted fallback):
  ros2 launch so101_gazebo pour_agent.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory('so101_gazebo')
    bringup_share = get_package_share_directory('so101_unified_bringup')

    world_path = os.path.join(gazebo_share, 'worlds', 'empty_world.sdf')

    # Pass GROQ_API_KEY from environment if available
    groq_key = os.environ.get('GROQ_API_KEY', '')

    # ── 1. Launch the full Gazebo + MoveIt + Controllers stack ──
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'main.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'world_name': 'empty',
            'rviz': LaunchConfiguration('rviz'),
        }.items(),
    )

    # ──# 2. Vision Detector Node (3D detector) ───────────────────────────────
    vision_node = Node(
        package='so101_gazebo',
        executable='object_detector_3d.py',
        name='object_detector_3d',
        output='screen',
        parameters=[{
            'use_sim_time': True,
        }],
    )

    # ──# 3. LLM Pour Agent Node (Self-Training Loop) ────────────────────────────────
    agent_node = Node(
        package='so101_gazebo',
        executable='self_training_agent.py',
        name='self_training_agent',
        output='screen',
        parameters=[{
            'use_sim_time': True,
        }],
    )

    # ── 4. Motion Executor Node ───────────────────────────────
    executor_node = Node(
        package='so101_gazebo',
        executable='motion_executor_node.py',
        name='motion_executor',
        output='screen',
        parameters=[{
            'use_sim_time': True,
        }],
    )
    # ── 5. Foxglove Bridge (Visualization) ───────────────────
    foxglove_params_file = os.path.join(
        gazebo_share, 'config', 'foxglove_params.yaml'
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[foxglove_params_file]
    )

    # ── 6. Foxglove Environment Visualizer ───────────────────
    foxglove_env_pub = Node(
        package='so101_gazebo',
        executable='foxglove_env_publisher.py',
        name='foxglove_env_pub',
        output='screen',
    )

    # ── 7. Camera Transform Fix for Gazebo/Foxglove ────────
    camera_tf_fix = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_fix',
        arguments=['0', '0', '0', '0', '0', '0', 'd435i_link', 'so101/gripper_link/d435i_rgbd']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Open RViz for visualization'
        ),
        
        DeclareLaunchArgument(
            'groq_key',
            default_value=groq_key,
            description='Groq API key for Llama 3 (optional — falls back to scripted)',
        ),

        # Launch Gazebo + MoveIt stack
        bringup,
        
        # Launch Visualization Bridge
        foxglove_bridge,
        
        # Launch Environment Visualizer
        foxglove_env_pub,
        
        # Launch Camera Transform Fix
        camera_tf_fix,

        # After everything is settled (controllers spawned ~11s, moveit ~12s),
        # start the pour agent nodes
        TimerAction(period=35.0, actions=[vision_node]),
        TimerAction(period=38.0, actions=[executor_node]),
        TimerAction(period=40.0, actions=[agent_node]),
    ])
