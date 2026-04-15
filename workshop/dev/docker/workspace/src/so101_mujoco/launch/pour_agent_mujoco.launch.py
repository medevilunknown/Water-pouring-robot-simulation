import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    so101_mujoco_share = get_package_share_directory('so101_mujoco')
    so101_gazebo_share = get_package_share_directory('so101_gazebo')

    # Force MuJoCo nodes to use HOST clock (False) instead of simulated /clock!
    use_sim_time = False 

    # 1. Base MuJoCo Environment (Loads Physics, MoveIt, Controllers, Native Viewer)
    mujoco_env = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(so101_mujoco_share, 'launch', 'so101_mujoco.launch.py')
        )
    )

    # 2. Foxglove Bridge (Optional remote viz, loaded securely via YAML)
    foxglove_params_file = os.path.join(
        so101_gazebo_share, 'config', 'foxglove_params.yaml'
    )
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[foxglove_params_file]
    )

    # 3. Environment Object Publisher (Marker Bounding Boxes)
    foxglove_env = Node(
        package='so101_gazebo',  # Reuse from gazebo package
        executable='foxglove_env_publisher.py',
        name='foxglove_env_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 4. Object Detector (Vision)
    vision_node = Node(
        package='so101_gazebo',
        executable='object_detector_3d.py',
        name='object_detector_3d',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 5. Motion Executor Node
    executor_node = Node(
        package='so101_gazebo',
        executable='motion_executor_node.py',
        name='motion_executor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 6. Self-Training AI Agent
    agent_node = Node(
        package='so101_gazebo',
        executable='self_training_agent.py',
        name='self_training_agent',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        # Start immediately
        mujoco_env,
        foxglove_bridge,
        foxglove_env,

        # Staggered logic starts (wait for MuJoCo and MoveGroup to fully initialize on the Host OS)
        TimerAction(period=15.0, actions=[vision_node]),
        TimerAction(period=20.0, actions=[executor_node]),
        TimerAction(period=25.0, actions=[agent_node]),
    ])
