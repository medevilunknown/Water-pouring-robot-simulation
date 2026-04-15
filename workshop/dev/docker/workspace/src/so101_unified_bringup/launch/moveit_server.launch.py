import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    package_share = get_package_share_directory("so101_unified_bringup")
    moveit_config = MoveItConfigsBuilder(
        "so101_new_calib", package_name="so101_moveit_config"
    ).to_moveit_configs()

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("move_group_name", default_value="arm"),
        DeclareLaunchArgument("collision_object_frame", default_value="world"),
        DeclareLaunchArgument("base_frame_id", default_value="base_link"),
        DeclareLaunchArgument("end_effector_frame_id", default_value="gripper_link"),
        DeclareLaunchArgument("wrist_roll_joint_name", default_value="wrist_roll"),
        Node(
            package="so101_unified_bringup",
            executable="moveit_server",
            name="moveit_server",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "move_group_name": LaunchConfiguration("move_group_name"),
                    "collision_object_frame": LaunchConfiguration("collision_object_frame"),
                    "base_frame_id": LaunchConfiguration("base_frame_id"),
                    "end_effector_frame_id": LaunchConfiguration("end_effector_frame_id"),
                    "wrist_roll_joint_name": LaunchConfiguration("wrist_roll_joint_name"),
                },
            ],
        ),
    ])
