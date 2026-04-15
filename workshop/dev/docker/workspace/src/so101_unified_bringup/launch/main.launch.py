import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _merge_resource_paths(*groups):
    merged = []
    for group in groups:
        if not group:
            continue
        entries = group.split(os.pathsep) if isinstance(group, str) else group
        for entry in entries:
            if entry and entry not in merged:
                merged.append(entry)
    return os.pathsep.join(merged)


def generate_launch_description():
    package_share = get_package_share_directory("so101_unified_bringup")
    so101_description_share = get_package_share_directory("so101_description")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    moveit_rviz_config = os.path.join(package_share, "config", "moveit.rviz")

    world_default = os.path.join(package_share, "worlds", "empty_world.sdf")

    moveit_config = MoveItConfigsBuilder(
        "so101_new_calib", package_name="so101_moveit_config"
    ).to_moveit_configs()

    gz_resource_paths = _merge_resource_paths(
        os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        [os.path.dirname(so101_description_share), package_share],
    )

    ign_resource_paths = _merge_resource_paths(
        os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        [os.path.dirname(so101_description_share), package_share],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s -v 4 ", LaunchConfiguration("world")],
            "on_exit_shutdown": "true",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description, {"use_sim_time": True}],
    )

    # Spawn robot using the MoveIt-generated robot_description, matching ras_docker flow.
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            LaunchConfiguration("world_name"),
            "-string",
            moveit_config.robot_description["robot_description"],
            "-name",
            LaunchConfiguration("entity_name"),
            "-x",
            LaunchConfiguration("x"),
            "-y",
            LaunchConfiguration("y"),
            "-z",
            LaunchConfiguration("z"),
            "-Y",
            LaunchConfiguration("yaw"),
        ],
        parameters=[{"use_sim_time": True}],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True, "ompl.num_planning_threads": 1}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    moveit_server = Node(
        package="so101_unified_bringup",
        executable="moveit_server",
        name="moveit_server",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": True,
                "move_group_name": LaunchConfiguration("move_group_name"),
                "collision_object_frame": LaunchConfiguration("collision_object_frame"),
                "base_frame_id": LaunchConfiguration("base_frame_id"),
                "end_effector_frame_id": LaunchConfiguration("end_effector_frame_id"),
                "wrist_roll_joint_name": LaunchConfiguration("wrist_roll_joint_name"),
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", moveit_rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    spawn_controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "gripper_controller",
            "-c",
            "/controller_manager",
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"config_file": LaunchConfiguration("bridge_config")}],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("bridge_config"), "' != ''"])
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=world_default),
            DeclareLaunchArgument("world_name", default_value="empty"),
            DeclareLaunchArgument("entity_name", default_value="so101"),
            DeclareLaunchArgument("x", default_value="-0.55"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.7774"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "bridge_config",
                default_value=os.path.join(package_share, "config", "camera_bridge.yaml"),
            ),
            DeclareLaunchArgument("move_group_name", default_value="arm"),
            DeclareLaunchArgument("collision_object_frame", default_value="world"),
            DeclareLaunchArgument("base_frame_id", default_value="base_link"),
            DeclareLaunchArgument("end_effector_frame_id", default_value="gripper_link"),
            DeclareLaunchArgument("wrist_roll_joint_name", default_value="wrist_roll"),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_resource_paths),
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", ign_resource_paths),
            gazebo,
            robot_state_publisher,
            ros_gz_bridge,
            TimerAction(period=10.0, actions=[spawn_robot]),
            TimerAction(period=20.0, actions=[move_group]),
            TimerAction(period=25.0, actions=[spawn_controllers]),
            TimerAction(period=26.0, actions=[moveit_server]),
            TimerAction(period=28.0, actions=[rviz]),
        ]
    )
