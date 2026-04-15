from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_config").to_moveit_configs()
    launch_description = generate_rsp_launch(moveit_config)
    launch_description.entities.insert(
        0,
        SetParameter(name="use_sim_time", value=LaunchConfiguration("use_sim_time")),
    )
    launch_description.entities.insert(
        0,
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    )
    return launch_description
