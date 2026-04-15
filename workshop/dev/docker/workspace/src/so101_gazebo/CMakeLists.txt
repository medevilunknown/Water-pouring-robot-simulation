cmake_minimum_required(VERSION 3.5)
project(so101_gazebo)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install world files
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install Python scripts as executables
install(PROGRAMS
  scripts/so101_gazebo_bridge.py
  scripts/motion_executor_node.py
  scripts/local_llm_client.py
  scripts/object_detector_3d.py
  scripts/episode_logger.py
  scripts/self_training_agent.py
  scripts/foxglove_env_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()