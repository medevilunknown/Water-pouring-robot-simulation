cmake_minimum_required(VERSION 3.8)
project(so101_mujoco)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  mujoco
  meshes
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/so101_mujoco_bridge.py
  scripts/so101_keyboard_teleop.py
  scripts/so101_button_teleop.py
  scripts/so101_phone_teleop.py
  scripts/so101_virtual_joystick.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
