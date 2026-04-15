cmake_minimum_required(VERSION 3.22)
project(so101_unified_bringup)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(trajectory_msgs REQUIRED)

set(srv_files
  "srv/ActionTraj.srv"
  "srv/JointReq.srv"
  "srv/JointSat.srv"
  "srv/PickFront.srv"
  "srv/PickLeft.srv"
  "srv/PickObject.srv"
  "srv/PickRear.srv"
  "srv/PickRight.srv"
  "srv/PlaceObject.srv"
  "srv/PoseReq.srv"
  "srv/RotateEffector.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES geometry_msgs moveit_msgs sensor_msgs trajectory_msgs
)

add_executable(moveit_server src/moveit_server.cpp)
target_include_directories(moveit_server PUBLIC include)
target_compile_features(moveit_server PUBLIC cxx_std_17)

ament_target_dependencies(moveit_server
  control_msgs
  geometric_shapes
  geometry_msgs
  interactive_markers
  moveit_msgs
  moveit_ros_planning
  moveit_ros_planning_interface
  rclcpp
  sensor_msgs
  std_srvs
  tf2
  trajectory_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(moveit_server "${cpp_typesupport_target}")

install(TARGETS moveit_server DESTINATION lib/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
  install(
    DIRECTORY include/
    DESTINATION include
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/worlds")
  install(
    DIRECTORY worlds
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

install(FILES README.md DESTINATION share/${PROJECT_NAME})
