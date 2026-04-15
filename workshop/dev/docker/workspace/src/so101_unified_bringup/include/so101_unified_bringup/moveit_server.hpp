#ifndef SO101_UNIFIED_BRINGUP_MOVEIT_SERVER_HPP
#define SO101_UNIFIED_BRINGUP_MOVEIT_SERVER_HPP

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <unordered_map>
#include <vector>
#include <memory>

#include "geometry_msgs/msg/pose.hpp"
#include "so101_unified_bringup/srv/action_traj.hpp"
#include "so101_unified_bringup/srv/joint_req.hpp"
#include "so101_unified_bringup/srv/joint_sat.hpp"
#include "so101_unified_bringup/srv/pick_front.hpp"
#include "so101_unified_bringup/srv/pick_left.hpp"
#include "so101_unified_bringup/srv/pick_object.hpp"
#include "so101_unified_bringup/srv/pick_rear.hpp"
#include "so101_unified_bringup/srv/pick_right.hpp"
#include "so101_unified_bringup/srv/place_object.hpp"
#include "so101_unified_bringup/srv/pose_req.hpp"
#include "so101_unified_bringup/srv/rotate_effector.hpp"

class MoveitServer : public rclcpp::Node
{
public:
    MoveitServer();
    void initialize_move_group();

private:
    void move_to_pose_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PoseReq::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PoseReq::Response> response);
    void move_to_joint_callback(
        const std::shared_ptr<so101_unified_bringup::srv::JointReq::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::JointReq::Response> response);
    void rotate_effector_callback(
        const std::shared_ptr<so101_unified_bringup::srv::RotateEffector::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::RotateEffector::Response> response);
    void place_object_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PlaceObject::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PlaceObject::Response> response);
    void pick_object_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PickObject::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PickObject::Response> response);
    void pick_front_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PickFront::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PickFront::Response> response);
    void pick_right_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PickRight::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PickRight::Response> response);
    void pick_left_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PickLeft::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PickLeft::Response> response);
    void pick_rear_callback(
        const std::shared_ptr<so101_unified_bringup::srv::PickRear::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::PickRear::Response> response);
    void trajectory_callback(
        const std::shared_ptr<so101_unified_bringup::srv::ActionTraj::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::ActionTraj::Response> response);
    void sync_callback(
        const std::shared_ptr<so101_unified_bringup::srv::JointSat::Request> request,
        std::shared_ptr<so101_unified_bringup::srv::JointSat::Response> response);
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

    void AddScenePlane();
    void set_constraints(const geometry_msgs::msg::Quaternion &quat);
    bool Execute(const geometry_msgs::msg::Pose &target_pose, bool apply_orientation_constraint = true);
    bool Execute(const sensor_msgs::msg::JointState &target_joints);
    bool execute_plan(moveit::planning_interface::MoveGroupInterface::Plan &plan);
    void configure_move_group(
        double planning_time,
        double goal_tolerance,
        double orientation_tolerance,
        double velocity_scaling,
        double acceleration_scaling,
        bool clear_constraints);
    std::vector<double> extract_joint_targets(
        const sensor_msgs::msg::JointState &target_joints,
        bool &ok,
        std::string &error) const;
    geometry_msgs::msg::Pose rotated_pose(
        const geometry_msgs::msg::Pose &pose,
        double roll_delta,
        double pitch_delta,
        double yaw_delta) const;
    bool move_pose_with_message(
        const geometry_msgs::msg::Pose &target_pose,
        std::string &error_message,
        const std::string &failure_context);
    void simulate_gripper_action(bool grip_state) const;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    std::unordered_map<std::string, double> current_joint_positions_;
    std::vector<std::string> arm_joint_names_;

    rclcpp::Service<so101_unified_bringup::srv::PoseReq>::SharedPtr move_to_pose_srv_;
    rclcpp::Service<so101_unified_bringup::srv::JointReq>::SharedPtr move_to_joint_srv_;
    rclcpp::Service<so101_unified_bringup::srv::RotateEffector>::SharedPtr rotate_effector_srv_;
    rclcpp::Service<so101_unified_bringup::srv::JointSat>::SharedPtr sync_srv_;
    rclcpp::Service<so101_unified_bringup::srv::ActionTraj>::SharedPtr execute_traj_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PlaceObject>::SharedPtr place_object_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PickObject>::SharedPtr pick_object_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PickFront>::SharedPtr pick_front_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PickRight>::SharedPtr pick_right_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PickLeft>::SharedPtr pick_left_srv_;
    rclcpp::Service<so101_unified_bringup::srv::PickRear>::SharedPtr pick_rear_srv_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;

    std::string collision_object_frame_;
    std::string base_frame_id_;
    std::string end_effector_frame_id_;
    std::string move_group_name_;
    std::string planner_id_;
    std::string wrist_roll_joint_name_;
};

#endif