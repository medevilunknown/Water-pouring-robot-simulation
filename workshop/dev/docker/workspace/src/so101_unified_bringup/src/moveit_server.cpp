#include "so101_unified_bringup/moveit_server.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace
{
    const rclcpp::Logger LOGGER = rclcpp::get_logger("so101_moveit_server");
} // namespace

MoveitServer::MoveitServer()
    : Node("moveit_server")
{
    this->declare_parameter("move_group_name", "arm");
    this->declare_parameter("collision_object_frame", "world");
    this->declare_parameter("base_frame_id", "base_link");
    this->declare_parameter("end_effector_frame_id", "gripper_link");
    this->declare_parameter("planner_id", "RRTConnect");
    this->declare_parameter("wrist_roll_joint_name", "wrist_roll");
    this->declare_parameter(
        "arm_joint_names",
        std::vector<std::string>{
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"});

    move_group_name_ = this->get_parameter("move_group_name").as_string();
    collision_object_frame_ = this->get_parameter("collision_object_frame").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();
    end_effector_frame_id_ = this->get_parameter("end_effector_frame_id").as_string();
    planner_id_ = this->get_parameter("planner_id").as_string();
    wrist_roll_joint_name_ = this->get_parameter("wrist_roll_joint_name").as_string();
    arm_joint_names_ = this->get_parameter("arm_joint_names").as_string_array();

    move_to_pose_srv_ = this->create_service<so101_unified_bringup::srv::PoseReq>(
        "/create_traj",
        std::bind(&MoveitServer::move_to_pose_callback, this, std::placeholders::_1, std::placeholders::_2));

    move_to_joint_srv_ = this->create_service<so101_unified_bringup::srv::JointReq>(
        "/move_to_joint_states",
        std::bind(&MoveitServer::move_to_joint_callback, this, std::placeholders::_1, std::placeholders::_2));

    rotate_effector_srv_ = this->create_service<so101_unified_bringup::srv::RotateEffector>(
        "/rotate_effector",
        std::bind(&MoveitServer::rotate_effector_callback, this, std::placeholders::_1, std::placeholders::_2));

    sync_srv_ = this->create_service<so101_unified_bringup::srv::JointSat>(
        "/sync_arm",
        std::bind(&MoveitServer::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

    execute_traj_srv_ = this->create_service<so101_unified_bringup::srv::ActionTraj>(
        "trajectory_topic",
        std::bind(&MoveitServer::trajectory_callback, this, std::placeholders::_1, std::placeholders::_2));

    place_object_srv_ = this->create_service<so101_unified_bringup::srv::PlaceObject>(
        "/place_object",
        std::bind(&MoveitServer::place_object_callback, this, std::placeholders::_1, std::placeholders::_2));

    pick_object_srv_ = this->create_service<so101_unified_bringup::srv::PickObject>(
        "/pick_object",
        std::bind(&MoveitServer::pick_object_callback, this, std::placeholders::_1, std::placeholders::_2));

    pick_front_srv_ = this->create_service<so101_unified_bringup::srv::PickFront>(
        "/pick_front",
        std::bind(&MoveitServer::pick_front_callback, this, std::placeholders::_1, std::placeholders::_2));

    pick_right_srv_ = this->create_service<so101_unified_bringup::srv::PickRight>(
        "/pick_right",
        std::bind(&MoveitServer::pick_right_callback, this, std::placeholders::_1, std::placeholders::_2));

    pick_left_srv_ = this->create_service<so101_unified_bringup::srv::PickLeft>(
        "/pick_left",
        std::bind(&MoveitServer::pick_left_callback, this, std::placeholders::_1, std::placeholders::_2));

    pick_rear_srv_ = this->create_service<so101_unified_bringup::srv::PickRear>(
        "/pick_rear",
        std::bind(&MoveitServer::pick_rear_callback, this, std::placeholders::_1, std::placeholders::_2));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states",
        10,
        std::bind(&MoveitServer::joint_state_callback, this, std::placeholders::_1));

    trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("trajectory_topic", 10);

    AddScenePlane();
}

void MoveitServer::initialize_move_group()
{
    move_group_arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        this->shared_from_this(),
        move_group_name_);
    RCLCPP_INFO(LOGGER, "MoveIt server ready for group '%s'", move_group_name_.c_str());
}

void MoveitServer::configure_move_group(
    double planning_time,
    double goal_tolerance,
    double orientation_tolerance,
    double velocity_scaling,
    double acceleration_scaling,
    bool clear_constraints)
{
    move_group_arm_->setWorkspace(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
    if (!planner_id_.empty())
    {
        move_group_arm_->setPlannerId(planner_id_);
    }
    move_group_arm_->setNumPlanningAttempts(5);
    move_group_arm_->setPlanningTime(planning_time);
    move_group_arm_->setGoalTolerance(goal_tolerance);
    move_group_arm_->setGoalOrientationTolerance(orientation_tolerance);
    move_group_arm_->setMaxVelocityScalingFactor(velocity_scaling);
    move_group_arm_->setMaxAccelerationScalingFactor(acceleration_scaling);
    if (clear_constraints)
    {
        move_group_arm_->clearPathConstraints();
    }
}

bool MoveitServer::execute_plan(moveit::planning_interface::MoveGroupInterface::Plan &plan)
{
    return move_group_arm_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
}

void MoveitServer::AddScenePlane()
{
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = collision_object_frame_;
    collision_object.id = "ground_plane";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 10.0;
    primitive.dimensions[primitive.BOX_Y] = 10.0;
    primitive.dimensions[primitive.BOX_Z] = 0.01;

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.z = -0.10;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;
    planning_scene_interface_.applyCollisionObject(collision_object);
    RCLCPP_INFO(LOGGER, "Added planning scene ground plane in frame '%s'", collision_object_frame_.c_str());
}

void MoveitServer::set_constraints(const geometry_msgs::msg::Quaternion &quat)
{
    moveit_msgs::msg::Constraints goal_constraints;
    moveit_msgs::msg::OrientationConstraint ori_constraint;
    ori_constraint.header.stamp = this->get_clock()->now();
    ori_constraint.header.frame_id = base_frame_id_;
    ori_constraint.orientation = quat;
    ori_constraint.link_name = end_effector_frame_id_;
    ori_constraint.absolute_x_axis_tolerance = 0.75;
    ori_constraint.absolute_y_axis_tolerance = 0.75;
    ori_constraint.absolute_z_axis_tolerance = 0.75;
    ori_constraint.weight = 1.0;
    ori_constraint.parameterization = moveit_msgs::msg::OrientationConstraint::ROTATION_VECTOR;
    goal_constraints.orientation_constraints.push_back(ori_constraint);
    move_group_arm_->setPathConstraints(goal_constraints);
}

std::vector<double> MoveitServer::extract_joint_targets(
    const sensor_msgs::msg::JointState &target_joints,
    bool &ok,
    std::string &error) const
{
    ok = false;
    std::vector<double> targets;

    if (target_joints.name.empty())
    {
        if (target_joints.position.size() == arm_joint_names_.size())
        {
            ok = true;
            return std::vector<double>(target_joints.position.begin(), target_joints.position.end());
        }
        error = "joint request must include joint names or exactly five arm joint positions";
        return targets;
    }

    if (target_joints.name.size() != target_joints.position.size())
    {
        error = "joint request has mismatched name and position lengths";
        return targets;
    }

    std::unordered_map<std::string, double> request_positions;
    for (size_t index = 0; index < target_joints.name.size(); ++index)
    {
        request_positions[target_joints.name[index]] = target_joints.position[index];
    }

    for (const auto &joint_name : arm_joint_names_)
    {
        const auto position = request_positions.find(joint_name);
        if (position == request_positions.end())
        {
            error = "joint request missing required joint '" + joint_name + "'";
            return {};
        }
        targets.push_back(position->second);
    }

    ok = true;
    return targets;
}

bool MoveitServer::Execute(const sensor_msgs::msg::JointState &target_joints)
{
    bool ok = false;
    std::string error;
    const std::vector<double> joint_targets = extract_joint_targets(target_joints, ok, error);
    if (!ok)
    {
        RCLCPP_ERROR(LOGGER, "%s", error.c_str());
        return false;
    }

    configure_move_group(3.0, 0.003, 0.003, 0.2, 0.4, false);
    move_group_arm_->setStartStateToCurrentState();
    move_group_arm_->setJointValueTarget(joint_targets);

    for (int attempt = 0; attempt < 15; ++attempt)
    {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        const bool planned =
            move_group_arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
        if (planned && execute_plan(plan))
        {
            return true;
        }

        if (attempt == 12)
        {
            move_group_arm_->clearPathConstraints();
            RCLCPP_INFO(LOGGER, "Cleared path constraints for joint-space fallback planning");
        }
    }

    return false;
}

bool MoveitServer::Execute(const geometry_msgs::msg::Pose &target_pose, bool apply_orientation_constraint)
{
    configure_move_group(3.0, 0.005, 0.05, 0.2, 0.4, true);
    move_group_arm_->setStartStateToCurrentState();
    move_group_arm_->clearPoseTargets();

    if (apply_orientation_constraint)
    {
        set_constraints(target_pose.orientation);
        move_group_arm_->setPoseTarget(target_pose);
    }
    else
    {
        // Pure position-only target: ignore orientation entirely
        move_group_arm_->setPositionTarget(target_pose.position.x, target_pose.position.y, target_pose.position.z);
    }

    bool use_position_only_goal = !apply_orientation_constraint;

    const int max_plan_count = 5;
    const double cost_difference_threshold = 0.35;
    std::vector<moveit::planning_interface::MoveGroupInterface::Plan> plans;
    moveit::planning_interface::MoveGroupInterface::Plan best_plan;
    double best_cost = std::numeric_limits<double>::max();
    double best_goal_error = std::numeric_limits<double>::max();
    int similar_cost_count = 0;

    auto robot_state = std::make_shared<moveit::core::RobotState>(move_group_arm_->getRobotModel());

    for (int attempt = 0; attempt < max_plan_count; ++attempt)
    {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        const bool planned =
            move_group_arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;

        if (!planned)
        {
            if (!use_position_only_goal && attempt >= max_plan_count - 2)
            {
                move_group_arm_->clearPathConstraints();
                use_position_only_goal = true;
                RCLCPP_WARN(LOGGER, "Falling back to unconstrained pose target");
            }
            else if (plans.size() < 3 && attempt == max_plan_count - 3)
            {
                move_group_arm_->clearPathConstraints();
            }
            continue;
        }

        plans.push_back(plan);
        double cost = 0.0;
        double goal_error = 0.0;

        if (!plan.trajectory_.joint_trajectory.points.empty())
        {
            const auto &last_point = plan.trajectory_.joint_trajectory.points.back();
            cost = last_point.time_from_start.sec + last_point.time_from_start.nanosec * 1e-9;
            robot_state->setVariablePositions(last_point.positions);
            robot_state->update();
            const auto &ee_pose = robot_state->getGlobalLinkTransform(end_effector_frame_id_);
            const auto &target_position = target_pose.position;
            const auto &ee_position = ee_pose.translation();
            goal_error =
                std::pow(target_position.x - ee_position.x(), 2) +
                std::pow(target_position.y - ee_position.y(), 2) +
                std::pow(target_position.z - ee_position.z(), 2);
        }

        cost += 0.1 * plan.trajectory_.joint_trajectory.points.size();
        if (cost < best_cost ||
            (cost - best_cost < cost_difference_threshold && goal_error < best_goal_error))
        {
            if (best_cost - cost < cost_difference_threshold)
            {
                ++similar_cost_count;
            }
            else
            {
                similar_cost_count = 0;
            }
            best_cost = cost;
            best_goal_error = goal_error;
            best_plan = plan;
        }

        if (similar_cost_count >= 3 || plans.size() >= 3)
        {
            break;
        }
    }

    if (plans.empty())
    {
        RCLCPP_WARN(LOGGER, "No valid plans found for pose target");
        return false;
    }

    const bool executed = execute_plan(best_plan);
    if (executed)
    {
        trajectory_pub_->publish(best_plan.trajectory_.joint_trajectory);
    }
    return executed;
}

void MoveitServer::move_to_pose_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PoseReq::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PoseReq::Response> response)
{
    response->success = Execute(request->object_pose, request->constraint);
}

void MoveitServer::move_to_joint_callback(
    const std::shared_ptr<so101_unified_bringup::srv::JointReq::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::JointReq::Response> response)
{
    response->success = Execute(request->joints);
}

void MoveitServer::trajectory_callback(
    const std::shared_ptr<so101_unified_bringup::srv::ActionTraj::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::ActionTraj::Response> response)
{
    moveit_msgs::msg::RobotTrajectory robot_trajectory;
    robot_trajectory.joint_trajectory = request->traj;

    configure_move_group(3.5, 0.003, 0.003, 0.2, 0.4, false);
    move_group_arm_->setStartStateToCurrentState();
    response->success =
        move_group_arm_->execute(robot_trajectory) == moveit::core::MoveItErrorCode::SUCCESS;
}

void MoveitServer::rotate_effector_callback(
    const std::shared_ptr<so101_unified_bringup::srv::RotateEffector::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::RotateEffector::Response> response)
{
    configure_move_group(2.0, 0.005, 0.005, 0.4, 0.4, true);

    std::vector<double> target_joint_values;
    target_joint_values.reserve(arm_joint_names_.size());

    for (const auto &joint_name : arm_joint_names_)
    {
        const auto position = current_joint_positions_.find(joint_name);
        if (position == current_joint_positions_.end())
        {
            RCLCPP_ERROR(LOGGER, "Missing current state for joint '%s'", joint_name.c_str());
            response->success = false;
            return;
        }
        target_joint_values.push_back(position->second);
    }

    const auto wrist_it =
        std::find(arm_joint_names_.begin(), arm_joint_names_.end(), wrist_roll_joint_name_);
    if (wrist_it == arm_joint_names_.end())
    {
        RCLCPP_ERROR(LOGGER, "Configured wrist roll joint '%s' is not in the arm joint list", wrist_roll_joint_name_.c_str());
        response->success = false;
        return;
    }

    const auto wrist_index = static_cast<size_t>(std::distance(arm_joint_names_.begin(), wrist_it));
    target_joint_values[wrist_index] += request->rotation_angle;

    for (int attempt = 0; attempt < 5; ++attempt)
    {
        move_group_arm_->setJointValueTarget(target_joint_values);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        const bool planned =
            move_group_arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
        if (planned && execute_plan(plan))
        {
            trajectory_pub_->publish(plan.trajectory_.joint_trajectory);
            response->success = true;
            return;
        }
    }

    response->success = false;
}

void MoveitServer::sync_callback(
    const std::shared_ptr<so101_unified_bringup::srv::JointSat::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::JointSat::Response> response)
{
    configure_move_group(2.0, 0.001, 0.001, 0.7, 0.7, true);

    bool ok = false;
    std::string error;
    const std::vector<double> joint_targets = extract_joint_targets(request->joint_state, ok, error);
    if (!ok)
    {
        RCLCPP_ERROR(LOGGER, "%s", error.c_str());
        response->successq = false;
        return;
    }

    move_group_arm_->setJointValueTarget(joint_targets);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned =
        move_group_arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    response->successq = planned && execute_plan(plan);
}

void MoveitServer::joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    const size_t count = std::min(msg->name.size(), msg->position.size());
    for (size_t index = 0; index < count; ++index)
    {
        current_joint_positions_[msg->name[index]] = msg->position[index];
    }
}

geometry_msgs::msg::Pose MoveitServer::rotated_pose(
    const geometry_msgs::msg::Pose &pose,
    double roll_delta,
    double pitch_delta,
    double yaw_delta) const
{
    tf2::Quaternion q(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w);
    tf2::Matrix3x3 matrix(q);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    matrix.getRPY(roll, pitch, yaw);
    roll += roll_delta;
    pitch += pitch_delta;
    yaw += yaw_delta;

    tf2::Quaternion rotated;
    rotated.setRPY(roll, pitch, yaw);

    geometry_msgs::msg::Pose output = pose;
    output.orientation.x = rotated.x();
    output.orientation.y = rotated.y();
    output.orientation.z = rotated.z();
    output.orientation.w = rotated.w();
    return output;
}

bool MoveitServer::move_pose_with_message(
    const geometry_msgs::msg::Pose &target_pose,
    std::string &error_message,
    const std::string &failure_context)
{
    if (Execute(target_pose))
    {
        return true;
    }
    error_message = failure_context;
    RCLCPP_ERROR(LOGGER, "%s", failure_context.c_str());
    return false;
}

void MoveitServer::simulate_gripper_action(bool grip_state) const
{
    RCLCPP_INFO(LOGGER, "Simulating gripper action: %s", grip_state ? "close" : "open");
    rclcpp::sleep_for(std::chrono::milliseconds(500));
}

void MoveitServer::place_object_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PlaceObject::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PlaceObject::Response> response)
{
    if (!move_pose_with_message(request->target_pose, response->message, "Failed to move to target pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);
    response->success = true;
    response->message = "Place object operation completed successfully";
}

void MoveitServer::pick_object_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PickObject::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PickObject::Response> response)
{
    if (!move_pose_with_message(request->target_pose, response->message, "Failed to move to pick pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);
    response->success = true;
    response->message = "Pick object operation completed successfully";
}

void MoveitServer::pick_front_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PickFront::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PickFront::Response> response)
{
    auto pick_pose = rotated_pose(request->target_pose, 0.0, -1.57, 0.0);
    pick_pose.position.x -= 0.12;
    if (!move_pose_with_message(pick_pose, response->message, "Failed to move to pick pose"))
    {
        response->success = false;
        return;
    }

    auto lowered_pose = pick_pose;
    lowered_pose.position.x += 0.045;
    if (!move_pose_with_message(lowered_pose, response->message, "Failed to move to lowered pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);

    auto safe_pose = pick_pose;
    safe_pose.position.x -= 0.05;
    if (!move_pose_with_message(safe_pose, response->message, "Failed to move to safe pose"))
    {
        response->success = false;
        return;
    }

    response->success = true;
    response->message = "Pick front operation completed successfully";
}

void MoveitServer::pick_right_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PickRight::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PickRight::Response> response)
{
    auto pick_pose = rotated_pose(request->target_pose, -1.57, 1.57, 0.0);
    pick_pose.position.y += 0.12;
    if (!move_pose_with_message(pick_pose, response->message, "Failed to move to pick pose"))
    {
        response->success = false;
        return;
    }

    auto lowered_pose = pick_pose;
    lowered_pose.position.y -= 0.045;
    if (!move_pose_with_message(lowered_pose, response->message, "Failed to move to lowered pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);

    auto safe_pose = pick_pose;
    safe_pose.position.z += 0.05;
    if (!move_pose_with_message(safe_pose, response->message, "Failed to move to safe pose"))
    {
        response->success = false;
        return;
    }

    response->success = true;
    response->message = "Pick right operation completed successfully";
}

void MoveitServer::pick_left_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PickLeft::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PickLeft::Response> response)
{
    auto pick_pose = rotated_pose(request->target_pose, -4.71, -1.57, 0.0);
    pick_pose.position.y -= 0.12;
    if (!move_pose_with_message(pick_pose, response->message, "Failed to move to pick pose"))
    {
        response->success = false;
        return;
    }

    auto lowered_pose = pick_pose;
    lowered_pose.position.y += 0.05;
    if (!move_pose_with_message(lowered_pose, response->message, "Failed to move to lowered pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);

    auto safe_pose = pick_pose;
    safe_pose.position.z += 0.045;
    if (!move_pose_with_message(safe_pose, response->message, "Failed to move to safe pose"))
    {
        response->success = false;
        return;
    }

    response->success = true;
    response->message = "Pick left operation completed successfully";
}

void MoveitServer::pick_rear_callback(
    const std::shared_ptr<so101_unified_bringup::srv::PickRear::Request> request,
    std::shared_ptr<so101_unified_bringup::srv::PickRear::Response> response)
{
    auto pick_pose = rotated_pose(request->target_pose, 0.0, 1.57, 0.0);
    pick_pose.position.x += 0.10;
    if (!move_pose_with_message(pick_pose, response->message, "Failed to move to pick pose"))
    {
        response->success = false;
        return;
    }

    auto lowered_pose = pick_pose;
    lowered_pose.position.x -= 0.03;
    if (!move_pose_with_message(lowered_pose, response->message, "Failed to move to lowered pose"))
    {
        response->success = false;
        return;
    }

    simulate_gripper_action(request->grip_state);

    auto safe_pose = pick_pose;
    safe_pose.position.z += 0.05;
    if (!move_pose_with_message(safe_pose, response->message, "Failed to move to safe pose"))
    {
        response->success = false;
        return;
    }

    response->success = true;
    response->message = "Pick rear operation completed successfully";
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto planner_node = std::make_shared<MoveitServer>();
    planner_node->initialize_move_group();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(planner_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
