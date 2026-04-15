#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import json

class FoxgloveEnvPublisher(Node):
    """
    Listens to the JSON object detections and publishes them as 3D geometric markers
    so Foxglove Studio can render the environment interactively on the web.
    """
    def __init__(self):
        super().__init__('foxglove_env_publisher')
        
        # Subscribe to the mock/real JSON detector output
        self.sub = self.create_subscription(
            String,
            '/detected_objects_3d',
            self.det_callback,
            10
        )
        
        # Publish Foxglove-native visual markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/foxglove/environment_markers',
            10
        )
        
        self.get_logger().info("Foxglove Environment Marker publisher started.")

    def det_callback(self, msg):
        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        marker_array = MarkerArray()
        
        # 1. Always publish the table permanently underneath
        table = Marker()
        table.header.frame_id = 'base_link'
        table.header.stamp = self.get_clock().now().to_msg()
        table.ns = 'environment'
        table.id = 0
        table.type = Marker.CUBE
        table.action = Marker.ADD
        table.pose.position.x = -0.3
        table.pose.position.y = 0.0
        table.pose.position.z = 0.775 / 2.0  # Center is half height
        table.pose.orientation.w = 1.0
        table.scale.x = 1.0
        table.scale.y = 1.5
        table.scale.z = 0.775  # Height of table
        table.color.r = 0.6
        table.color.g = 0.4
        table.color.b = 0.2
        table.color.a = 1.0
        marker_array.markers.append(table)
        
        # 2. Publish dynamically detected objects
        for i, obj in enumerate(detections):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'objects'
            m.id = i + 1
            
            # Read shape and pose
            shape = obj.get('shape', 'cylinder')
            if shape == 'cylinder':
                m.type = Marker.CYLINDER
            else:
                m.type = Marker.CUBE
                
            m.action = Marker.ADD
            pose = obj.get('pose', [0, 0, 0, 0, 0, 0])
            m.pose.position.x = float(pose[0])
            m.pose.position.y = float(pose[1])
            m.pose.position.z = float(pose[2])
            # Assuming simple upright objects for now
            m.pose.orientation.w = 1.0
            
            # Dimensions
            radius = obj.get('radius', 0.03)
            height = obj.get('height', 0.1)
            
            # User request: "put a box to know how to pick it"
            # We override the actual geometry to display as a distinct bounding box (CUBE)
            m.type = Marker.CUBE
            m.scale.x = radius * 2.5  # Slightly wider bounding box
            m.scale.y = radius * 2.5
            m.scale.z = height + 0.02
            
            # Colors logically distinct
            if 'bottle' in obj.get('name', '').lower():
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 0.3  # Translucent green bounding box
            else:
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.3  # Translucent red bounding box
                
            marker_array.markers.append(m)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = FoxgloveEnvPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
