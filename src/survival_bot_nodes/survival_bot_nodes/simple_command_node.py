#!/usr/bin/env python3
"""
Simple Command Interface - Send commands to robot
Use this to manually send turn and stop commands
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleCommandNode(Node):
    def __init__(self):
        super().__init__('simple_command_node')
        
        # Publisher for robot commands
        self.command_pub = self.create_publisher(String, 'robot/command', 10)
        
        self.get_logger().info("🎮 Simple Command Interface Started")
        self.get_logger().info("   Use 'ros2 topic pub' to send commands:")
        self.get_logger().info("   ros2 topic pub /robot/command std_msgs/String 'data: \"TURN,45\"'")
        self.get_logger().info("   ros2 topic pub /robot/command std_msgs/String 'data: \"STOP\"'")
        
        # Create a service or topic subscriber for commands (optional)
        # For now, this node just publishes the topic and provides examples
        
        # Send initial status
        self.send_command("STATUS")
    
    def send_command(self, command):
        """Send command to robot"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.get_logger().info(f"📤 Command: {command}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleCommandNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Simple Command Interface stopped")
        # Send stop command
        if hasattr(node, 'send_command'):
            node.send_command("STOP")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 