#!/usr/bin/env python3
"""
Joystick Controller Node - Manual robot control
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class JoystickControllerNode(Node):
    def __init__(self):
        super().__init__('joystick_controller_node')
        
        if not PYGAME_AVAILABLE:
            self.get_logger().error("❌ pygame not available - install with: pip install pygame")
            return
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            self.get_logger().warning("⚠️ No joystick detected")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.get_logger().info(f"🎮 Joystick connected: {self.joystick.get_name()}")
        
        # Publisher
        self.command_pub = self.create_publisher(String, 'robot/command', 10)
        
        # Timer for joystick polling
        self.create_timer(0.1, self.joystick_callback)
        
        self.get_logger().info("🎮 Joystick Controller Node Started")
        self.get_logger().info("   Use joystick to control robot manually")
    
    def joystick_callback(self):
        """Poll joystick for input"""
        if not self.joystick:
            return
        
        pygame.event.pump()
        
        # Get joystick values
        left_stick_x = self.joystick.get_axis(0)  # Left/right
        left_stick_y = self.joystick.get_axis(1)  # Forward/back
        
        # Simple movement commands
        if abs(left_stick_y) > 0.5:
            if left_stick_y < -0.5:  # Forward
                self.send_command("FORWARD,1.0")
            elif left_stick_y > 0.5:  # Backward
                self.send_command("BACKWARD,1.0")
        elif abs(left_stick_x) > 0.5:
            if left_stick_x < -0.5:  # Turn left
                self.send_command("TURN,-30")
            elif left_stick_x > 0.5:  # Turn right
                self.send_command("TURN,30")
        else:
            # Stop if no input
            self.send_command("STOP")
    
    def send_command(self, command):
        """Send command to robot"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    if not PYGAME_AVAILABLE:
        print("❌ Cannot start joystick controller - pygame not available")
        return
    
    node = JoystickControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 