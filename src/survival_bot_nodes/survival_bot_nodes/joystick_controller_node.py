#!/usr/bin/env python3
"""
Joystick Controller Node - Manual robot control with PWM support
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
        
        # Control mode: 'discrete' or 'pwm'
        self.control_mode = 'pwm'  # Use PWM for smoother control
        
        # Previous command to avoid spamming
        self.last_command = ""
        
        # Timer for joystick polling
        self.create_timer(0.1, self.joystick_callback)
        
        self.get_logger().info("🎮 Joystick Controller Node Started")
        self.get_logger().info(f"   Control mode: {self.control_mode}")
        self.get_logger().info("   Use joystick to control robot manually")
        self.get_logger().info("   Left stick: Movement, Buttons: Mode switch")
    
    def joystick_callback(self):
        """Poll joystick for input"""
        if not self.joystick:
            return
        
        pygame.event.pump()
        
        # Get joystick values
        left_stick_x = self.joystick.get_axis(0)  # Left/right
        left_stick_y = self.joystick.get_axis(1)  # Forward/back
        
        # Check buttons for mode switching
        try:
            # Button 0 (usually X or A) - switch to discrete mode
            if self.joystick.get_button(0):
                if self.control_mode != 'discrete':
                    self.control_mode = 'discrete'
                    self.get_logger().info("🔄 Switched to discrete control mode")
            
            # Button 1 (usually O or B) - switch to PWM mode  
            if self.joystick.get_button(1):
                if self.control_mode != 'pwm':
                    self.control_mode = 'pwm'
                    self.get_logger().info("🔄 Switched to PWM control mode")
        except:
            pass  # Some controllers might not have these buttons
        
        # Apply deadzone
        deadzone = 0.15
        if abs(left_stick_x) < deadzone:
            left_stick_x = 0.0
        if abs(left_stick_y) < deadzone:
            left_stick_y = 0.0
        
        # Generate command based on mode
        if self.control_mode == 'pwm':
            self.handle_pwm_control(left_stick_x, left_stick_y)
        else:
            self.handle_discrete_control(left_stick_x, left_stick_y)
    
    def handle_pwm_control(self, stick_x, stick_y):
        """Handle continuous PWM control"""
        # Convert stick inputs to motor PWM values (-255 to 255)
        max_pwm = 255
        
        # Calculate base forward/backward movement
        forward_pwm = -stick_y * max_pwm  # Invert Y axis
        
        # Calculate turning component
        turn_pwm = stick_x * max_pwm * 0.7  # Reduce turn sensitivity
        
        # Calculate left and right motor PWM
        left_pwm = int(forward_pwm + turn_pwm)
        right_pwm = int(forward_pwm - turn_pwm)
        
        # Clamp values
        left_pwm = max(-max_pwm, min(max_pwm, left_pwm))
        right_pwm = max(-max_pwm, min(max_pwm, right_pwm))
        
        # Create command
        if left_pwm == 0 and right_pwm == 0:
            command = "STOP"
        else:
            command = f"PWM,{left_pwm},{right_pwm}"
        
        # Only send if different from last command
        if command != self.last_command:
            self.send_command(command)
            self.last_command = command
    
    def handle_discrete_control(self, stick_x, stick_y):
        """Handle discrete movement commands"""
        command = ""
        
        # Prioritize forward/backward over turning
        if abs(stick_y) > 0.5:
            if stick_y < -0.5:  # Forward (stick up)
                command = "FORWARD,1.0"
            elif stick_y > 0.5:  # Backward (stick down)
                command = "BACKWARD,1.0"
        elif abs(stick_x) > 0.5:
            if stick_x < -0.5:  # Turn left
                command = "TURN_LEFT"
            elif stick_x > 0.5:  # Turn right
                command = "TURN_RIGHT"
        else:
            # Stop if no significant input
            command = "STOP"
        
        # Only send if different from last command
        if command != self.last_command:
            self.send_command(command)
            self.last_command = command
    
    def send_command(self, command):
        """Send command to robot"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        
        # Log significant commands
        if command != "STOP" and not command.startswith("PWM,0,0"):
            self.get_logger().info(f"🎮 {command}")

def main(args=None):
    rclpy.init(args=args)
    
    if not PYGAME_AVAILABLE:
        print("❌ Cannot start joystick controller - pygame not available")
        print("   Install with: pip install pygame")
        return
    
    node = JoystickControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Send stop command on exit
        if hasattr(node, 'send_command'):
            node.send_command("STOP")
        print("\n🛑 Joystick controller stopped")
    finally:
        if hasattr(node, 'destroy_node'):
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 