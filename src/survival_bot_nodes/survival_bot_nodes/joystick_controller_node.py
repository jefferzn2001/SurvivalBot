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
        
        # Timer for joystick polling - FASTER RATE
        self.create_timer(0.05, self.joystick_callback)  # 20Hz instead of 10Hz
        
        self.get_logger().info("🎮 Joystick Controller Node Started")
        self.get_logger().info(f"   Control mode: {self.control_mode}")
        self.get_logger().info("   Left stick: Movement, LT/RT: Turn left/right")
        self.get_logger().info("   Buttons: Mode switch")
    
    def joystick_callback(self):
        """Poll joystick for input"""
        if not self.joystick:
            return
        
        pygame.event.pump()
        
        # Get joystick values
        left_stick_x = self.joystick.get_axis(0)  # Left/right
        left_stick_y = self.joystick.get_axis(1)  # Forward/back
        
        # Get trigger values (LT = left trigger, RT = right trigger)
        try:
            # Xbox controller: LT = axis 2, RT = axis 5 (or different depending on controller)
            # PlayStation: LT = axis 3, RT = axis 4
            # Try common trigger mappings
            lt_value = 0.0
            rt_value = 0.0
            
            # Try different controller layouts
            if self.joystick.get_numaxes() >= 6:
                # Xbox controller layout
                lt_value = (self.joystick.get_axis(2) + 1) / 2  # Convert from -1,1 to 0,1
                rt_value = (self.joystick.get_axis(5) + 1) / 2  # Convert from -1,1 to 0,1
            elif self.joystick.get_numaxes() >= 4:
                # Alternative layout
                lt_value = max(0, self.joystick.get_axis(2))  # Only positive values
                rt_value = max(0, self.joystick.get_axis(3))  # Only positive values
        except:
            lt_value = 0.0
            rt_value = 0.0
        
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
        
        # Apply deadzone to stick inputs
        deadzone = 0.15
        if abs(left_stick_x) < deadzone:
            left_stick_x = 0.0
        if abs(left_stick_y) < deadzone:
            left_stick_y = 0.0
        
        # Apply deadzone to triggers
        trigger_deadzone = 0.1
        if lt_value < trigger_deadzone:
            lt_value = 0.0
        if rt_value < trigger_deadzone:
            rt_value = 0.0
        
        # Generate command based on mode
        if self.control_mode == 'pwm':
            self.handle_pwm_control(left_stick_x, left_stick_y, lt_value, rt_value)
        else:
            self.handle_discrete_control(left_stick_x, left_stick_y, lt_value, rt_value)
    
    def handle_pwm_control(self, stick_x, stick_y, lt_value, rt_value):
        """Handle continuous PWM control with trigger turning"""
        # Convert stick inputs to motor PWM values (-255 to 255)
        max_pwm = 255
        trigger_turn_pwm = 150  # Exact PWM for trigger turning
        
        # Priority: Triggers override ALL other movement
        if lt_value > 0 or rt_value > 0:
            # Pure trigger turning - ignore stick inputs
            if lt_value > rt_value:
                # Left trigger pressed - turn left (FIXED AGAIN: swapped back)
                left_pwm = -trigger_turn_pwm  # -150
                right_pwm = trigger_turn_pwm  # +150
            else:
                # Right trigger pressed - turn right (FIXED AGAIN: swapped back)
                left_pwm = trigger_turn_pwm   # +150
                right_pwm = -trigger_turn_pwm # -150
        else:
            # Normal stick control when no triggers pressed
            # Calculate base forward/backward movement from left stick Y
            forward_pwm = -stick_y * max_pwm  # Invert Y axis
            
            # Calculate turning component from stick X (FIXED: flipped for correct turning)
            turn_pwm = -stick_x * max_pwm * 0.7  # Flipped sign and reduce turn sensitivity
            
            # Calculate left and right motor PWM
            left_pwm = int(forward_pwm + turn_pwm)
            right_pwm = int(forward_pwm - turn_pwm)
            
            # Clamp values
            left_pwm = max(-max_pwm, min(max_pwm, left_pwm))
            right_pwm = max(-max_pwm, min(max_pwm, right_pwm))
            
            # Reset to 0 if joystick is centered
            if abs(stick_x) == 0.0 and abs(stick_y) == 0.0:
                left_pwm = 0
                right_pwm = 0
        
        # HARDWARE FIX: Remap negative PWM values for correct hardware operation
        # Hardware expects: -1 = fastest, -255 = slowest (inverted from normal)
        # ONLY remap first value (right wheels), keep second value (left wheels) unchanged
        def remap_first_value_only(pwm_value):
            if pwm_value < 0:
                # Remap -255 to -1 (fastest) and -1 to -255 (slowest)
                return -(256 - abs(pwm_value))
            return pwm_value
        
        # Apply remapping ONLY to right wheels (first PWM value)
        remapped_right = remap_first_value_only(right_pwm)
        # Keep left wheels unchanged
        remapped_left = left_pwm
        
        # DIRECTIONAL FIX: When both PWM values are positive, swap them
        if remapped_right > 0 and remapped_left > 0:
            # Swap for positive values
            command = f"PWM,{remapped_left},{remapped_right}"
        else:
            # Keep normal order for negative values (working correctly)
            command = f"PWM,{remapped_right},{remapped_left}"
        
        # ALWAYS send command for immediate response
        self.send_command(command)
        self.last_command = command
    
    def handle_discrete_control(self, stick_x, stick_y, lt_value, rt_value):
        """Handle discrete movement commands with trigger turning"""
        command = ""
        
        # Check triggers first (higher priority)
        if lt_value > 0.5:
            command = "TURN_LEFT"
        elif rt_value > 0.5:
            command = "TURN_RIGHT"
        # Then check stick inputs
        elif abs(stick_y) > 0.5:
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
        
        # ALWAYS log PWM commands to see responsiveness
        if command.startswith("PWM,"):
            self.get_logger().info(f"🎮 {command}")
        elif command != "STOP":
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