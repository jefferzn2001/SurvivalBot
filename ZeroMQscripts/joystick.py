#!/usr/bin/env python3
"""
Joystick Controller - Manual robot control with PWM support
Uses ZeroMQ data server instead of ROS2
"""

import time
import sys
import logging

# Import data clients
sys.path.append('data')
from command_client import CommandClient

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JoystickController:
    def __init__(self, server_ip="10.102.200.37"):
        if not PYGAME_AVAILABLE:
            logger.error("❌ pygame not available - install with: pip install pygame")
            return
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Enhanced joystick detection
        joystick_count = pygame.joystick.get_count()
        logger.info(f"🔍 Found {joystick_count} joystick(s)")
        
        if joystick_count == 0:
            logger.error("❌ No joystick detected by pygame")
            logger.info("   Try: ls /dev/input/js* or ls /dev/input/event*")
            self.joystick = None
        else:
            # List all detected joysticks and find the game controller
            game_controller_index = None
            for i in range(joystick_count):
                joy = pygame.joystick.Joystick(i)
                joy.init()
                name = joy.get_name()
                logger.info(f"   Joystick {i}: {name}")
                
                # Look for actual game controllers (not keyboards/mice)
                if any(keyword in name.lower() for keyword in ['controller', 'gamepad', 'xbox', 'playstation', '8bitdo']):
                    if game_controller_index is None:  # Use the first game controller found
                        game_controller_index = i
                        logger.info(f"   → Selected as game controller")
                joy.quit()
            
            # Use game controller or fallback to joystick 0
            if game_controller_index is not None:
                controller_index = game_controller_index
                logger.info(f"🎮 Using game controller at index {controller_index}")
            else:
                controller_index = 0
                logger.warning(f"⚠️ No game controller found, using joystick 0")
            
            # Initialize the selected controller
            self.joystick = pygame.joystick.Joystick(controller_index)
            self.joystick.init()
            logger.info(f"🎮 Active controller: {self.joystick.get_name()}")
            logger.info(f"   Axes: {self.joystick.get_numaxes()}")
            logger.info(f"   Buttons: {self.joystick.get_numbuttons()}")
            logger.info(f"   Hats: {self.joystick.get_numhats()}")
        
        # Setup command client
        self.command_client = CommandClient(server_ip=server_ip)
        
        # Control mode: 'discrete' or 'pwm'
        self.control_mode = 'pwm'  # Use PWM for smoother control
        
        # Previous command to avoid spamming
        self.last_command = ""
        
        # Running state
        self.running = False
        
        logger.info("🎮 Joystick Controller Started")
        logger.info(f"   Control mode: {self.control_mode}")
        logger.info("   Left stick: Movement (up=forward, left=turn left)")
        logger.info("   LT=Turn left, RT=Turn right")
        logger.info("   PWM format: PWM,right_wheels,left_wheels")
        logger.info("   FIXED: Stick left → robot turns left, LT → left turn")
    
    def joystick_callback(self):
        """Poll joystick for input"""
        if not self.joystick:
            return
        
        try:
            pygame.event.pump()
            
            # Get joystick values
            left_stick_x = self.joystick.get_axis(0)  # Left/right
            left_stick_y = self.joystick.get_axis(1)  # Forward/back
            
            # Get trigger values (LT = left trigger, RT = right trigger)
            try:
                # 8BitDo and similar controllers use different mappings
                lt_value = 0.0
                rt_value = 0.0
                
                # Try different controller layouts based on number of axes
                if self.joystick.get_numaxes() >= 6:
                    # Most controllers: Try axis 2 and 5 first (Xbox style)
                    lt_raw = self.joystick.get_axis(2)  # Often LT
                    rt_raw = self.joystick.get_axis(5)  # Often RT
                    
                    # 8BitDo style: triggers are 0-1 range, not -1 to 1
                    if lt_raw >= 0 and rt_raw >= 0:
                        # Triggers are already 0-1 range
                        lt_value = lt_raw
                        rt_value = rt_raw
                    else:
                        # Xbox style: Convert -1,1 to 0,1
                        lt_value = (lt_raw + 1) / 2
                        rt_value = (rt_raw + 1) / 2
                        
                elif self.joystick.get_numaxes() >= 4:
                    # Alternative layout for older controllers
                    lt_value = max(0, self.joystick.get_axis(2))
                    rt_value = max(0, self.joystick.get_axis(3))
                
            except Exception as e:
                logger.error(f"Error reading triggers: {e}")
                lt_value = 0.0
                rt_value = 0.0
            
            # Check buttons for mode switching
            try:
                # Button 0 (usually X or A) - switch to discrete mode
                if self.joystick.get_button(0):
                    if self.control_mode != 'discrete':
                        self.control_mode = 'discrete'
                        logger.info("🔄 Switched to discrete control mode")
                
                # Button 1 (usually O or B) - switch to PWM mode  
                if self.joystick.get_button(1):
                    if self.control_mode != 'pwm':
                        self.control_mode = 'pwm'
                        logger.info("🔄 Switched to PWM control mode")
            except:
                pass  # Some controllers might not have these buttons
            
            # Apply deadzone to stick inputs
            deadzone = 0.05  # Reduce deadzone to be more sensitive
            if abs(left_stick_x) < deadzone:
                left_stick_x = 0.0
            if abs(left_stick_y) < deadzone:
                left_stick_y = 0.0
            
            # Apply deadzone to triggers
            trigger_deadzone = 0.1  # Reduce back to 0.1 for 8BitDo controllers that read 0.0 at rest
            if lt_value < trigger_deadzone:
                lt_value = 0.0
            if rt_value < trigger_deadzone:
                rt_value = 0.0
            
            # Generate command based on mode
            if self.control_mode == 'pwm':
                self.handle_pwm_control(left_stick_x, left_stick_y, lt_value, rt_value)
            else:
                self.handle_discrete_control(left_stick_x, left_stick_y, lt_value, rt_value)
                
        except Exception as e:
            logger.error(f"❌ Exception in joystick callback: {e}")
    
    def handle_pwm_control(self, stick_x, stick_y, lt_value, rt_value):
        """Handle continuous PWM control with trigger turning - FIXED DIRECTIONS"""
        # Convert stick inputs to motor PWM values (-255 to 255)
        max_pwm = 255
        trigger_turn_pwm = 150   # Trigger turn PWM (symmetric for both LT and RT)
        
        # Priority: Triggers override ALL other movement
        if lt_value > 0 or rt_value > 0:
            # Pure trigger turning - ignore stick inputs
            if lt_value > rt_value:
                # Left trigger pressed - turn left (left wheels slower/backward, right wheels faster/forward)
                right_pwm = trigger_turn_pwm   # Right wheels forward (150)
                left_pwm = -trigger_turn_pwm   # Left wheels backward (-150)
            else:
                # Right trigger pressed - turn right (right wheels slower/backward, left wheels faster/forward)
                right_pwm = -trigger_turn_pwm  # Right wheels backward (-150)
                left_pwm = trigger_turn_pwm    # Left wheels forward (150)
                
            # Clamp trigger values
            right_pwm = max(-max_pwm, min(max_pwm, right_pwm))
            left_pwm = max(-max_pwm, min(max_pwm, left_pwm))
        else:
            # Normal stick control when no triggers pressed
            # IMPORTANT: Check if stick is actually centered FIRST
            if abs(stick_x) < 0.001 and abs(stick_y) < 0.001:
                # Stick is perfectly centered - send zero PWM
                right_pwm = 0
                left_pwm = 0
            else:
                # Calculate base forward/backward movement from left stick Y
                forward_pwm = -stick_y * max_pwm  # Invert Y axis (up = forward)
            
                # Calculate turning component from stick X
                # FIXED: When stick goes left (negative), we want to turn left
                # Left turn = left wheels slower, right wheels faster
                turn_pwm = stick_x * max_pwm * 0.7  # Remove negative sign to fix direction
            
                # Calculate left and right motor PWM - FIXED LOGIC
                # When stick_x is negative (left), turn_pwm is negative
                # right_pwm gets MORE power (forward_pwm - negative = forward_pwm + positive)
                # left_pwm gets LESS power (forward_pwm + negative = forward_pwm - positive)
                right_pwm = int(forward_pwm - turn_pwm)  # Right wheels
                left_pwm = int(forward_pwm + turn_pwm)   # Left wheels
            
                # Clamp values
                right_pwm = max(-max_pwm, min(max_pwm, right_pwm))
                left_pwm = max(-max_pwm, min(max_pwm, left_pwm))
        
        # Standard PWM command: PWM,right_wheels,left_wheels (no remapping needed)
        command = f"PWM,{right_pwm},{left_pwm}"
        
        # Send command if different from last command
        if command != self.last_command:
            self.send_command(command)
            self.last_command = command
    
    def handle_discrete_control(self, stick_x, stick_y, lt_value, rt_value):
        """Handle discrete movement commands with trigger turning - FIXED DIRECTIONS"""
        command = ""
        
        # Check triggers first (higher priority)
        if lt_value > 0.5:
            command = "TURN,-90"  # Left trigger = turn left
        elif rt_value > 0.5:
            command = "TURN,90"   # Right trigger = turn right
        # Then check stick inputs
        elif abs(stick_y) > 0.5:
            if stick_y < -0.5:  # Forward (stick up)
                command = "FORWARD,1.0"
            elif stick_y > 0.5:  # Backward (stick down)
                command = "BACKWARD,1.0"
        elif abs(stick_x) > 0.5:
            if stick_x < -0.5:  # Turn left (stick left)
                command = "TURN,-90"
            elif stick_x > 0.5:  # Turn right (stick right)
                command = "TURN,90"
        else:
            # Stop if no significant input
            command = "STOP"
        
        # Only send if different from last command
        if command != self.last_command:
            self.send_command(command)
            self.last_command = command
    
    def send_command(self, command):
        """Send command to robot"""
        self.command_client.send_command(command)
        
        # ALWAYS log PWM commands to see responsiveness
        if command.startswith("PWM,"):
            logger.info(f"🎮 {command}")
        elif command != "STOP":
            logger.info(f"🎮 {command}")
    
    def run(self):
        """Main run loop"""
        if not self.joystick:
            logger.error("❌ Cannot run - no joystick available")
            return
        
        self.running = True
        logger.info("🎮 Joystick controller running - press Ctrl+C to stop")
        
        try:
            while self.running:
                self.joystick_callback()
                time.sleep(0.05)  # 20Hz polling rate
                
        except KeyboardInterrupt:
            logger.info("🛑 Joystick controller stopped by user")
        finally:
            self.command_client.stop()
            self.command_client.close()
            if self.joystick:
                self.joystick.quit()
            pygame.quit()

def main():
    import argparse
    
    if not PYGAME_AVAILABLE:
        print("❌ Cannot start joystick controller - pygame not available")
        print("   Install with: pip install pygame")
        return
    
    parser = argparse.ArgumentParser(description='Joystick Controller with Data Server')
    parser.add_argument('--server-ip', type=str, default='10.102.200.37',
                      help='IP address of the robot running data_server.py')
    args = parser.parse_args()
    
    controller = JoystickController(server_ip=args.server_ip)
    controller.run()

if __name__ == '__main__':
    main() 