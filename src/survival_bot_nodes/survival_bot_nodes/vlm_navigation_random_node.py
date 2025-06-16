#!/usr/bin/env python3
"""
VLM Navigation Random Node - VLM navigation with random distance -1 to 3m variation
Implements proper data collection sequence with monitoring
"""

import rclpy
from rclpy.node import Node
import json
import time
import os
import sys
import cv2
import numpy as np
import random
from datetime import datetime
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# Import VLMNAV components safely
try:
    # Import functions without triggering serial connection
    import sys
    import os
    import importlib.util
    
    # Load modules manually to avoid main.py serial initialization
    vlmnav_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'SurvivalBot', 'src', 'survival_bot_nodes', 'VLMNAV'))
    
    # Import annotation module
    annotation_spec = importlib.util.spec_from_file_location("annotation", os.path.join(vlmnav_path, "annotation.py"))
    annotation_module = importlib.util.module_from_spec(annotation_spec)
    annotation_spec.loader.exec_module(annotation_module)
    annotate_image = annotation_module.annotate_image
    
    # Import prompt module
    prompt_spec = importlib.util.spec_from_file_location("prompt", os.path.join(vlmnav_path, "prompt.py"))
    prompt_module = importlib.util.module_from_spec(prompt_spec)
    prompt_spec.loader.exec_module(prompt_module)
    construct_action_prompt = prompt_module.construct_action_prompt
    
    # Import Gemini API directly
    import google.generativeai as genai
    from dotenv import load_dotenv
    import base64
    import json
    import re
    
    # Load API key
    load_dotenv(os.path.join(vlmnav_path, '.env'))
    api_key = os.getenv("API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        VLM_READY = True
        print("✅ Gemini API configured")
    else:
        VLM_READY = False
        print("❌ No API key found")
        
except ImportError as e:
    print(f"❌ VLMNAV import failed: {e}")
    VLM_READY = False

# VLM functions (copied from main.py to avoid serial import)
def generate_response(image_path, goal, turn_around_available):
    """Generate a response based on the image and goal."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        num_actions = 5
        prompt = construct_action_prompt(goal, num_actions, turn_around_available)
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": encoded_image},
            prompt
        ])
        
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def parse_action(response):
    """Extract action number from response."""
    try:
        content = response.text
        match = re.search(r'\{.*?\}', content)
        if match:
            action_json = match.group(0)
            action_json = action_json.replace("'", '"')
            action_dict = json.loads(action_json)
            return int(action_dict.get("action", -1))
    except Exception as e:
        print(f"Error parsing action: {e}")
    return -1

class VLMNavigationRandomNode(Node):
    def __init__(self):
        super().__init__('vlm_navigation_random_node')
        
        if not VLM_READY:
            self.get_logger().error("❌ VLM not ready! Check VLMNAV setup and API key")
            return
        
        # Parameters
        self.declare_parameter('goal', 'Max Sunlight Location')
        self.declare_parameter('max_iterations', 10.0)
        self.declare_parameter('navigation_interval', 15.0)  # Longer interval for data collection sequence
        self.declare_parameter('vlm_session_dir', '')  # Shared VLM session directory
        
        self.goal = self.get_parameter('goal').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.navigation_interval = self.get_parameter('navigation_interval').value
        self.vlm_session_dir = self.get_parameter('vlm_session_dir').value
        
        # State
        self.latest_image = None
        self.latest_sensor_data = None
        self.cycle_count = 0
        self.sequence_state = "waiting"  # waiting, action_executing, post_action_wait
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.vlm_session_dir:
            self.session_dir = f"./data_random_{timestamp}"
        else:
            self.session_dir = self.vlm_session_dir
        
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/annotated", exist_ok=True)
        
        # ROS2 setup
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        self.sensor_sub = self.create_subscription(
            String, 'robot/sensor_data', self.sensor_callback, 10)
        self.command_pub = self.create_publisher(String, 'robot/command', 10)
        
        # Data collection publishers
        self.vlm_decision_pub = self.create_publisher(String, 'vlm/decision', 10)
        self.action_status_pub = self.create_publisher(String, 'vlm/action_status', 10)
        self.final_current_pub = self.create_publisher(String, 'vlm/final_current', 10)
        
        # Action mapping with base distance 1m
        self.actions = {
            1: {"angle": 60, "base_distance": 1.0, "desc": "Turn right 60° then forward"},
            2: {"angle": 35, "base_distance": 1.0, "desc": "Turn right 35° then forward"},
            3: {"angle": 0, "base_distance": 1.0, "desc": "Move straight forward"},
            4: {"angle": -35, "base_distance": 1.0, "desc": "Turn left 35° then forward"},
            5: {"angle": -60, "base_distance": 1.0, "desc": "Turn left 60° then forward"}
        }
        
        # Start navigation with initial assumptions (all zeros for first action)
        self.first_action = True
        self.navigation_timer = self.create_timer(self.navigation_interval, self.navigation_cycle)
        
        self.get_logger().info("🎲 VLM Random Navigation Node Started (Data Collection Mode)")
        self.get_logger().info(f"   Goal: {self.goal}")
        self.get_logger().info(f"   Max cycles: {self.max_iterations}")
        self.get_logger().info(f"   Random distance: -1.0 to +3.0m added to base 1.0m")
        self.get_logger().info(f"   Sequence: VLM → Action → Wait 3s → Final current → Repeat")
        self.get_logger().info(f"   Session: {self.session_dir}")
    
    def image_callback(self, msg):
        """Store latest camera image"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_image = frame
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")
    
    def sensor_callback(self, msg):
        """Store latest sensor data"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
    
    def navigation_cycle(self):
        """Main VLM navigation cycle with data collection sequence"""
        self.cycle_count += 1
        
        if self.cycle_count > self.max_iterations:
            self.get_logger().info(f"🏁 Completed {self.max_iterations} cycles")
            self.navigation_timer.cancel()
            return
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"🎲 VLM RANDOM DATA COLLECTION CYCLE #{self.cycle_count}/{self.max_iterations}")
        self.get_logger().info(f"{'='*60}")
        
        # Ensure robot starts stationary
        self.send_command("STOP")
        time.sleep(0.5)  # Brief pause to ensure stop
        
        if self.latest_image is None:
            self.get_logger().warning("⏳ No camera image available")
            return
        
        try:
            # Step 1: VLM Decision (robot is stationary)
            self.get_logger().info("📍 Step 1: Robot stationary, getting VLM decision...")
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_file = f"image_{self.cycle_count:03d}_{timestamp}.jpg"
            image_path = f"{self.session_dir}/images/{image_file}"
            cv2.imwrite(image_path, self.latest_image)
            
            # Annotate image
            annotate_start = time.time()
            annotated_file = f"annotated_{self.cycle_count:03d}_{timestamp}.jpg"
            annotated_path = f"{self.session_dir}/annotated/{annotated_file}"
            annotate_image(image_path, annotated_path)
            annotate_time = time.time() - annotate_start
            
            self.get_logger().info(f"🎨 Image annotated ({annotate_time:.2f}s)")
            
            # Get VLM decision
            vlm_start = time.time()
            self.get_logger().info("🧠 Sending to Gemini VLM...")
            
            if self.first_action:
                # For first action, use default action (assume initial state is all zeros)
                action = 3  # Move straight forward
                reasoning = "Initial action - moving straight forward"
                random_distance = 0.0  # No randomness for first action
                self.first_action = False
                self.get_logger().info("🎯 Using initial action (3) with zero assumptions")
            else:
                response = generate_response(annotated_path, self.goal, turn_around_available=False)
                vlm_time = time.time() - vlm_start
                
                if not response:
                    self.get_logger().error("❌ VLM response failed")
                    return
                
                action = parse_action(response)
                reasoning = response.text if response else "No reasoning"
                
                if action < 1 or action > 5:
                    self.get_logger().error(f"❌ Invalid action: {action}")
                    return
                
                self.get_logger().info(f"🧠 VLM responded ({vlm_time:.2f}s): Action {action}")
            
            # Generate random distance from -1 to 3 meters
            random_distance = random.uniform(-1.0, 3.0)
            total_distance = max(0.1, self.actions[action]["base_distance"] + random_distance)  # Ensure minimum 0.1m
            
            # Publish VLM decision for data collection
            self.publish_vlm_decision(action, random_distance, reasoning)
            
            self.get_logger().info(f"🎲 Random distance: {random_distance:.3f}m (Total: {total_distance:.3f}m)")
            
            # Step 2: Execute action with monitoring
            self.get_logger().info("🚀 Step 2: Executing action with data monitoring...")
            self.publish_action_status("action_started")
            
            action_result = self.execute_action(action, total_distance)
            
            self.publish_action_status("action_completed")
            self.get_logger().info(f"✅ Action completed: {action_result}")
            
            # Step 3: Wait 3 seconds at destination
            self.get_logger().info("⏳ Step 3: Waiting 3 seconds at destination...")
            self.send_command("STOP")
            time.sleep(3.0)
            
            # Step 4: Read final current sensor
            self.get_logger().info("🔋 Step 4: Reading final current sensor...")
            final_current = self.read_final_current()
            
            # Publish final current for data collection
            self.publish_final_current(final_current)
            
            self.get_logger().info(f"💾 Cycle #{self.cycle_count} complete - Final current: {final_current:.3f}A")
            
        except Exception as e:
            self.get_logger().error(f"Navigation cycle failed: {e}")
            self.send_command("STOP")
    
    def execute_action(self, action, total_distance):
        """Execute action with specified total distance"""
        action_info = self.actions[action]
        angle = action_info["angle"]
        description = action_info["desc"]
        
        self.get_logger().info(f"🎯 Executing: {description}")
        self.get_logger().info(f"   Angle: {angle}°, Distance: {total_distance:.3f}m")
        
        # Turn first if needed
        if angle != 0:
            turn_command = f"TURN,{angle}"
            self.send_command(turn_command)
            time.sleep(abs(angle) / 30)  # Rough timing based on turn rate
        
        # Move forward with calculated distance
        move_command = f"FORWARD,{total_distance:.3f}"
        self.send_command(move_command)
        
        # Wait for movement to complete (rough estimate)
        move_time = max(2.0, total_distance * 1.5)  # At least 2 seconds, scale with distance
        time.sleep(move_time)
        
        return f"Action {action}: {angle}° turn, {total_distance:.3f}m forward"
    
    def read_final_current(self):
        """Read final current sensor value (placeholder for solar panel)"""
        if self.latest_sensor_data:
            # Handle new current format which is now a dict with 'in' and 'out'
            current_data = self.latest_sensor_data.get('current', 0.0)
            
            if isinstance(current_data, dict):
                # New format: {"in": value, "out": value}
                # Use 'out' value as the main current reading (motor current)
                current_reading = current_data.get('out', 0.0)
            else:
                # Old format: simple number
                current_reading = current_data
                
            return max(0.0, current_reading)  # Filter negative values
        return 0.0  # Placeholder
    
    def publish_vlm_decision(self, action, random_distance, reasoning):
        """Publish VLM decision for data collection"""
        decision_data = {
            'cycle': self.cycle_count,
            'action': action,
            'random_distance': random_distance,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
        
        msg = String()
        msg.data = json.dumps(decision_data)
        self.vlm_decision_pub.publish(msg)
        
        self.get_logger().info(f"📡 Published VLM decision: Action {action}, Random {random_distance:.3f}m")
    
    def publish_action_status(self, status):
        """Publish action execution status"""
        status_data = {
            'cycle': self.cycle_count,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        msg = String()
        msg.data = json.dumps(status_data)
        self.action_status_pub.publish(msg)
    
    def publish_final_current(self, final_current):
        """Publish final current reading"""
        current_data = {
            'cycle': self.cycle_count,
            'final_current': final_current,
            'timestamp': datetime.now().isoformat()
        }
        
        msg = String()
        msg.data = json.dumps(current_data)
        self.final_current_pub.publish(msg)
    
    def send_command(self, command):
        """Send command to robot"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.get_logger().info(f"📤 Command: {command}")

def main(args=None):
    rclpy.init(args=args)
    node = VLMNavigationRandomNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 VLM Random Navigation stopped")
        # Send stop command
        if hasattr(node, 'send_command'):
            node.send_command("STOP")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 