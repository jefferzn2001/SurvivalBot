#!/usr/bin/env python3
"""
VLM Navigation Random Node - VLM navigation with random distance variation
"""

import rclpy
from rclpy.node import Node
import json
import time
import os
import sys
import cv2
import numpy as np
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
    vlmnav_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLMNAV'))
    
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
        
        model = genai.GenerativeModel("gemini-1.5-flash-001")
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
        self.declare_parameter('max_iterations', 10)  # Changed to 10
        self.declare_parameter('navigation_interval', 10.0)
        
        self.goal = self.get_parameter('goal').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.navigation_interval = self.get_parameter('navigation_interval').value
        
        # State
        self.latest_image = None
        self.cycle_count = 0
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"./vlm_random_session_{timestamp}"
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/annotated", exist_ok=True)
        
        # ROS2 setup
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        self.command_pub = self.create_publisher(String, 'robot/command', 10)
        
        # VLM decision publisher for data collection
        self.vlm_decision_pub = self.create_publisher(String, 'vlm/decision', 10)
        
        # Status subscription for real robot feedback
        self.status_sub = self.create_subscription(
            String, 'robot/status', self.status_callback, 10)
        self.waiting_for_completion = False
        self.last_command = ""
        
        # Action mapping
        self.actions = {
            1: {"angle": -60, "desc": "Turn left 60° then forward with random distance"},
            2: {"angle": -35, "desc": "Turn left 35° then forward with random distance"},
            3: {"angle": 0, "desc": "Move straight forward with random distance"},
            4: {"angle": 35, "desc": "Turn right 35° then forward with random distance"},
            5: {"angle": 60, "desc": "Turn right 60° then forward with random distance"}
        }
        
        # Start navigation
        self.navigation_timer = self.create_timer(self.navigation_interval, self.navigation_cycle)
        
        self.get_logger().info("🎲 VLM Random Navigation Node Started")
        self.get_logger().info(f"   Goal: {self.goal}")
        self.get_logger().info(f"   Max cycles: {self.max_iterations}")
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
    
    def navigation_cycle(self):
        """Main VLM navigation cycle"""
        self.cycle_count += 1
        
        if self.cycle_count > self.max_iterations:
            self.get_logger().info(f"🏁 Completed {self.max_iterations} cycles")
            self.navigation_timer.cancel()
            return
        
        self.get_logger().info(f"\n{'='*50}")
        self.get_logger().info(f"🎲 VLM RANDOM CYCLE #{self.cycle_count}/{self.max_iterations}")
        self.get_logger().info(f"{'='*50}")
        
        if self.latest_image is None:
            self.get_logger().warning("⏳ No camera image available")
            return
        
        try:
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
            
            response = generate_response(annotated_path, self.goal, turn_around_available=False)
            vlm_time = time.time() - vlm_start
            
            if not response:
                self.get_logger().error("❌ VLM failed")
                return
            
            action = parse_action(response)
            if action not in [1, 2, 3, 4, 5]:
                self.get_logger().warning(f"⚠️ Invalid action {action}, using 3")
                action = 3
            
            # Generate random distance (1 + 0-3 meters)
            random_distance = 1.0 + np.random.uniform(0, 3)
            
            # Log results
            action_desc = self.actions[action]["desc"]
            self.get_logger().info(f"🧠 VLM DECISION: Action {action} - {action_desc}")
            self.get_logger().info(f"🎲 RANDOM DISTANCE: {random_distance:.2f}m")
            self.get_logger().info(f"⏱️ VLM Response Time: {vlm_time:.2f} seconds")
            self.get_logger().info(f"📝 VLM Reasoning: {response.text[:100]}...")
            
            # Execute action
            execute_start = time.time()
            self.execute_action(action, random_distance)
            execute_time = time.time() - execute_start
            
            # Summary
            total_time = annotate_time + vlm_time + execute_time
            self.get_logger().info(f"\n⏱️ TIMING SUMMARY:")
            self.get_logger().info(f"   Annotation: {annotate_time:.2f}s")
            self.get_logger().info(f"   VLM Response: {vlm_time:.2f}s ⭐")
            self.get_logger().info(f"   Execution: {execute_time:.2f}s")
            self.get_logger().info(f"   Total: {total_time:.2f}s")
            
            # Save decision log
            with open(f"{self.session_dir}/decisions.txt", "a") as f:
                f.write(f"\nCYCLE #{self.cycle_count}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"VLM Response Time: {vlm_time:.2f}s\n")
                f.write(f"Action: {action} - {action_desc}\n")
                f.write(f"Random Distance: {random_distance:.2f}m\n")
                f.write(f"Response: {response.text[:200]}...\n")
                f.write("-" * 40 + "\n")
                
            # Publish VLM decision
            vlm_decision_data = {
                'cycle': self.cycle_count,
                'action': action,
                'description': action_desc,
                'random_distance': random_distance,
                'angle': self.actions[action]["angle"],
                'vlm_response_time': vlm_time,
                'reasoning': response.text[:200],
                'timestamp': datetime.now().isoformat()
            }
            vlm_decision_msg = String()
            vlm_decision_msg.data = json.dumps(vlm_decision_data)
            self.vlm_decision_pub.publish(vlm_decision_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Navigation cycle failed: {e}")
    
    def execute_action(self, action, random_distance):
        """Execute VLM action with random distance"""
        try:
            angle = self.actions[action]["angle"]
            desc = self.actions[action]["desc"]
            
            self.get_logger().info(f"🚀 EXECUTING: {desc} ({random_distance:.2f}m)")
            
            # Turn if needed
            if angle != 0:
                turn_cmd = f"TURN,{angle}"
                self.send_command(turn_cmd)
                self.get_logger().info(f"   🔄 Turning {angle}°...")
                self.wait_for_completion(turn_cmd)
            
            # Move forward with random distance
            forward_cmd = f"FORWARD,{random_distance:.2f}"
            self.send_command(forward_cmd)
            self.get_logger().info(f"   ⬆️ Moving forward {random_distance:.2f}m...")
            self.wait_for_completion(forward_cmd)
            
            # Stop
            stop_cmd = "STOP"
            self.send_command(stop_cmd)
            self.wait_for_completion(stop_cmd)
            
            self.get_logger().info(f"✅ Action {action} completed with {random_distance:.2f}m")
            
        except Exception as e:
            self.get_logger().error(f"❌ Execution failed: {e}")
    
    def send_command(self, command):
        """Send command to robot"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)

    def status_callback(self, msg):
        """Handle status updates from the robot"""
        status_data = msg.data
        if status_data.startswith("COMPLETED:"):
            completed_cmd = status_data.replace("COMPLETED:", "")
            if completed_cmd == self.last_command:
                self.waiting_for_completion = False
                self.get_logger().info(f"✅ Robot confirmed: {completed_cmd}")
    
    def wait_for_completion(self, command, timeout=10.0):
        """Wait for robot to complete the command"""
        self.last_command = command
        self.waiting_for_completion = True
        
        # Use fixed delays for simulation, or wait for feedback for real robot
        if "TURN" in command:
            time.sleep(3.0)  # Simulation delay
        elif "FORWARD" in command:
            time.sleep(3.0)  # Simulation delay  
        elif "STOP" in command:
            time.sleep(1.0)  # Simulation delay
        
        # For real robot, you could implement:
        # start_time = time.time()
        # while self.waiting_for_completion and (time.time() - start_time) < timeout:
        #     rclpy.spin_once(self, timeout_sec=0.1)
        
        self.waiting_for_completion = False

def main(args=None):
    rclpy.init(args=args)
    
    if not VLM_READY:
        print("❌ Cannot start VLM Random Navigation:")
        print("   - Check VLMNAV folder exists")
        print("   - Check .env file with API_KEY")
        return
    
    node = VLMNavigationRandomNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 VLM Random Navigation stopped")
    finally:
        if hasattr(node, 'destroy_node'):
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 