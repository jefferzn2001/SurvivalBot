#!/usr/bin/env python3
"""
VLM Navigation - Standard VLM navigation with continuous raw data recording
Uses ZeroMQ data server, records all raw data with timestamps and action states
Optimized timing: VLM processing starts immediately after action completion
"""

import json
import time
import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import logging

# Import data clients
from data_client import DataClient
from command_client import CommandClient

# Import panel_current for Solar_In calculation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.FakeSolar import panel_current

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import VLMNAV components
try:
    # Import functions from VLMNAV directory
    vlmnav_path = os.path.abspath(os.path.join(os.path.expanduser('~'), 'SurvivalBot', 'VLMNAV'))
    sys.path.append(vlmnav_path)
    
    from annotation import annotate_image
    from prompt import construct_action_prompt
    
    # Import Gemini API directly
    import google.generativeai as genai
    from dotenv import load_dotenv
    import base64
    import re
    
    # Load API key
    load_dotenv(os.path.join(vlmnav_path, '.env'))
    api_key = os.getenv("API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        VLM_READY = True
        logger.info("✅ Gemini API configured")
    else:
        VLM_READY = False
        logger.error("❌ No API key found")
        
except ImportError as e:
    logger.error(f"❌ VLMNAV import failed: {e}")
    VLM_READY = False

# Global pause state for keyboard input handling
paused = False
pause_lock = threading.Lock()

def keyboard_input_handler():
    """Handle keyboard input in a separate thread"""
    global paused
    while True:
        try:
            key = input()  # Wait for Enter key press
            if key.lower() == 'p':
                with pause_lock:
                    paused = not paused
                    if paused:
                        logger.info("⏸️ PAUSED. Type 'p' and press Enter to resume.")
                    else:
                        logger.info("▶️ RESUMED.")
        except (EOFError, KeyboardInterrupt):
            break

# VLM functions
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
        logger.error(f"Error generating response: {e}")
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
        logger.error(f"Error parsing action: {e}")
    return -1

class VLMNavigation:
    def __init__(self, server_ip="10.102.225.181", goal="Max Sunlight Location", max_iterations=10):
        if not VLM_READY:
            logger.error("❌ VLM not ready! Check VLMNAV setup and API key")
            return
        
        # Parameters
        self.goal = goal
        self.max_iterations = max_iterations
        self.navigation_interval = 0.1  # Minimal interval - VLM processing starts immediately
        
        # State
        self.latest_image = None
        self.latest_sensor_data = None
        self.cycle_count = 0
        self.running = False
        self.state_instance = False  # Track when we're capturing a state instance
        
        # Action state tracking
        self.current_action_state = "idle"  # "idle" or action number (0-5)
        self.action_start_time = None
        
        # VLM processing state
        self.vlm_processing = False
        self.vlm_result = None
        self.vlm_thread = None
        
        # Setup directories in external Data folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_folder = os.path.abspath(os.path.join(os.path.expanduser('~'), 'SurvivalBot', 'data'))
        self.session_dir = os.path.join(data_folder, f"data_{timestamp}")
        
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/annotated", exist_ok=True)
        
        # Raw data recording
        self.raw_data_file = f"{self.session_dir}/raw_data.csv"
        self.raw_data_buffer = []
        self.save_interval = 1.0  # Save every 1 second
        self.last_save_time = time.time()
        
        # Reasoning CSV for VLM decisions
        self.reasoning_csv_path = f"{self.session_dir}/reasoning.csv"
        self.reasoning_csv_initialized = False
        
        # Setup data connections
        self.data_client = DataClient(server_ip=server_ip)
        self.command_client = CommandClient(server_ip=server_ip)
        
        # Add callbacks
        self.data_client.add_sensor_callback(self.sensor_callback)
        self.data_client.add_camera_callback(self.camera_callback)
        
        # Action mapping - base distance 0.3 meter
        self.actions = {
            0: {"angle": 60, "base_distance": 0.0, "desc": "Turn right 60° in place"},
            1: {"angle": 60, "base_distance": 0.3, "desc": "Turn right 60° then forward 0.3m"},
            2: {"angle": 35, "base_distance": 0.3, "desc": "Turn right 35° then forward 0.3m"},
            3: {"angle": 0, "base_distance": 0.3, "desc": "Move straight forward 0.3m"},
            4: {"angle": -35, "base_distance": 0.3, "desc": "Turn left 35° then forward 0.3m"},
            5: {"angle": -60, "base_distance": 0.3, "desc": "Turn left 60° then forward 0.3m"}
        }
        
        logger.info("🧠 VLM Navigation Started (Optimized Timing + Continuous Raw Data)")
        logger.info(f"   Goal: {self.goal}")
        logger.info(f"   Max cycles: {self.max_iterations}")
        logger.info(f"   Session: {self.session_dir}")
        logger.info(f"   Raw data: {self.raw_data_file}")
        logger.info(f"   Timing: VLM processing starts immediately after action")
    
    def camera_callback(self, frame):
        """Store latest camera image"""
        if frame is not None:
            self.latest_image = frame.copy()

    def sensor_callback(self, sensor_data):
        """Store latest sensor data and record to raw data buffer"""
        self.latest_sensor_data = sensor_data
        
        # Record all raw data with timestamp and action state
        if sensor_data:
            # Create raw data record with dev machine time-of-day timestamp
            now = datetime.now()
            time_of_day = now.hour + now.minute/60.0 + now.second/3600.0 + now.microsecond/3600000000.0
            time_of_day_rounded = round(time_of_day, 4)  # 0.01 second precision (4 decimal places = 0.0001 hours)
            
            # Calculate Solar_In using panel_current and LDR values
            ldr = sensor_data.get('ldr', {})
            ldr_left = ldr.get('left', 512)
            ldr_right = ldr.get('right', 512)
            ldr_avg = (ldr_left + ldr_right) / 2.0
            solar_in = panel_current(ldr_avg)
            
            record = {
                'timestamp': time_of_day_rounded,  # Time of day in decimal hours (13.03 format)
                'action_state': self.current_action_state,
                # Flatten all sensor data (excluding server timestamp and fake fields)
                **self.flatten_sensor_data(sensor_data),
                'Solar_In': solar_in  # Add Solar_In column
            }
            
            # Add to buffer
            self.raw_data_buffer.append(record)
            
            # Save periodically
            if time.time() - self.last_save_time >= self.save_interval:
                self.save_raw_data_buffer()
            
            # Immediately set state_instance back to False after recording one data point
            if self.state_instance:
                self.state_instance = False
    
    def flatten_sensor_data(self, sensor_data):
        """Flatten nested sensor data for CSV storage, excluding server timestamp and fake fields"""
        flattened = {}
        
        # Add state instance marker
        flattened['state_instance'] = 1 if self.state_instance else 0
        
        # IMU data
        imu = sensor_data.get('imu', {})
        flattened['imu_roll'] = imu.get('roll', 0.0)
        flattened['imu_pitch'] = imu.get('pitch', 0.0)
        flattened['imu_yaw'] = imu.get('yaw', 0.0)
        
        # Encoder data
        encoders = sensor_data.get('encoders', {})
        flattened['encoder_left'] = encoders.get('left', 0)
        flattened['encoder_right'] = encoders.get('right', 0)
        
        # Power data
        power = sensor_data.get('power', {})
        power_in = power.get('in', {})
        power_out = power.get('out', {})
        flattened['power_in_voltage'] = power_in.get('voltage', 0.0)
        flattened['power_in_current'] = power_in.get('current', 0.0)
        flattened['power_out_voltage'] = power_out.get('voltage', 0.0)
        flattened['power_out_current'] = power_out.get('current', 0.0)
        
        # LDR data
        ldr = sensor_data.get('ldr', {})
        flattened['ldr_left'] = ldr.get('left', 512)
        flattened['ldr_right'] = ldr.get('right', 512)
        
        # Environment data
        env = sensor_data.get('environment', {})
        flattened['temperature'] = env.get('temperature', 25.0)
        flattened['humidity'] = env.get('humidity', 50.0)
        flattened['pressure'] = env.get('pressure', 1013.25)
        
        # Bumper data
        bumpers = sensor_data.get('bumpers', {})
        flattened['bumper_top'] = bumpers.get('top', 0)
        flattened['bumper_bottom'] = bumpers.get('bottom', 0)
        flattened['bumper_left'] = bumpers.get('left', 0)
        flattened['bumper_right'] = bumpers.get('right', 0)
        
        # Motion field only (explicitly exclude timestamp and fake fields from data_server)
        flattened['motion'] = sensor_data.get('motion', 'stop')
        
        # Note: Explicitly NOT including 'timestamp' or 'fake' fields from sensor_data
        # as these are server-side fields we don't want in our raw data CSV
        
        return flattened
    
    def save_raw_data_buffer(self):
        """Save raw data buffer to CSV file"""
        if not self.raw_data_buffer:
            return
            
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.raw_data_buffer)
            
            # Save to CSV (append if file exists)
            if os.path.exists(self.raw_data_file):
                df.to_csv(self.raw_data_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.raw_data_file, mode='w', header=True, index=False)
                logger.info(f"📄 Raw data CSV created: {self.raw_data_file}")
            
            # Print the latest raw values
            latest_record = self.raw_data_buffer[-1]  # Get most recent record
            logger.info("📊 Latest Raw Values:")
            logger.info(f"  Time: {latest_record['timestamp']:.4f}h")
            logger.info(f"  State: {latest_record['action_state']}")
            logger.info(f"  IMU: roll={latest_record['imu_roll']:.2f}° pitch={latest_record['imu_pitch']:.2f}° yaw={latest_record['imu_yaw']:.2f}°")
            logger.info(f"  Encoders: L={latest_record['encoder_left']} R={latest_record['encoder_right']}")
            logger.info(f"  LDR: L={latest_record['ldr_left']} R={latest_record['ldr_right']}")
            logger.info(f"  Solar_In: {latest_record['Solar_In']:.3f}A")
            logger.info(f"  Power: Vin={latest_record['power_in_voltage']:.2f}V Iin={latest_record['power_in_current']:.2f}A")
            logger.info(f"  Environment: {latest_record['temperature']:.1f}°C {latest_record['humidity']:.1f}%RH {latest_record['pressure']:.1f}hPa")
            logger.info(f"  Motion: {latest_record['motion']}")
            
            # Clear buffer
            self.raw_data_buffer = []
            self.last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")
    
    def start_vlm_processing_async(self, image_path, annotated_path, timestamp_str, image_file):
        """Start VLM processing in background thread"""
        def vlm_processing_thread():
            try:
                # Get VLM decision
                vlm_start = time.time()
                logger.info("🧠 Sending to Gemini VLM (async)...")
                
                response = generate_response(annotated_path, self.goal, turn_around_available=False)
                vlm_time = time.time() - vlm_start
                
                if not response:
                    logger.error("❌ VLM response failed")
                    self.vlm_result = None
                    return
                
                action = parse_action(response)
                reasoning = response.text if response else "No reasoning"
                
                if action < 1 or action > 5:
                    logger.error(f"❌ Invalid action: {action}")
                    self.vlm_result = None
                    return
                
                logger.info(f"🧠 VLM responded ({vlm_time:.2f}s): Action {action}")
                
                # Store result
                self.vlm_result = {
                    'action': action,
                    'reasoning': reasoning,
                    'timestamp_str': timestamp_str,
                    'image_file': image_file,
                    'annotated_path': annotated_path
                }
                
            except Exception as e:
                logger.error(f"VLM processing failed: {e}")
                self.vlm_result = None
            finally:
                self.vlm_processing = False
        
        # Start VLM processing in background
        self.vlm_processing = True
        self.vlm_result = None
        self.vlm_thread = threading.Thread(target=vlm_processing_thread, daemon=True)
        self.vlm_thread.start()
    
    def execute_action_immediately(self, action):
        """Execute action and wait for natural completion with motion state detection"""
        # Set action state
        self.current_action_state = str(action)
        self.action_start_time = time.time()
        
        action_info = self.actions[action]
        angle = action_info["angle"]
        total_distance = action_info["base_distance"]
        description = action_info["desc"]
        
        logger.info(f"🎯 Executing: {description}")
        logger.info(f"   Angle: {angle}°, Distance: {total_distance:.3f}m")
        
        # Special handling for action 0 (turn only)
        if action == 0:
            self.command_client.turn(angle)
            logger.info("⏳ Waiting for turn to complete...")
            # Wait for motion to stop
            self.wait_for_motion_stop(timeout=10.0)
            logger.info("✅ Action 0 (turn) completed")
            self.current_action_state = "idle"
            return
        
        # Normal handling for other actions with motion state detection
        # Turn first if needed
        if angle != 0:
            logger.info(f"🔄 Starting turn: {angle}°")
            self.command_client.turn(angle)
            
            # Wait for turn to complete (motion stops)
            logger.info("⏳ Waiting for turn to complete...")
            self.wait_for_motion_stop(timeout=10.0)
            logger.info("✅ Turn completed")
        
        # Move forward with calculated distance
        if total_distance > 0:
            logger.info(f"➡️ Starting forward movement: {total_distance:.3f}m")
            self.command_client.move_forward(total_distance)
            
            # Wait for forward movement to complete (motion stops)
            logger.info("⏳ Waiting for forward movement to complete...")
            self.wait_for_motion_stop(timeout=15.0)
            logger.info("✅ Forward movement completed")
        
        logger.info(f"✅ Action {action} sequence completed")
        
        # Return to idle state only after everything is complete
        self.current_action_state = "idle"
        logger.info("🟢 Robot state: idle")
    
    def wait_for_motion_stop(self, timeout=10.0):
        """Wait for robot motion to stop, with timeout"""
        start_time = time.time()
        consecutive_stops = 0
        required_stops = 3  # Need 3 consecutive 'stop' readings to confirm stopped
        
        while (time.time() - start_time) < timeout:
            if self.latest_sensor_data and 'motion' in self.latest_sensor_data:
                motion_state = self.latest_sensor_data['motion']
                
                if motion_state == 'stop':
                    consecutive_stops += 1
                    if consecutive_stops >= required_stops:
                        logger.info(f"✅ Motion confirmed stopped after {consecutive_stops} readings")
                        return True
                else:
                    consecutive_stops = 0  # Reset counter if motion detected
                    logger.debug(f"Motion detected: {motion_state}")
            
            time.sleep(0.1)  # Check every 100ms
        
        logger.warning(f"⚠️ Timeout waiting for motion to stop after {timeout}s")
        return False
    
    def navigation_cycle(self):
        """Optimized VLM navigation cycle with immediate processing (runs forever until stopped)"""
        self.cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 VLM NAVIGATION CYCLE #{self.cycle_count}")
        logger.info(f"{'='*60}")
        # Only proceed if robot is in idle state (action fully completed)
        if self.current_action_state != "idle":
            logger.warning(f"⚠️ Robot still in action state: {self.current_action_state}, waiting...")
            return
        # Additional safety stop (robot should already be stopped from previous action)
        self.command_client.stop()
        if self.latest_image is None:
            logger.warning("⏳ No camera image available")
            return
        try:
            # Mark this as a state instance (critical data episode)
            # This will be reset to False after the first sensor reading is recorded
            self.state_instance = True
            # Image Capture and Annotation Phase (immediate)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            logger.info("📸 Capturing and annotating image...")
            # Save image
            image_file = f"cycle_{self.cycle_count:03d}_{timestamp_str}.jpg"
            image_path = f"{self.session_dir}/images/{image_file}"
            cv2.imwrite(image_path, self.latest_image)
            # Annotate image and save to annotated folder
            annotated_path = f"{self.session_dir}/annotated/{image_file}"
            annotate_image(image_path, annotated_path)
            # Start VLM processing immediately in background, using annotated image
            self.start_vlm_processing_async(image_path, annotated_path, timestamp_str, image_file)
            # Always wait full 0.3 seconds for VLM stability
            logger.info("⏳ Waiting 0.3s for VLM stability...")
            time.sleep(0.3)
            # Check if VLM processing is complete
            if not self.vlm_processing and self.vlm_result:
                # VLM finished during wait - execute action
                result = self.vlm_result
                action = result['action']
                logger.info(f"🚀 VLM ready! Executing action {action}")
                # Add action label and save reasoning
                self.add_action_label_to_image(result['annotated_path'], action)
                self.save_vlm_reasoning(action, result['reasoning'], result['timestamp_str'], result['image_file'])
                # Execute action (this will handle all timing internally)
                self.execute_action_immediately(action)
            else:
                # VLM still processing - wait for completion
                logger.info("🧠 VLM still processing, waiting for completion...")
                # Wait for VLM to complete (with timeout)
                timeout = 10.0  # 10 second timeout
                start_wait = time.time()
                while self.vlm_processing and (time.time() - start_wait) < timeout:
                    time.sleep(0.1)
                if self.vlm_result:
                    result = self.vlm_result
                    action = result['action']
                    logger.info(f"🚀 VLM completed! Executing action {action}")
                    # Add action label and save reasoning
                    self.add_action_label_to_image(result['annotated_path'], action)
                    self.save_vlm_reasoning(action, result['reasoning'], result['timestamp_str'], result['image_file'])
                    # Execute action (this will handle all timing internally)
                    self.execute_action_immediately(action)
                else:
                    logger.error("❌ VLM processing timed out")
                    return
        except Exception as e:
            logger.error(f"Navigation cycle failed: {e}")
            self.current_action_state = "idle"
            self.command_client.stop()
    
    def add_action_label_to_image(self, image_path, action):
        """Add action label to annotated image"""
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Unable to load image for action annotation: {image_path}")
            return

        # Define annotation text and position
        annotation_text = f"Executed Action: {action}"
        position = (50, 50)  # Top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 255)  # Yellow (BGR format)
        thickness = 2

        # Add text annotation to the image
        cv2.putText(img, annotation_text, position, font, font_scale, font_color, thickness)

        # Save the updated image
        cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"🏷️ Action annotation added: {annotation_text}")
    
    def save_vlm_reasoning(self, action, reasoning, timestamp_str, image_file):
        """Save VLM reasoning to CSV file"""
        try:
            reasoning_data = {
                'cycle': self.cycle_count,
                'timestamp': datetime.now().isoformat(),
                'timestamp_str': timestamp_str,
                'image_file': image_file,
                'action': action,
                'reasoning': reasoning.replace('\n', ' ').replace('\r', ' ')  # Clean reasoning text
            }
            
            # Initialize reasoning CSV if first entry
            if not self.reasoning_csv_initialized:
                df = pd.DataFrame([reasoning_data])
                df.to_csv(self.reasoning_csv_path, index=False)
                self.reasoning_csv_initialized = True
                logger.info(f"📄 Reasoning CSV initialized: {self.reasoning_csv_path}")
            else:
                # Append to existing reasoning CSV
                df = pd.DataFrame([reasoning_data])
                df.to_csv(self.reasoning_csv_path, mode='a', header=False, index=False)
            
            logger.info(f"💭 VLM reasoning saved: Cycle {self.cycle_count}, Action {action}")
        except Exception as e:
            logger.error(f"Failed to save VLM reasoning: {e}")
    
    def execute_action(self, action, total_distance):
        """Execute action with specified total distance"""
        action_info = self.actions[action]
        angle = action_info["angle"]
        description = action_info["desc"]
        
        logger.info(f"🎯 Executing: {description}")
        logger.info(f"   Angle: {angle}°, Distance: {total_distance:.3f}m")
        
        # Turn first if needed
        if angle != 0:
            self.command_client.turn(angle)
            time.sleep(abs(angle) / 30)  # Rough timing based on turn rate
        
        # Move forward with calculated distance
        self.command_client.move_forward(total_distance)
        
        # Wait for movement to complete (rough estimate)
        move_time = max(2.0, total_distance * 1.5)  # At least 2 seconds, scale with distance
        time.sleep(move_time)
        
        return f"Action {action}: {angle}° turn, {total_distance:.3f}m forward"
    
    def run(self):
        """Main run loop with pause/resume support (type 'p' and press Enter to pause/resume)"""
        global paused
        self.running = True
        
        # Start keyboard input handler thread
        input_thread = threading.Thread(target=keyboard_input_handler, daemon=True)
        input_thread.start()
        
        logger.info("🎮 Navigation controls:")
        logger.info("   Type 'p' and press Enter to pause/resume")
        logger.info("   Press Ctrl+C to stop")
        
        # Start data client in separate thread
        data_thread = threading.Thread(target=self.data_client.run, daemon=True)
        data_thread.start()
        
        # Wait for initial data
        logger.info("⏳ Waiting for camera and sensor data...")
        while self.latest_image is None or self.latest_sensor_data is None:
            time.sleep(0.1)
        
        logger.info("✅ Data connection established")
        logger.info(f"📊 Raw data recording to: {self.raw_data_file}")
        
        try:
            while self.running:
                # Check pause state
                with pause_lock:
                    is_paused = paused
                
                if is_paused:
                    time.sleep(0.1)
                    continue
                    
                self.navigation_cycle()
                time.sleep(self.navigation_interval)  # Minimal delay (0.1s)
        except KeyboardInterrupt:
            logger.info("🛑 VLM Navigation stopped by user")
        finally:
            # Save any remaining raw data
            self.save_raw_data_buffer()
            self.command_client.stop()
            self.command_client.close()
            logger.info("💾 Final raw data save completed")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM Navigation with Optimized Timing + Raw Data Recording')
    parser.add_argument('--server-ip', type=str, default='10.102.225.181',
                      help='IP address of the robot running data_server.py')
    parser.add_argument('--goal', type=str, default='Max Sunlight Location',
                      help='Navigation goal for VLM')
    parser.add_argument('--max-iterations', type=int, default=10,
                      help='Maximum number of navigation cycles (not used - runs forever)')
    args = parser.parse_args()
    
    # Create and run VLM navigation
    vlm_nav = VLMNavigation(
        server_ip=args.server_ip,
        goal=args.goal,
        max_iterations=args.max_iterations
    )
    
    vlm_nav.run()

if __name__ == '__main__':
    main() 