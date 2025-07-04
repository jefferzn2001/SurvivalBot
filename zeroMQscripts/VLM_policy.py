#!/usr/bin/env python3
"""
VLM Policy Navigation - VLM navigation with multimodal actor-based distance scaling and stop certainty
Uses ZeroMQ data server, records all raw data with timestamps, action states, and policy decisions
Integrates with multimodal policy actor for variable distance control (0.3-0.6m) and stop certainty
Processes camera images through CLIP vision encoder for enhanced policy decisions
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
import torch
from PIL import Image

# Import data clients
from data_client import DataClient
from command_client import CommandClient

# Import panel_current for Solar_In calculation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.FakeSolar import panel_current

# Import multimodal policy actor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAC')))
from policy_actor import VLMPolicyActor

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

class VLMPolicyNavigation:
    def __init__(self, server_ip="10.102.225.181", goal="Max Sunlight Location", 
                 max_iterations=10, model_path=None, stop_threshold=0.95, device='cpu'):
        if not VLM_READY:
            logger.error("❌ VLM not ready! Check VLMNAV setup and API key")
            return
        
        # Parameters
        self.goal = goal
        self.max_iterations = max_iterations
        self.navigation_interval = 0.1  # Minimal interval
        self.stop_threshold = stop_threshold  # Stop if certainty > this value
        self.device = device
        
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
        
        # Policy inference timing
        self.policy_inference_active = False  # Track when policy inference is happening
        
        # Initialize policy values to None (not yet computed)
        self.current_distance_scale = None
        self.current_stop_certainty = None
        
        # Multimodal policy actor setup
        self.obs_dim = 77  # 64 vision + 13 sensor
        self.actor = VLMPolicyActor(obs_dim=self.obs_dim, device=device)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self.actor.load_state_dict(torch.load(model_path, map_location=device))
                logger.info(f"✅ Multimodal policy actor loaded from: {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load policy actor: {e}")
                logger.info("🔄 Using randomly initialized multimodal policy actor")
        else:
            logger.info("🎲 Using randomly initialized multimodal policy actor")
        
        self.actor.eval()  # Set to evaluation mode
        
        # Setup directories in external Data folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_folder = os.path.abspath(os.path.join(os.path.expanduser('~'), 'SurvivalBot', 'data'))
        self.session_dir = os.path.join(data_folder, f"policy_data_{timestamp}")
        
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/annotated", exist_ok=True)
        
        # Raw data recording (with additional policy columns)
        self.raw_data_file = f"{self.session_dir}/raw_data.csv"
        self.raw_data_buffer = []
        self.save_interval = 1.0  # Save every 1 second
        self.last_save_time = time.time()
        
        # State data recording (exact policy input states - now 77D)
        self.state_csv_path = f"{self.session_dir}/state.csv"
        self.state_csv_initialized = False
        
        # Reasoning CSV for VLM and policy decisions
        self.reasoning_csv_path = f"{self.session_dir}/reasoning.csv"
        self.reasoning_csv_initialized = False
        
        # Setup data connections
        self.data_client = DataClient(server_ip=server_ip)
        self.command_client = CommandClient(server_ip=server_ip)
        
        # Add callbacks
        self.data_client.add_sensor_callback(self.sensor_callback)
        self.data_client.add_camera_callback(self.camera_callback)
        
        # Action mapping - base distance will be scaled by actor
        self.actions = {
            0: {"angle": 60, "base_distance": 0.0, "desc": "Turn right 60° in place"},
            1: {"angle": 60, "base_distance": 0.3, "desc": "Turn right 60° then forward [scaled]"},
            2: {"angle": 35, "base_distance": 0.3, "desc": "Turn right 35° then forward [scaled]"},
            3: {"angle": 0, "base_distance": 0.3, "desc": "Move straight forward [scaled]"},
            4: {"angle": -35, "base_distance": 0.3, "desc": "Turn left 35° then forward [scaled]"},
            5: {"angle": -60, "base_distance": 0.3, "desc": "Turn left 60° then forward [scaled]"}
        }
        
        logger.info("🧠 VLM Multimodal Policy Navigation Started")
        logger.info(f"   Goal: {self.goal}")
        logger.info(f"   Max cycles: {self.max_iterations}")
        logger.info(f"   Stop threshold: {self.stop_threshold}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Session: {self.session_dir}")
        logger.info(f"   Raw data: {self.raw_data_file}")
        logger.info(f"   State data: {self.state_csv_path}")
        logger.info(f"   Multimodal policy actor obs_dim: {self.obs_dim}")
    
    def camera_callback(self, frame):
        """Store latest camera image"""
        if frame is not None:
            self.latest_image = frame.copy()

    def sensor_callback(self, sensor_data):
        """Store latest sensor data and record to raw data buffer"""
        self.latest_sensor_data = sensor_data
        
        # Record all raw data with timestamp, action state, and policy decisions
        if sensor_data:
            # Create raw data record with dev machine time-of-day timestamp
            now = datetime.now()
            time_of_day = now.hour + now.minute/60.0 + now.second/3600.0 + now.microsecond/3600000000.0
            time_of_day_rounded = round(time_of_day, 4)  # 0.01 second precision
            
            # Calculate Solar_In using panel_current and LDR values
            ldr = sensor_data.get('ldr', {})
            ldr_left = ldr.get('left', 512)
            ldr_right = ldr.get('right', 512)
            ldr_avg = (ldr_left + ldr_right) / 2.0
            solar_in = panel_current(ldr_avg)
            
            # Get current policy predictions if available
            distance_scale = getattr(self, 'current_distance_scale', None)
            stop_certainty = getattr(self, 'current_stop_certainty', None)
            
            # Use 0.0 for policy values if not yet computed
            if distance_scale is None:
                distance_scale = 0.0
            if stop_certainty is None:
                stop_certainty = 0.0
            
            # Policy inference timing (1 during inference, 0 otherwise)
            policy_inference_marker = 1 if self.policy_inference_active else 0
            
            record = {
                'timestamp': time_of_day_rounded,
                'action_state': self.current_action_state,
                'distance_scale': distance_scale,      # New column: actor distance scaling
                'stop_certainty': stop_certainty,      # New column: actor stop certainty
                'policy_inference': policy_inference_marker,  # New column: policy inference timing
                # Flatten all sensor data (excluding server timestamp and fake fields)
                **self.flatten_sensor_data(sensor_data),
                'Solar_In': solar_in
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
        
        # Motion field
        flattened['motion'] = sensor_data.get('motion', 'stop')
        
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
            
            # Print the latest raw values including policy decisions
            latest_record = self.raw_data_buffer[-1]
            logger.info("📊 Latest Raw Values:")
            logger.info(f"  Time: {latest_record['timestamp']:.4f}h")
            logger.info(f"  State: {latest_record['action_state']}")
            logger.info(f"  Policy: distance={latest_record['distance_scale']:.3f}m, stop_cert={latest_record['stop_certainty']:.3f}")
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
                
                if action < 0 or action > 5:
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
    
    def get_policy_decision(self, image_path, sensor_data, vlm_action):
        """
        Get policy decision from multimodal actor model
        
        Args:
            image_path: Path to the camera image
            sensor_data: Sensor data dictionary
            vlm_action: VLM action number (0-5)
        
        Returns:
            tuple: (distance_scale, stop_certainty, should_stop)
        """
        try:
            # Mark policy inference as active
            self.policy_inference_active = True
            
            # Load image as PIL
            image_pil = Image.open(image_path).convert('RGB')
            
            # Get multimodal policy decision
            distance_scale, stop_certainty = self.actor.get_action(image_pil, sensor_data, vlm_action)
            
            # Debug logging
            logger.info(f"🔍 Policy inference completed: distance={distance_scale:.3f}, certainty={stop_certainty:.3f}")
            
            # Save state to CSV (extract features for logging)
            self.save_state_to_csv(image_pil, sensor_data, vlm_action, distance_scale, stop_certainty)
            
            # Determine if we should stop
            should_stop = stop_certainty > self.stop_threshold
            
            logger.info(f"🎯 Multimodal Policy Decision: VLM_action={vlm_action}, distance={distance_scale:.3f}m, certainty={stop_certainty:.3f}, stop={should_stop}")
            
            return distance_scale, stop_certainty, should_stop
            
        except Exception as e:
            logger.error(f"Multimodal policy decision failed: {e}")
            return 0.3, 0.0, False  # Default values
        finally:
            # Reset policy inference flag
            self.policy_inference_active = False
    
    def save_state_to_csv(self, image_pil, sensor_data, vlm_action, distance_scale, stop_certainty):
        """Save multimodal policy input state to CSV file"""
        try:
            # Extract the same features the policy actor uses
            multimodal_features = self.actor.extract_multimodal_features(image_pil, sensor_data, vlm_action)
            
            # Create state record with all 77 features
            state_record = {
                'cycle': self.cycle_count,
                'timestamp': datetime.now().isoformat(),
                'vlm_action': vlm_action,
                'distance_scale': distance_scale,
                'stop_certainty': stop_certainty,
            }
            
            # Add all 77 multimodal features
            for i, feature_val in enumerate(multimodal_features.detach().cpu().numpy()):
                if i < 64:
                    state_record[f'vision_{i}'] = feature_val
                else:
                    # Sensor features (13D) - match State.py but excluding solar_in and current_out
                    sensor_idx = i - 64
                    sensor_names = [
                        'time_category',      # 0: time_category (0-23)
                        'soc',                # 1: soc (0-100)
                        'temperature',        # 2: temperature (°C)
                        'humidity',           # 3: humidity (%RH)
                        'pressure',           # 4: pressure (hPa)
                        'roll',               # 5: roll (degrees)
                        'pitch',              # 6: pitch (degrees)
                        'ldr_left',           # 7: ldr_left (0-1023)
                        'ldr_right',          # 8: ldr_right (0-1023)
                        'bumper_hit',         # 9: bumper_hit (0 or 1)
                        'encoder_left',       # 10: encoder_left
                        'encoder_right',      # 11: encoder_right
                        'action',             # 12: action (0-5)
                    ]
                    state_record[sensor_names[sensor_idx]] = feature_val
            
            # Initialize state CSV if first entry
            if not self.state_csv_initialized:
                df = pd.DataFrame([state_record])
                df.to_csv(self.state_csv_path, index=False)
                self.state_csv_initialized = True
                logger.info(f"📄 Multimodal state CSV initialized: {self.state_csv_path}")
            else:
                # Append to existing state CSV
                df = pd.DataFrame([state_record])
                df.to_csv(self.state_csv_path, mode='a', header=False, index=False)
            
            logger.info(f"📊 Multimodal state saved: Cycle {self.cycle_count}, VLM_action {vlm_action}, Vision features: 64D, Sensor features: 13D")
        except Exception as e:
            logger.error(f"Failed to save multimodal state: {e}")
    
    def execute_action_immediately(self, action, distance_scale):
        """Execute action with policy-scaled distance and wait for natural completion"""
        # Set action state
        self.current_action_state = str(action)
        self.action_start_time = time.time()
        
        action_info = self.actions[action]
        angle = action_info["angle"]
        # Use policy distance scale directly (0.3-0.6m) instead of multiplying base distance
        total_distance = distance_scale if action_info["base_distance"] > 0 else 0
        description = action_info["desc"].replace("[scaled]", f"{total_distance:.3f}m")
        
        logger.info(f"🎯 Executing: {description}")
        logger.info(f"   Angle: {angle}°, Distance: {total_distance:.3f}m (multimodal policy distance)")
        
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
        
        # Move forward with policy distance
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
        """VLM navigation cycle with multimodal policy integration"""
        self.cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 VLM MULTIMODAL POLICY NAVIGATION CYCLE #{self.cycle_count}")
        logger.info(f"{'='*60}")
        
        # Only proceed if robot is in idle state
        if self.current_action_state != "idle":
            logger.warning(f"⚠️ Robot still in action state: {self.current_action_state}, waiting...")
            return
        
        # Additional safety stop
        self.command_client.stop()
        
        if self.latest_image is None:
            logger.warning("⏳ No camera image available")
            return
        
        try:
            # Mark this as a state instance (critical data episode)
            self.state_instance = True
            
            # Image Capture and Annotation Phase
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            logger.info("📸 Capturing and annotating image...")
            
            # Save image
            image_file = f"cycle_{self.cycle_count:03d}_{timestamp_str}.jpg"
            image_path = f"{self.session_dir}/images/{image_file}"
            cv2.imwrite(image_path, self.latest_image)
            
            # Annotate image and save to annotated folder
            annotated_path = f"{self.session_dir}/annotated/{image_file}"
            annotate_image(image_path, annotated_path)
            
            # Start VLM processing immediately in background
            self.start_vlm_processing_async(image_path, annotated_path, timestamp_str, image_file)
            
            # Wait for VLM processing to complete to get the action
            logger.info("⏳ Waiting for VLM processing to get action...")
            timeout = 10.0
            start_wait = time.time()
            while self.vlm_processing and (time.time() - start_wait) < timeout:
                time.sleep(0.1)
            
            if not self.vlm_result:
                logger.error("❌ VLM processing failed or timed out")
                return
            
            # Get VLM action
            vlm_action = self.vlm_result['action']
            logger.info(f"🧠 VLM Action: {vlm_action}")
            
            # Now get multimodal policy decision using VLM action and image
            distance_scale, stop_certainty, should_stop = self.get_policy_decision(image_path, self.latest_sensor_data, vlm_action)
            
            # Store current policy decisions for data recording
            self.current_distance_scale = distance_scale
            self.current_stop_certainty = stop_certainty
            
            # Check if policy wants to stop
            if should_stop:
                logger.info(f"🛑 Multimodal policy decision: STOP (certainty={stop_certainty:.3f} > {self.stop_threshold})")
                self.command_client.stop()
                # Save the stop decision
                self.save_vlm_reasoning(0, f"Multimodal policy stop decision: certainty={stop_certainty:.3f}", 
                                       timestamp_str, image_file, distance_scale, stop_certainty)
                return
            
            # Execute action with policy scaling
            result = self.vlm_result
            logger.info(f"🚀 Executing VLM action {vlm_action} with multimodal policy scaling")
            
            # Add action label and save reasoning
            self.add_action_label_to_image(result['annotated_path'], vlm_action, distance_scale, stop_certainty)
            self.save_vlm_reasoning(vlm_action, result['reasoning'], result['timestamp_str'], 
                                   result['image_file'], distance_scale, stop_certainty)
            
            # Execute action with policy-scaled distance
            self.execute_action_immediately(vlm_action, distance_scale)
                    
        except Exception as e:
            logger.error(f"Navigation cycle failed: {e}")
            self.current_action_state = "idle"
            self.command_client.stop()
    
    def add_action_label_to_image(self, image_path, action, distance_scale, stop_certainty):
        """Add action and policy information to annotated image"""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Unable to load image for action annotation: {image_path}")
            return

        # Action annotation
        action_text = f"Action: {action}"
        policy_text = f"Multimodal Policy: {distance_scale:.3f}m, stop={stop_certainty:.3f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Action label (yellow)
        cv2.putText(img, action_text, (50, 50), font, font_scale, (0, 255, 255), thickness)
        
        # Policy label (cyan)
        cv2.putText(img, policy_text, (50, 80), font, font_scale, (255, 255, 0), thickness)

        # Save the updated image
        cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"🏷️ Action and multimodal policy annotations added")
    
    def save_vlm_reasoning(self, action, reasoning, timestamp_str, image_file, distance_scale, stop_certainty):
        """Save VLM reasoning and policy decisions to CSV file"""
        try:
            reasoning_data = {
                'cycle': self.cycle_count,
                'timestamp': datetime.now().isoformat(),
                'timestamp_str': timestamp_str,
                'image_file': image_file,
                'action': action,
                'distance_scale': distance_scale,
                'stop_certainty': stop_certainty,
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
            
            logger.info(f"💭 VLM reasoning saved: Cycle {self.cycle_count}, Action {action}, Multimodal Policy {distance_scale:.3f}m/{stop_certainty:.3f}")
        except Exception as e:
            logger.error(f"Failed to save VLM reasoning: {e}")
    
    def run(self):
        """Main run loop with pause/resume support"""
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
                time.sleep(self.navigation_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 VLM Multimodal Policy Navigation stopped by user")
        finally:
            # Save any remaining raw data
            self.save_raw_data_buffer()
            self.command_client.stop()
            self.command_client.close()
            logger.info("💾 Final raw data save completed")

def main():
    # Create and run VLM multimodal policy navigation with defaults
    vlm_policy = VLMPolicyNavigation(
        server_ip="10.102.225.181",
        goal="Max Sunlight Location", 
        max_iterations=10,
        model_path=None,  # Use randomly initialized model
        stop_threshold=0.95,
        device='cuda'
    )
    
    vlm_policy.run()

if __name__ == '__main__':
    main() 