#!/usr/bin/env python3
"""
VLM-Triggered Data Collection Node - Records one dataset per VLM action
Dataset includes: image, action (1-5), random_distance (-1 to 3), bumper_event, total_current, final_current, encoder_movement, IMU data, environmental data
VLM reasoning saved separately in session directory
"""

import rclpy
from rclpy.node import Node
import json
import time
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        
        # Parameters
        self.declare_parameter('vlm_triggered_mode', True)
        self.declare_parameter('output_dir', './train')
        self.declare_parameter('session_name', 'vlm_session')
        self.declare_parameter('vlm_session_dir', '')  # Shared VLM session directory
        
        self.vlm_triggered_mode = self.get_parameter('vlm_triggered_mode').value
        self.output_dir = self.get_parameter('output_dir').value
        self.session_name = self.get_parameter('session_name').value
        self.vlm_session_dir = self.get_parameter('vlm_session_dir').value
        
        # Setup simplified output structure - single directory for everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use shared VLM session directory if provided, otherwise create default
        if not self.vlm_session_dir:
            self.session_dir = f"./data_{self.session_name}_{timestamp}"
        else:
            self.session_dir = self.vlm_session_dir
        
        # Create single session directory with subdirectories
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/annotated", exist_ok=True)
        
        # Single CSV file for entire session
        self.csv_file_path = f"{self.session_dir}/dataset.csv"
        self.csv_initialized = False
        
        # Reasoning CSV file for entire session
        self.reasoning_csv_path = f"{self.session_dir}/reasoning.csv"
        self.reasoning_csv_initialized = False
        
        # Data collection state
        self.dataset_counter = 0
        self.collecting = False  # True when monitoring action execution
        self.action_start_time = None
        self.current_dataset = {}
        
        # Latest sensor data
        self.latest_image = None
        self.latest_sensor_data = None
        
        # Bumper monitoring during action
        self.bumper_event_detected = "none"
        self.bumper_detected_first = None
        
        # Current monitoring for filtering
        self.current_readings = []
        self.current_monitoring_start = None
        
        # Encoder tracking
        self.encoder_start_values = {"left": 0, "right": 0}
        self.encoder_movement = {"left": 0, "right": 0}
        
        # ROS2 Subscriptions
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        self.sensor_sub = self.create_subscription(
            String, 'robot/sensor_data', self.sensor_callback, 10)
        
        # VLM decision subscription - triggers data collection
        self.vlm_decision_sub = self.create_subscription(
            String, 'vlm/decision', self.vlm_decision_callback, 10)
        
        # Action status subscription - monitors action execution
        self.action_status_sub = self.create_subscription(
            String, 'vlm/action_status', self.action_status_callback, 10)
        
        # Final current reading subscription - triggers dataset completion
        self.final_current_sub = self.create_subscription(
            String, 'vlm/final_current', self.final_current_callback, 10)
        
        self.get_logger().info("📊 VLM-Triggered Data Collection Node Started")
        self.get_logger().info(f"   Session Directory: {self.session_dir}")
        self.get_logger().info(f"   Dataset CSV: {self.csv_file_path}")
        self.get_logger().info(f"   Session: {self.session_name}")
        self.get_logger().info(f"   Recording: image, action, random_distance, bumper_event, current_data, encoders, IMU, environment")
    
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
        """Store latest sensor data and monitor for bumper events during action"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
            
            # Monitor bumper events only during action execution
            if self.collecting:
                self.monitor_bumper_events()
                self.monitor_current_readings()
                
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
    
    def monitor_bumper_events(self):
        """Monitor for bumper events during action execution"""
        if not self.latest_sensor_data:
            return
            
        bumpers = self.latest_sensor_data.get('bumpers', {})
        
        # Check for any bumper activation (only record first one)
        if self.bumper_event_detected == "none":
            for bumper_name, value in bumpers.items():
                if value == 1:  # Bumper activated
                    self.bumper_event_detected = bumper_name
                    self.get_logger().info(f"🚨 Bumper event detected: {bumper_name}")
                    break
    
    def monitor_current_readings(self):
        """Monitor current readings during action for total current calculation"""
        if not self.latest_sensor_data:
            return
            
        current_value = self.latest_sensor_data.get('current', 0.0)
        
        # Filter out negative values and collect readings
        if current_value >= 0:
            self.current_readings.append(current_value)
    
    def vlm_decision_callback(self, msg):
        """VLM decision received - start new dataset collection"""
        try:
            vlm_data = json.loads(msg.data)
            
            if self.latest_image is None:
                self.get_logger().warning("⏳ No camera image available for VLM decision")
                return
            
            # Start new dataset
            self.dataset_counter += 1
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Capture sensor state at moment of VLM decision
            decision_sensor_data = self.capture_decision_sensor_data()
            
            self.current_dataset = {
                'dataset_id': self.dataset_counter,
                'timestamp': datetime.now().isoformat(),
                'action': vlm_data.get('action', 0),
                'random_distance': vlm_data.get('random_distance', 0.0),
                'image_path': None,
                'timestamp_str': timestamp_str,
                # Sensor data at VLM decision moment
                'imu_roll': decision_sensor_data['imu_roll'],
                'imu_pitch': decision_sensor_data['imu_pitch'],
                'temperature': decision_sensor_data['temperature'],
                'pressure': decision_sensor_data['pressure'],
                'humidity': decision_sensor_data['humidity']
            }
            
            # Save VLM reasoning to file in session directory
            reasoning = vlm_data.get('reasoning', '')
            if reasoning:
                self.save_vlm_reasoning(reasoning, timestamp_str)
            
            # Save the image at moment of VLM decision
            self.save_vlm_decision_image()
            
            # Reset monitoring state and record starting encoder values
            self.bumper_event_detected = "none"
            self.current_readings = []
            self.record_encoder_start()
            
            self.get_logger().info(f"📸 Dataset #{self.dataset_counter} started - Action: {self.current_dataset['action']}, Random Distance: {self.current_dataset['random_distance']}")
            self.get_logger().info(f"   IMU: R{decision_sensor_data['imu_roll']:.2f}°, P{decision_sensor_data['imu_pitch']:.2f}°")
            self.get_logger().info(f"   Env: T{decision_sensor_data['temperature']:.1f}°C, H{decision_sensor_data['humidity']:.1f}%, P{decision_sensor_data['pressure']:.1f}hPa")
            
        except Exception as e:
            self.get_logger().error(f"VLM decision parse failed: {e}")
    
    def capture_decision_sensor_data(self):
        """Capture IMU and environmental data at moment of VLM decision"""
        decision_data = {
            'imu_roll': 0.0,
            'imu_pitch': 0.0,
            'temperature': 25.0,
            'pressure': 1013.25,
            'humidity': 50.0
        }
        
        if self.latest_sensor_data:
            # Extract IMU data
            imu_data = self.latest_sensor_data.get('imu', {})
            decision_data['imu_roll'] = imu_data.get('roll', 0.0)
            decision_data['imu_pitch'] = imu_data.get('pitch', 0.0)
            
            # Extract environmental data
            env_data = self.latest_sensor_data.get('environment', {})
            decision_data['temperature'] = env_data.get('temperature', 25.0)
            decision_data['pressure'] = env_data.get('pressure', 1013.25)
            decision_data['humidity'] = env_data.get('humidity', 50.0)
        
        return decision_data
    
    def action_status_callback(self, msg):
        """Action status updates - start/stop monitoring"""
        try:
            status_data = json.loads(msg.data)
            status = status_data.get('status', '')
            
            if status == 'action_started':
                self.collecting = True
                self.action_start_time = time.time()
                self.current_monitoring_start = time.time()
                self.get_logger().info("🚀 Action started - monitoring bumpers, current, and encoders")
                
            elif status == 'action_completed':
                self.collecting = False
                
                # Calculate total current drawn (sum of filtered readings)
                total_current = sum(self.current_readings) if self.current_readings else 0.0
                
                # Calculate encoder movement before they get cleared
                self.calculate_encoder_movement()
                
                # Update dataset with monitoring results
                self.current_dataset['bumper_event'] = self.bumper_event_detected
                self.current_dataset['total_current'] = total_current
                self.current_dataset['current_samples'] = len(self.current_readings)
                self.current_dataset['encoder_left'] = self.encoder_movement['left']
                self.current_dataset['encoder_right'] = self.encoder_movement['right']
                
                self.get_logger().info(f"✅ Action completed - Bumper: {self.bumper_event_detected}, Total Current: {total_current:.3f}, Encoders: L{self.encoder_movement['left']}, R{self.encoder_movement['right']}")
                
        except Exception as e:
            self.get_logger().error(f"Action status parse failed: {e}")
    
    def final_current_callback(self, msg):
        """Final current reading - complete the dataset"""
        try:
            final_data = json.loads(msg.data)
            final_current = final_data.get('final_current', 0.0)
            
            # Complete the dataset
            self.current_dataset['final_current'] = final_current
            
            # Save dataset to CSV
            self.save_dataset_to_csv()
            
            self.get_logger().info(f"💾 Dataset #{self.dataset_counter} completed - Final Current: {final_current:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Final current parse failed: {e}")
    
    def record_encoder_start(self):
        """Record encoder values at start of action"""
        if self.latest_sensor_data:
            encoders = self.latest_sensor_data.get('encoders', {"left": 0, "right": 0})
            self.encoder_start_values = {
                "left": encoders.get('left', 0),
                "right": encoders.get('right', 0)
            }
            self.get_logger().info(f"📏 Encoder start: L{self.encoder_start_values['left']}, R{self.encoder_start_values['right']}")
    
    def calculate_encoder_movement(self):
        """Calculate encoder movement during action (before they get cleared)"""
        if self.latest_sensor_data:
            current_encoders = self.latest_sensor_data.get('encoders', {"left": 0, "right": 0})
            self.encoder_movement = {
                "left": current_encoders.get('left', 0) - self.encoder_start_values['left'],
                "right": current_encoders.get('right', 0) - self.encoder_start_values['right']
            }
        else:
            self.encoder_movement = {"left": 0, "right": 0}
    
    def save_vlm_reasoning(self, reasoning, timestamp_str):
        """Save VLM reasoning to CSV file as new row"""
        try:
            reasoning_data = {
                'dataset_id': self.dataset_counter,
                'timestamp': datetime.now().isoformat(),
                'action': self.current_dataset['action'],
                'random_distance': self.current_dataset['random_distance'],
                'imu_roll': self.current_dataset['imu_roll'],
                'imu_pitch': self.current_dataset['imu_pitch'],
                'temperature': self.current_dataset['temperature'],
                'humidity': self.current_dataset['humidity'],
                'pressure': self.current_dataset['pressure'],
                'reasoning': reasoning
            }
            
            # Initialize reasoning CSV if first entry
            if not self.reasoning_csv_initialized:
                df = pd.DataFrame([reasoning_data])
                df.to_csv(self.reasoning_csv_path, index=False)
                self.reasoning_csv_initialized = True
                self.get_logger().info(f"📄 Reasoning CSV initialized: {self.reasoning_csv_path}")
            else:
                # Append to existing reasoning CSV
                df = pd.DataFrame([reasoning_data])
                df.to_csv(self.reasoning_csv_path, mode='a', header=False, index=False)
            
            self.get_logger().info(f"💭 VLM reasoning saved to CSV: Dataset {self.dataset_counter}")
        except Exception as e:
            self.get_logger().error(f"Failed to save VLM reasoning: {e}")
    
    def save_vlm_decision_image(self):
        """Save the image at the moment of VLM decision and create annotated version"""
        if self.latest_image is None:
            return
            
        image_filename = f"dataset_{self.dataset_counter:03d}_{self.current_dataset['timestamp_str']}.jpg"
        image_path = f"{self.session_dir}/images/{image_filename}"
        
        # Save original image
        cv2.imwrite(image_path, self.latest_image)
        self.current_dataset['image_path'] = image_filename
        
        # Create annotated version with action label
        annotated_image = self.latest_image.copy()
        action_text = f"Action: {self.current_dataset['action']}"
        
        # Get image dimensions for positioning
        height, width = annotated_image.shape[:2]
        
        # Set up text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)  # Black text
        thickness = 2
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(action_text, font, font_scale, thickness)
        
        # Position in top right corner with some padding
        padding = 10
        x = width - text_width - padding
        y = text_height + padding
        
        # Add text to image
        cv2.putText(annotated_image, action_text, (x, y), font, font_scale, color, thickness)
        
        # Save annotated image
        annotated_filename = f"annotated_{self.dataset_counter:03d}_{self.current_dataset['timestamp_str']}.jpg"
        annotated_path = f"{self.session_dir}/annotated/{annotated_filename}"
        cv2.imwrite(annotated_path, annotated_image)
        
        self.get_logger().info(f"📸 VLM decision image saved: {image_filename}")
        self.get_logger().info(f"🏷️  Annotated image saved: {annotated_filename}")
    
    def save_dataset_to_csv(self):
        """Save completed dataset to CSV file (single file for entire session)"""
        if not self.current_dataset:
            return
            
        # Prepare data row with all required fields including IMU and environmental data
        data_row = {
            'dataset_id': self.current_dataset['dataset_id'],
            'timestamp': self.current_dataset['timestamp'],
            'image_path': self.current_dataset['image_path'],
            'action': self.current_dataset['action'],
            'random_distance': self.current_dataset['random_distance'],
            'bumper_event': self.current_dataset['bumper_event'],
            'total_current': self.current_dataset['total_current'],
            'final_current': self.current_dataset['final_current'],
            'current_samples': self.current_dataset.get('current_samples', 0),
            'encoder_left': self.current_dataset.get('encoder_left', 0),
            'encoder_right': self.current_dataset.get('encoder_right', 0),
            # IMU data at VLM decision moment
            'imu_roll': self.current_dataset['imu_roll'],
            'imu_pitch': self.current_dataset['imu_pitch'],
            # Environmental data at VLM decision moment  
            'temperature': self.current_dataset['temperature'],
            'pressure': self.current_dataset['pressure'],
            'humidity': self.current_dataset['humidity']
        }
        
        # Initialize CSV if first dataset
        if not self.csv_initialized:
            df = pd.DataFrame([data_row])
            df.to_csv(self.csv_file_path, index=False)
            self.csv_initialized = True
            self.get_logger().info(f"📄 CSV initialized: {self.csv_file_path}")
        else:
            # Append to existing CSV
            df = pd.DataFrame([data_row])
            df.to_csv(self.csv_file_path, mode='a', header=False, index=False)
        
        self.get_logger().info(f"💾 Dataset saved to CSV: ID{data_row['dataset_id']}, Action{data_row['action']}, IMU({data_row['imu_roll']:.2f},{data_row['imu_pitch']:.2f}), Temp{data_row['temperature']:.1f}°C")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Data Collection Node stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()