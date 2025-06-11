#!/usr/bin/env python3
"""
Data Collection Node - Continuous training data collection with event detection
Records at 10Hz continuously, 20Hz for 2 seconds when bumpers activate
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

try:
    import torch
    import pickle
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - will save as pickle instead")

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        
        # Parameters
        self.declare_parameter('output_dir', './train/data')
        
        self.output_dir = self.get_parameter('output_dir').value
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"{self.output_dir}/session_{timestamp}"
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/data", exist_ok=True)
        
        # Single CSV file for continuous data
        self.csv_file_path = f"{self.session_dir}/data/continuous_data.csv"
        self.csv_initialized = False
        
        # Data storage
        self.data_counter = 0
        
        # Latest data
        self.latest_image = None
        self.latest_sensor_data = None
        self.latest_vlm_action = None
        self.latest_command = None
        
        # Recording frequencies
        self.base_frequency = 10.0  # 10 Hz
        self.rapid_frequency = 20.0  # 20 Hz for events
        self.current_frequency = self.base_frequency
        
        # Bumper event detection
        self.rapid_recording = False
        self.rapid_start_time = None
        self.rapid_duration = 2.0  # 2 seconds of rapid recording
        self.last_bumper_state = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Motion tracking
        self.last_motion_state = "stop"
        
        # ROS2 Subscriptions
        self.image_sub = self.create_subscription(
            CompressedImage, 'robot/camera/compressed', self.image_callback, 10)
        self.sensor_sub = self.create_subscription(
            String, 'robot/sensor_data', self.sensor_callback, 10)
        self.command_sub = self.create_subscription(
            String, 'robot/command', self.command_callback, 10)
        
        # VLM decision subscription (custom topic for VLM decisions)
        self.vlm_decision_sub = self.create_subscription(
            String, 'vlm/decision', self.vlm_decision_callback, 10)
        
        # Publishers for VLM decisions (so VLM nodes can publish their decisions)
        self.vlm_decision_pub = self.create_publisher(String, 'vlm/decision', 10)
        
        # Timer for continuous data collection (starts at 10Hz)
        self.collection_timer = self.create_timer(1.0 / self.base_frequency, self.collect_data_continuous)
        
        self.get_logger().info("📊 Continuous Data Collection Node Started")
        self.get_logger().info(f"   Output: {self.session_dir}")
        self.get_logger().info(f"   Strategy: Continuous recording at {self.base_frequency}Hz")
        self.get_logger().info(f"   Rapid mode: {self.rapid_frequency}Hz for {self.rapid_duration}s on bumper events")
        self.get_logger().info(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
        self.get_logger().info(f"   Single CSV: {self.csv_file_path}")
    
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
        """Store latest sensor data and detect bumper events"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
            
            # Track motion state changes
            current_motion = self.latest_sensor_data.get('motion', 'stop')
            if self.last_motion_state != current_motion:
                self.last_motion_state = current_motion
            
            # Check for bumper events
            current_bumpers = self.latest_sensor_data.get('bumpers', {})
            bumper_activated = False
            
            for bumper in ['top', 'bottom', 'left', 'right']:
                current_state = current_bumpers.get(bumper, 0)
                last_state = self.last_bumper_state.get(bumper, 0)
                
                # Detect bumper activation (0 -> 1 transition)
                if current_state == 1 and last_state == 0:
                    bumper_activated = True
                    self.get_logger().info(f"🚨 BUMPER EVENT: {bumper} bumper activated!")
                    break
            
            # Update last bumper state
            self.last_bumper_state = current_bumpers.copy()
            
            # Start rapid recording if bumper activated
            if bumper_activated:
                self.start_rapid_recording()
                
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
    
    def command_callback(self, msg):
        """Store latest robot command"""
        self.latest_command = msg.data
    
    def vlm_decision_callback(self, msg):
        """Store latest VLM decision"""
        try:
            vlm_data = json.loads(msg.data)
            self.latest_vlm_action = vlm_data
        except Exception as e:
            self.get_logger().error(f"VLM decision parse failed: {e}")
    
    def start_rapid_recording(self):
        """Start rapid recording mode for bumper events"""
        if not self.rapid_recording:
            self.rapid_recording = True
            self.rapid_start_time = time.time()
            
            # Switch to rapid frequency
            self.current_frequency = self.rapid_frequency
            self.collection_timer.cancel()
            self.collection_timer = self.create_timer(1.0 / self.rapid_frequency, self.collect_data_continuous)
            
            self.get_logger().info(f"🚀 Started rapid recording at {self.rapid_frequency}Hz for {self.rapid_duration}s")
    
    def check_rapid_recording_timeout(self):
        """Check if rapid recording should end"""
        if self.rapid_recording and self.rapid_start_time is not None:
            elapsed_time = time.time() - self.rapid_start_time
            if elapsed_time >= self.rapid_duration:
                # End rapid recording
                self.rapid_recording = False
                self.current_frequency = self.base_frequency
                
                # Switch back to base frequency
                self.collection_timer.cancel()
                self.collection_timer = self.create_timer(1.0 / self.base_frequency, self.collect_data_continuous)
                
                self.get_logger().info(f"📊 Returned to normal recording at {self.base_frequency}Hz")
    
    def collect_data_continuous(self):
        """Continuous data collection at current frequency"""
        # Check if rapid recording should timeout
        self.check_rapid_recording_timeout()
        
        # Only collect if we have both image and sensor data
        if self.latest_image is None or self.latest_sensor_data is None:
            return
        
        try:
            # Save image
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_filename = f"img_{self.data_counter:06d}_{timestamp_str}.jpg"
            image_path = f"{self.session_dir}/images/{image_filename}"
            cv2.imwrite(image_path, self.latest_image)
            
            # Determine motion state and collection type
            current_motion = self.latest_sensor_data.get('motion', 'stop')
            motion_category = "moving" if current_motion == "moving" else "stationary"
            
            # Determine collection type
            if self.rapid_recording:
                collection_type = "rapid_event"
            else:
                collection_type = "continuous"
            
            # Check if any bumper is currently activated
            bumpers = self.latest_sensor_data.get('bumpers', {})
            bumper_active = any(bumpers.get(b, 0) == 1 for b in ['top', 'bottom', 'left', 'right'])
            
            # Prepare CLEAN data point (only essential fields)
            data_point = {
                'timestamp': timestamp.isoformat(),
                'data_id': self.data_counter,
                'image_filename': image_filename,
                'motion_state': motion_category,  # Categorical: moving/stationary
                'collection_type': collection_type,  # continuous/rapid_event
                'recording_frequency': self.current_frequency,
                'bumper_event_active': bumper_active,  # Boolean flag for current bumper state
                
                # Essential sensor data only
                'sensor_imu_roll': self.latest_sensor_data.get('imu', {}).get('roll', 0.0),
                'sensor_imu_pitch': self.latest_sensor_data.get('imu', {}).get('pitch', 0.0),
                'sensor_imu_yaw': self.latest_sensor_data.get('imu', {}).get('yaw', 0.0),
                'sensor_encoders_left': self.latest_sensor_data.get('encoders', {}).get('left', 0),
                'sensor_encoders_right': self.latest_sensor_data.get('encoders', {}).get('right', 0),
                'sensor_current': self.latest_sensor_data.get('current', 0.0),
                'sensor_ldr_left': self.latest_sensor_data.get('ldr', {}).get('left', 512),
                'sensor_ldr_right': self.latest_sensor_data.get('ldr', {}).get('right', 512),
                'sensor_environment_temperature': self.latest_sensor_data.get('environment', {}).get('temperature', 25.0),
                'sensor_environment_humidity': self.latest_sensor_data.get('environment', {}).get('humidity', 50.0),
                'sensor_environment_pressure': self.latest_sensor_data.get('environment', {}).get('pressure', 1013.25),
                'sensor_bumpers_top': self.latest_sensor_data.get('bumpers', {}).get('top', 0),
                'sensor_bumpers_bottom': self.latest_sensor_data.get('bumpers', {}).get('bottom', 0),
                'sensor_bumpers_left': self.latest_sensor_data.get('bumpers', {}).get('left', 0),
                'sensor_bumpers_right': self.latest_sensor_data.get('bumpers', {}).get('right', 0),
            }
            
            # Save to continuous CSV file
            self.save_to_csv(data_point)
            
            self.data_counter += 1
            
            # Log periodically (every 50 samples to avoid spam)
            if self.data_counter % 50 == 0:
                mode = "RAPID" if self.rapid_recording else "NORMAL"
                self.get_logger().info(f"📊 Collected {self.data_counter} samples [{mode} {self.current_frequency}Hz] ({motion_category})")
            
        except Exception as e:
            self.get_logger().error(f"Data collection failed: {e}")
    
    def save_to_csv(self, data_point):
        """Save single data point to continuous CSV file"""
        try:
            df = pd.DataFrame([data_point])
            
            # Initialize CSV file with headers if first time
            if not self.csv_initialized:
                df.to_csv(self.csv_file_path, mode='w', header=True, index=False)
                self.csv_initialized = True
                self.get_logger().info(f"📁 Initialized CSV file: {self.csv_file_path}")
            else:
                # Append to existing file
                df.to_csv(self.csv_file_path, mode='a', header=False, index=False)
            
        except Exception as e:
            self.get_logger().error(f"CSV save failed: {e}")
    
    def publish_vlm_decision(self, action, distance=None, reasoning=""):
        """Helper method for VLM nodes to publish their decisions"""
        decision_data = {
            'action': action,
            'distance': distance,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
        
        msg = String()
        msg.data = json.dumps(decision_data)
        self.vlm_decision_pub.publish(msg)

    def save_final_summary(self):
        """Save final summary and backup files on shutdown"""
        try:
            if not os.path.exists(self.csv_file_path):
                return
                
            # Load the full dataset
            df = pd.read_csv(self.csv_file_path)
            
            # Create summary statistics
            summary = {
                'total_samples': len(df),
                'session_duration': (pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[0])).total_seconds(),
                'motion_states': df['motion_state'].value_counts().to_dict(),
                'collection_types': df['collection_type'].value_counts().to_dict(),
                'bumper_events': df['bumper_event_active'].sum(),
                'rapid_recordings': len(df[df['collection_type'] == 'rapid_event']),
                'frequencies_used': df['recording_frequency'].value_counts().to_dict(),
            }
            
            # Save summary
            summary_path = f"{self.session_dir}/data/session_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save backup as pickle
            pickle_path = f"{self.session_dir}/data/continuous_data.pkl"
            df.to_pickle(pickle_path)
            
            # Save as PyTorch if available
            if TORCH_AVAILABLE:
                torch_path = f"{self.session_dir}/data/continuous_data.pt"
                torch.save({
                    'dataframe': df,
                    'summary': summary,
                    'metadata': {
                        'session_dir': self.session_dir,
                        'collection_strategy': 'continuous_with_event_detection',
                        'base_frequency': self.base_frequency,
                        'rapid_frequency': self.rapid_frequency,
                        'rapid_duration': self.rapid_duration,
                        'tensor_ready': True
                    }
                }, torch_path)
            
            self.get_logger().info(f"💾 Final save complete:")
            self.get_logger().info(f"   Total samples: {summary['total_samples']}")
            self.get_logger().info(f"   Duration: {summary['session_duration']:.1f}s")
            self.get_logger().info(f"   Motion states: {summary['motion_states']}")
            self.get_logger().info(f"   Bumper events: {summary['bumper_events']}")
            self.get_logger().info(f"   Rapid recordings: {summary['rapid_recordings']}")
            self.get_logger().info(f"   CSV: {self.csv_file_path}")
            self.get_logger().info(f"   Summary: {summary_path}")
            
        except Exception as e:
            self.get_logger().error(f"Final save failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = DataCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Data Collection stopped")
        # Save final summary and backup
        node.save_final_summary()
    finally:
        if hasattr(node, 'destroy_node'):
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 