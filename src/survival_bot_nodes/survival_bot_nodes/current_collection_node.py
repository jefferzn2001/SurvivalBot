#!/usr/bin/env python3
"""
Current Collection Node - Records all sensor data at 10Hz with precise timing
Records: current sensors, LDR sensors, IMU, encoders, environment, bumpers
For video synchronization and data analysis
"""

import rclpy
from rclpy.node import Node
import json
import time
import os
import pandas as pd
from datetime import datetime
from std_msgs.msg import String

class CurrentCollectionNode(Node):
    def __init__(self):
        super().__init__('current_collection_node')
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./sensor_data_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # CSV file for all sensor data
        self.csv_file_path = f"{self.output_dir}/sensor_data_10hz.csv"
        
        # Data collection state
        self.data_records = []
        self.recording = True  # Start recording immediately
        self.last_save_time = time.time()
        self.save_interval = 5.0  # Save to file every 5 seconds
        self.record_count = 0
        
        # Latest sensor data
        self.latest_sensor_data = None
        
        # Subscribe to sensor data
        self.sensor_sub = self.create_subscription(
            String, 'robot/sensor_data', self.sensor_callback, 10)
        
        # Timer for 10Hz data collection (0.1 seconds = 100ms)
        self.create_timer(0.1, self.collect_data_callback)
        
        # Timer for periodic saving
        self.create_timer(self.save_interval, self.save_data_callback)
        
        self.get_logger().info("🔋 Sensor Collection Node Started (10Hz)")
        self.get_logger().info(f"   Output Directory: {self.output_dir}")
        self.get_logger().info(f"   CSV File: {self.csv_file_path}")
        self.get_logger().info(f"   Recording Rate: 10Hz (every 100ms)")
        self.get_logger().info(f"   Auto-save Interval: {self.save_interval}s")
        self.get_logger().info("   Recording: ALL sensors with precise timestamps")
        
        # Initialize CSV file with headers
        self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize CSV file with headers for all sensors"""
        try:
            headers = [
                # Timing
                'timestamp_iso', 'unix_time', 'record_id',
                # Current sensors (placeholders set to 0)
                'current_in', 'current_out',
                # LDR sensors  
                'ldr_left', 'ldr_right',
                # IMU data
                'imu_roll', 'imu_pitch', 'imu_yaw',
                # Encoder data
                'encoder_left', 'encoder_right',
                # Environment data
                'temperature', 'humidity', 'pressure',
                # Bumper data
                'bumper_top', 'bumper_bottom', 'bumper_left', 'bumper_right',
                # Motion state
                'motion_state'
            ]
            
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file_path, index=False)
            self.get_logger().info(f"📄 CSV initialized with {len(headers)} columns")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CSV: {e}")
    
    def sensor_callback(self, msg):
        """Store latest sensor data"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
            self.latest_sensor_data = None
    
    def collect_data_callback(self):
        """Collect all sensor data at 10Hz with precise timing"""
        if not self.recording:
            return
            
        if self.latest_sensor_data is None:
            return
        
        try:
            self.record_count += 1
            
            # Precise timing
            current_time = datetime.now()
            timestamp_iso = current_time.isoformat()
            unix_time = time.time()
            
            # Extract current data (set to 0 as placeholders)
            current_data = self.latest_sensor_data.get('current', {})
            current_in = 0.0  # Placeholder as requested
            current_out = 0.0  # Placeholder as requested
            
            # Extract LDR data
            ldr_data = self.latest_sensor_data.get('ldr', {})
            ldr_left = ldr_data.get('left', 0)
            ldr_right = ldr_data.get('right', 0)
            
            # Extract IMU data
            imu_data = self.latest_sensor_data.get('imu', {})
            imu_roll = imu_data.get('roll', 0.0)
            imu_pitch = imu_data.get('pitch', 0.0)
            imu_yaw = imu_data.get('yaw', 0.0)
            
            # Extract encoder data
            encoder_data = self.latest_sensor_data.get('encoders', {})
            encoder_left = encoder_data.get('left', 0)
            encoder_right = encoder_data.get('right', 0)
            
            # Extract environment data
            env_data = self.latest_sensor_data.get('environment', {})
            temperature = env_data.get('temperature', 0.0)
            humidity = env_data.get('humidity', 0.0)
            pressure = env_data.get('pressure', 0.0)
            
            # Extract bumper data
            bumper_data = self.latest_sensor_data.get('bumpers', {})
            bumper_top = bumper_data.get('top', 0)
            bumper_bottom = bumper_data.get('bottom', 0)
            bumper_left = bumper_data.get('left', 0)
            bumper_right = bumper_data.get('right', 0)
            
            # Extract motion state
            motion_state = self.latest_sensor_data.get('motion', 'unknown')
            
            # Create comprehensive data record
            data_record = {
                # Timing
                'timestamp_iso': timestamp_iso,
                'unix_time': unix_time,
                'record_id': self.record_count,
                # Current sensors (placeholders)
                'current_in': current_in,
                'current_out': current_out,
                # LDR sensors
                'ldr_left': ldr_left,
                'ldr_right': ldr_right,
                # IMU data
                'imu_roll': imu_roll,
                'imu_pitch': imu_pitch,
                'imu_yaw': imu_yaw,
                # Encoder data
                'encoder_left': encoder_left,
                'encoder_right': encoder_right,
                # Environment data
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                # Bumper data
                'bumper_top': bumper_top,
                'bumper_bottom': bumper_bottom,
                'bumper_left': bumper_left,
                'bumper_right': bumper_right,
                # Motion state
                'motion_state': motion_state
            }
            
            # Add to records list
            self.data_records.append(data_record)
            
            # Log every 50 records (5 seconds at 10Hz) for feedback
            if self.record_count % 50 == 0:
                self.get_logger().info(f"📊 Record {self.record_count} - "
                                     f"LDR L/R: {ldr_left}/{ldr_right}, "
                                     f"IMU: R{imu_roll:.1f}°/P{imu_pitch:.1f}°/Y{imu_yaw:.1f}°, "
                                     f"Motion: {motion_state}")
                
        except Exception as e:
            self.get_logger().error(f"Data collection failed: {e}")
    
    def save_data_callback(self):
        """Periodically save data to CSV"""
        if not self.data_records:
            return
            
        try:
            # Create DataFrame from collected records
            df = pd.DataFrame(self.data_records)
            
            # Append to CSV file
            df.to_csv(self.csv_file_path, mode='a', header=False, index=False)
            
            records_saved = len(self.data_records)
            self.get_logger().info(f"💾 Saved {records_saved} records to CSV (Total: {self.record_count})")
            
            # Clear the records list after saving
            self.data_records = []
            
        except Exception as e:
            self.get_logger().error(f"Failed to save data: {e}")
    
    def stop_recording(self):
        """Stop recording and save final data"""
        self.recording = False
        
        # Save any remaining data
        if self.data_records:
            self.save_data_callback()
        
        self.get_logger().info(f"🛑 Recording stopped - Total records: {self.record_count}")
        self.get_logger().info(f"   Data saved to: {self.csv_file_path}")

def main(args=None):
    rclpy.init(args=args)
    node = CurrentCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n🛑 Sensor Collection Node stopped")
        node.stop_recording()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 