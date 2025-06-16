#!/usr/bin/env python3
"""
Current Collection Node - Records all sensor data at 10Hz
Starts recording immediately when launched and saves to timestamped CSV file
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
        
        # CSV file for sensor data
        self.csv_file = f"{self.output_dir}/all_sensors_{timestamp}.csv"
        self.data_buffer = []
        self.csv_initialized = False
        
        # Latest sensor data
        self.latest_sensor_data = None
        
        # Collection state
        self.start_time = time.time()
        self.sample_count = 0
        
        # ROS2 Subscription
        self.sensor_sub = self.create_subscription(
            String, 'robot/sensor_data', self.sensor_callback, 10)
        
        # Timer for 10Hz data collection
        self.collection_timer = self.create_timer(0.1, self.collect_data)  # 10Hz = 0.1s
        
        # Timer for periodic CSV saves (every 5 seconds)
        self.save_timer = self.create_timer(5.0, self.save_to_csv)
        
        self.get_logger().info("📊 All Sensors Collection Node Started")
        self.get_logger().info(f"   Output directory: {self.output_dir}")
        self.get_logger().info(f"   CSV file: {self.csv_file}")
        self.get_logger().info(f"   Collection rate: 10Hz")
        self.get_logger().info(f"   Recording: All available sensor data from Arduino")
        self.get_logger().info("   Press Ctrl+C to stop and save final data")
    
    def sensor_callback(self, msg):
        """Store latest sensor data"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
    
    def collect_data(self):
        """Collect all sensor data at 10Hz"""
        if not self.latest_sensor_data:
            return
        
        # Get current timestamp
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # Start with timestamp data
        data_row = {
            'timestamp': datetime.now().isoformat(),
            'relative_time_s': round(relative_time, 3),
            'sample_count': self.sample_count,
        }
        
        # Extract current sensor data (handle both old and new formats)
        current_data = self.latest_sensor_data.get('current', {})
        if isinstance(current_data, dict):
            # New format: {"in": value, "out": value}
            data_row['current_in_A'] = round(current_data.get('in', 0.0), 4)
            data_row['current_out_A'] = round(current_data.get('out', 0.0), 4)
        else:
            # Old format: simple number (assume it's the output current)
            data_row['current_in_A'] = 0.0
            data_row['current_out_A'] = round(current_data, 4)
        
        # Extract LDR sensor data
        ldr_data = self.latest_sensor_data.get('ldr', {})
        data_row['ldr_left_raw'] = ldr_data.get('left', 0)
        data_row['ldr_right_raw'] = ldr_data.get('right', 0)
        
        # Extract IMU data
        imu_data = self.latest_sensor_data.get('imu', {})
        data_row['imu_roll_deg'] = round(imu_data.get('roll', 0.0), 2)
        data_row['imu_pitch_deg'] = round(imu_data.get('pitch', 0.0), 2)
        data_row['imu_yaw_deg'] = round(imu_data.get('yaw', 0.0), 2)
        
        # Extract encoder data
        encoders_data = self.latest_sensor_data.get('encoders', {})
        data_row['encoder_left_count'] = encoders_data.get('left', 0)
        data_row['encoder_right_count'] = encoders_data.get('right', 0)
        
        # Extract environmental data (BME280)
        environment_data = self.latest_sensor_data.get('environment', {})
        data_row['temperature_C'] = round(environment_data.get('temperature', 0.0), 1)
        data_row['humidity_percent'] = round(environment_data.get('humidity', 0.0), 1)
        data_row['pressure_hPa'] = round(environment_data.get('pressure', 0.0), 1)
        
        # Extract bumper sensor data
        bumpers_data = self.latest_sensor_data.get('bumpers', {})
        data_row['bumper_top'] = bool(bumpers_data.get('top', 0))
        data_row['bumper_bottom'] = bool(bumpers_data.get('bottom', 0))
        data_row['bumper_left'] = bool(bumpers_data.get('left', 0))
        data_row['bumper_right'] = bool(bumpers_data.get('right', 0))
        
        # Extract motion status
        data_row['motion_status'] = self.latest_sensor_data.get('motion', 'stop')
        
        # Add to buffer
        self.data_buffer.append(data_row)
        self.sample_count += 1
        
        # Log every 50 samples (every 5 seconds at 10Hz)
        if self.sample_count % 50 == 0:
            self.get_logger().info(f"📊 Sample {self.sample_count}: I_in={data_row['current_in_A']:.3f}A, I_out={data_row['current_out_A']:.3f}A, Temp={data_row['temperature_C']:.1f}°C, LDR_L={data_row['ldr_left_raw']}, LDR_R={data_row['ldr_right_raw']}")
    
    def save_to_csv(self):
        """Save buffered data to CSV file"""
        if not self.data_buffer:
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffer)
            
            if not self.csv_initialized:
                # Create new CSV file with headers
                df.to_csv(self.csv_file, index=False)
                self.csv_initialized = True
                self.get_logger().info(f"📄 CSV file created: {self.csv_file}")
            else:
                # Append to existing CSV file
                df.to_csv(self.csv_file, mode='a', header=False, index=False)
            
            # Clear buffer after saving
            buffer_size = len(self.data_buffer)
            self.data_buffer.clear()
            
            self.get_logger().info(f"💾 Saved {buffer_size} samples to CSV (Total: {self.sample_count})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV: {e}")
    
    def on_shutdown(self):
        """Save remaining data on shutdown"""
        if self.data_buffer:
            self.save_to_csv()
            self.get_logger().info(f"🛑 Final save completed - Total samples: {self.sample_count}")

def main(args=None):
    rclpy.init(args=args)
    node = CurrentCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 All Sensors Collection Node stopping...")
        node.on_shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 