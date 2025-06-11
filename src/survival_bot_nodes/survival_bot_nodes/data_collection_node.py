#!/usr/bin/env python3
"""
Data Collection Node - Intelligent training data collection
Collects 10 data points from VLM command start to robot stop, with more during turns
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
        self.declare_parameter('save_interval', 30.0)  # Save every 30 seconds
        
        self.output_dir = self.get_parameter('output_dir').value
        self.save_interval = self.get_parameter('save_interval').value
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"{self.output_dir}/session_{timestamp}"
        os.makedirs(f"{self.session_dir}/images", exist_ok=True)
        os.makedirs(f"{self.session_dir}/data", exist_ok=True)
        
        # Data storage
        self.data_buffer = []
        self.data_counter = 0
        
        # Latest data
        self.latest_image = None
        self.latest_sensor_data = None
        self.latest_vlm_action = None
        self.latest_command = None
        
        # Collection strategy - collect 10 points from command start to stop
        self.collection_active = False
        self.collection_start_time = None
        self.target_collection_count = 10
        self.current_collection_count = 0
        self.last_motion_state = "stop"
        self.command_in_progress = False
        
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
        
        # Timer for intelligent data collection
        self.collection_timer = self.create_timer(0.5, self.intelligent_collect_data)  # 2Hz collection rate
        
        # Timer for saving data
        self.save_timer = self.create_timer(self.save_interval, self.save_data_batch)
        
        self.get_logger().info("📊 Data Collection Node Started")
        self.get_logger().info(f"   Output: {self.session_dir}")
        self.get_logger().info(f"   Strategy: 10 points per VLM command (start to stop)")
        self.get_logger().info(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    
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
        """Store latest sensor data and track motion state"""
        try:
            self.latest_sensor_data = json.loads(msg.data)
            
            # Track motion state changes
            current_motion = self.latest_sensor_data.get('motion', 'stop')
            
            # Detect motion state transitions
            if self.last_motion_state != current_motion:
                if current_motion == "moving" and self.last_motion_state == "stop":
                    # Robot started moving - start collection if command was issued
                    if self.command_in_progress:
                        self.start_data_collection()
                elif current_motion == "stop" and self.last_motion_state == "moving":
                    # Robot stopped moving - end collection
                    if self.collection_active:
                        self.end_data_collection()
                
                self.last_motion_state = current_motion
                
        except Exception as e:
            self.get_logger().error(f"Sensor data parse failed: {e}")
    
    def command_callback(self, msg):
        """Store latest robot command and initiate collection"""
        self.latest_command = msg.data
        
        # Parse command to extract action information
        try:
            if msg.data.startswith("TURN"):
                parts = msg.data.split(",")
                angle = float(parts[1]) if len(parts) > 1 else 0.0
                self.latest_random_action = {"type": "turn", "angle": angle, "distance": 0.0}
                self.command_in_progress = True
                # For turns, collect more data points
                self.target_collection_count = 15
                
            elif msg.data.startswith("FORWARD"):
                parts = msg.data.split(",")
                distance = float(parts[1]) if len(parts) > 1 else 1.0
                self.latest_random_action = {"type": "forward", "angle": 0.0, "distance": distance}
                self.command_in_progress = True
                self.target_collection_count = 10
                
            elif msg.data == "STOP":
                self.latest_random_action = {"type": "stop", "angle": 0.0, "distance": 0.0}
                self.command_in_progress = False
                if self.collection_active:
                    self.end_data_collection()
                    
        except Exception as e:
            self.get_logger().error(f"Command parse failed: {e}")
    
    def vlm_decision_callback(self, msg):
        """Store latest VLM decision"""
        try:
            vlm_data = json.loads(msg.data)
            self.latest_vlm_action = vlm_data
            # VLM decision indicates new command cycle
            self.command_in_progress = True
        except Exception as e:
            self.get_logger().error(f"VLM decision parse failed: {e}")
    
    def start_data_collection(self):
        """Start intelligent data collection for this command cycle"""
        if not self.collection_active:
            self.collection_active = True
            self.collection_start_time = time.time()
            self.current_collection_count = 0
            self.get_logger().info(f"📊 Started collecting {self.target_collection_count} data points")
    
    def end_data_collection(self):
        """End data collection for this command cycle"""
        if self.collection_active:
            collection_duration = time.time() - self.collection_start_time
            self.collection_active = False
            self.command_in_progress = False
            self.get_logger().info(f"📊 Collection complete: {self.current_collection_count}/{self.target_collection_count} points in {collection_duration:.2f}s")
    
    def intelligent_collect_data(self):
        """Intelligent data collection - only when robot is active"""
        # Only collect if:
        # 1. Collection is active (command in progress)
        # 2. Robot is moving OR we haven't reached target count
        # 3. We have all required data
        
        if not self.collection_active:
            return
            
        if (self.latest_image is not None and 
            self.latest_sensor_data is not None):
            
            # Determine if we should collect this point
            current_motion = self.latest_sensor_data.get('motion', 'stop')
            should_collect = False
            
            if current_motion == "moving":
                # Always collect when moving
                should_collect = True
            elif self.current_collection_count == 0:
                # Always collect first point (command start)
                should_collect = True
            elif self.current_collection_count < self.target_collection_count and current_motion == "stop":
                # Collect final point when stopped
                should_collect = True
                
            if should_collect and self.current_collection_count < self.target_collection_count:
                try:
                    # Save image
                    timestamp = datetime.now()
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    image_filename = f"img_{self.data_counter:06d}_{timestamp_str}.jpg"
                    image_path = f"{self.session_dir}/images/{image_filename}"
                    cv2.imwrite(image_path, self.latest_image)
                    
                    # Categorize motion state
                    motion_category = "moving" if current_motion == "moving" else "stationary"
                    
                    # Prepare data point
                    data_point = {
                        'timestamp': timestamp.isoformat(),
                        'data_id': self.data_counter,
                        'collection_point': self.current_collection_count + 1,
                        'target_points': self.target_collection_count,
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'image_shape': self.latest_image.shape,
                        'sensor_data': self.latest_sensor_data.copy(),
                        'vlm_action': self.latest_vlm_action.copy() if self.latest_vlm_action else None,
                        'random_action': getattr(self, 'latest_random_action', None),
                        'latest_command': self.latest_command,
                        'motion_state': motion_category,  # Categorical: moving/stationary
                        'raw_motion': current_motion,     # Raw Arduino motion string
                    }
                    
                    # Convert image to tensor format for neural network
                    if TORCH_AVAILABLE:
                        # Normalize image to [0,1] and convert to CHW format
                        image_tensor = torch.from_numpy(self.latest_image).float() / 255.0
                        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
                        data_point['image_tensor_shape'] = image_tensor.shape
                    else:
                        # Store as numpy array
                        data_point['image_array'] = self.latest_image.copy()
                    
                    self.data_buffer.append(data_point)
                    self.data_counter += 1
                    self.current_collection_count += 1
                    
                    self.get_logger().info(f"📊 Collected point {self.current_collection_count}/{self.target_collection_count} ({motion_category})")
                    
                    # End collection if we've reached target
                    if self.current_collection_count >= self.target_collection_count:
                        self.end_data_collection()
                    
                except Exception as e:
                    self.get_logger().error(f"Data collection failed: {e}")
    
    def save_data_batch(self):
        """Save collected data to disk"""
        if not self.data_buffer:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create DataFrame
            df_data = []
            for data_point in self.data_buffer:
                # Flatten sensor data
                sensor_flat = self.flatten_dict(data_point['sensor_data'], 'sensor_')
                
                # Flatten VLM action
                vlm_flat = self.flatten_dict(data_point['vlm_action'], 'vlm_') if data_point['vlm_action'] else {}
                
                # Flatten random action
                random_flat = self.flatten_dict(data_point['random_action'], 'random_') if data_point['random_action'] else {}
                
                # Combine all data
                row = {
                    'timestamp': data_point['timestamp'],
                    'data_id': data_point['data_id'],
                    'collection_point': data_point['collection_point'],
                    'target_points': data_point['target_points'],
                    'image_filename': data_point['image_filename'],
                    'image_height': data_point['image_shape'][0],
                    'image_width': data_point['image_shape'][1],
                    'image_channels': data_point['image_shape'][2],
                    'latest_command': data_point['latest_command'],
                    'motion_state': data_point['motion_state'],      # Categorical
                    'raw_motion': data_point['raw_motion'],          # Raw string
                }
                
                row.update(sensor_flat)
                row.update(vlm_flat)
                row.update(random_flat)
                
                df_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            
            # Save DataFrame
            csv_path = f"{self.session_dir}/data/batch_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save as pickle for faster loading
            pickle_path = f"{self.session_dir}/data/batch_{timestamp}.pkl"
            df.to_pickle(pickle_path)
            
            # If PyTorch available, save as tensor dataset
            if TORCH_AVAILABLE:
                torch_path = f"{self.session_dir}/data/batch_{timestamp}.pt"
                # Create tensor dataset (simplified - just save the DataFrame)
                torch.save({
                    'dataframe': df,
                    'metadata': {
                        'session_dir': self.session_dir,
                        'collection_time': timestamp,
                        'num_samples': len(df),
                        'image_format': 'jpg',
                        'tensor_ready': True,
                        'collection_strategy': 'intelligent_10_point'
                    }
                }, torch_path)
            
            # Count motion states
            motion_counts = df['motion_state'].value_counts().to_dict()
            collection_points = len(df['collection_point'].unique()) if 'collection_point' in df.columns else 0
            
            self.get_logger().info(f"💾 Saved {len(self.data_buffer)} samples:")
            self.get_logger().info(f"   Motion states: {motion_counts}")
            self.get_logger().info(f"   Collection cycles: {collection_points}")
            self.get_logger().info(f"   CSV: {csv_path}")
            self.get_logger().info(f"   Pickle: {pickle_path}")
            if TORCH_AVAILABLE:
                self.get_logger().info(f"   PyTorch: {torch_path}")
            
            # Clear buffer
            self.data_buffer.clear()
            
        except Exception as e:
            self.get_logger().error(f"Data save failed: {e}")
    
    def flatten_dict(self, d, prefix=''):
        """Flatten nested dictionary for DataFrame"""
        if not d:
            return {}
        
        flattened = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, dict):
                flattened.update(self.flatten_dict(value, f"{new_key}_"))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    flattened[f"{new_key}_{i}"] = item
            else:
                flattened[new_key] = value
        return flattened
    
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

def main(args=None):
    rclpy.init(args=args)
    
    node = DataCollectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Data Collection stopped")
        # Save any remaining data
        if node.data_buffer:
            node.save_data_batch()
            print(f"💾 Final save: {len(node.data_buffer)} samples")
    finally:
        if hasattr(node, 'destroy_node'):
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 