#!/usr/bin/env python3
"""
Data Client for receiving sensor data and camera feed from the robot
"""

import zmq
import json
import cv2
import numpy as np
import time
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataClient:
    """
    ZeroMQ-based client that receives:
    - Sensor data from Arduino
    - Camera feed
    """
    
    def __init__(self, server_ip="10.102.200.37", data_port=5555):
        """
        Initialize the Data Client
        
        Args:
            server_ip (str): IP address or hostname of the Raspberry Pi running data_server.py
            data_port (int): Port number for data communication
        """
        self.server_ip = server_ip
        self.data_port = data_port
        
        # ZeroMQ setup
        self.context = zmq.Context()
        
        # Subscriber for sensor data and camera
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_ip}:{data_port}")
        
        # Subscribe to all topics
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Callback handlers
        self.sensor_callbacks = []
        self.camera_callbacks = []
        
        logger.info(f"📡 Data Client connected to {server_ip}:{data_port}")
    
    def add_sensor_callback(self, callback):
        """Add callback for sensor data"""
        self.sensor_callbacks.append(callback)
    
    def add_camera_callback(self, callback):
        """Add callback for camera frames"""
        self.camera_callbacks.append(callback)
    
    def process_sensor_data(self, data):
        """Process and dispatch sensor data to callbacks"""
        try:
            sensor_data = json.loads(data)
            for callback in self.sensor_callbacks:
                callback(sensor_data)
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
    
    def process_camera_frame(self, frame_data):
        """Process and dispatch camera frame to callbacks"""
        try:
            # Convert compressed image to cv2 format
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            for callback in self.camera_callbacks:
                callback(frame)
        except Exception as e:
            logger.error(f"Error processing camera frame: {e}")
    
    def run(self):
        """Main run loop"""
        try:
            while True:
                try:
                    # Receive multipart message
                    topic, data = self.sub_socket.recv_multipart()
                    topic = topic.decode()
                    
                    if topic == "sensor_data":
                        self.process_sensor_data(data)
                    elif topic == "camera":
                        self.process_camera_frame(data)
                    
                except Exception as e:
                    logger.error(f"Error receiving data: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
                    
        except KeyboardInterrupt:
            print("\n🛑 Data Client stopped")
        finally:
            self.sub_socket.close()
            self.context.term()

def example_sensor_callback(sensor_data):
    """Example callback for sensor data - prints to terminal"""
    print("\n" + "="*50)
    print("📊 SENSOR DATA:")
    print("="*50)
    
    # IMU Data - no 'working' field in Arduino JSON
    imu = sensor_data.get('imu', {})
    print(f"🧭 IMU: Roll={imu.get('roll', 0):.2f}° Pitch={imu.get('pitch', 0):.2f}° Yaw={imu.get('yaw', 0):.2f}°")
    
    # Encoders
    encoders = sensor_data.get('encoders', {})
    print(f"🔄 Encoders: Left={encoders.get('left', 0)} Right={encoders.get('right', 0)}")
    
    # Power
    power = sensor_data.get('power', {})
    power_in = power.get('in', {})
    power_out = power.get('out', {})
    print(f"⚡ Power In: {power_in.get('voltage', 0):.2f}V {power_in.get('current', 0):.2f}A")
    print(f"⚡ Power Out: {power_out.get('voltage', 0):.2f}V {power_out.get('current', 0):.2f}A")
    
    # Light sensors
    ldr = sensor_data.get('ldr', {})
    print(f"💡 Light: Left={ldr.get('left', 0)} Right={ldr.get('right', 0)}")
    
    # Environment - no 'working' field in Arduino JSON
    env = sensor_data.get('environment', {})
    print(f"🌡️ Environment: {env.get('temperature', 0):.1f}°C {env.get('humidity', 0):.1f}% {env.get('pressure', 0):.1f}hPa")
    
    # Bumpers
    bumpers = sensor_data.get('bumpers', {})
    print(f"🚧 Bumpers: Top={bumpers.get('top', 0)} Bottom={bumpers.get('bottom', 0)} Left={bumpers.get('left', 0)} Right={bumpers.get('right', 0)}")
    
    # Motion and status
    print(f"🤖 Motion: {sensor_data.get('motion', 'unknown')}")
    
    # Local machine time in milliseconds
    now = datetime.now()
    local_time = now.strftime("%H:%M:%S.%f")[:-3]  # Remove last 3 digits to get milliseconds
    print(f"⏰ Time: {local_time}")
    
    if sensor_data.get('fake', False):
        print("⚠️ Using FAKE sensor data (no Arduino connected)")
    
    print("="*50)

def example_camera_callback(frame):
    """Example callback for camera frames - display in window"""
    cv2.imshow("Robot Camera", frame)
    cv2.waitKey(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot Data Client')
    parser.add_argument('--server-ip', type=str, default='10.102.200.37',
                      help='IP address or hostname of the Raspberry Pi running data_server.py')
    parser.add_argument('--data-port', type=int, default=5555,
                      help='Port number for data communication')
    args = parser.parse_args()

    print(f"🚀 Starting Data Client...")
    print(f"📡 Connecting to server at {args.server_ip}:{args.data_port}")

    # Create client with command line arguments
    client = DataClient(server_ip=args.server_ip, data_port=args.data_port)
    
    # Add example callbacks
    client.add_sensor_callback(example_sensor_callback)
    client.add_camera_callback(example_camera_callback)
    
    print("✅ Client started! Waiting for sensor data...")
    print("Press Ctrl+C to stop")
    
    client.run()

if __name__ == '__main__':
    main() 