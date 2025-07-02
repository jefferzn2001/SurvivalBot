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
    """Example callback for sensor data"""
    print("Sensor Data:", json.dumps(sensor_data, indent=2))

def example_camera_callback(frame):
    """Example callback for camera frames"""
    cv2.imshow("Robot Camera", frame)
    cv2.waitKey(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot Data Client')
    parser.add_argument('--server-ip', type=str, default='raspberrypi.local',
                      help='IP address or hostname of the Raspberry Pi running data_server.py')
    parser.add_argument('--data-port', type=int, default=5555,
                      help='Port number for data communication')
    args = parser.parse_args()

    # Create client with command line arguments
    client = DataClient(server_ip=args.server_ip, data_port=args.data_port)
    
    # Add example callbacks
    client.add_sensor_callback(example_sensor_callback)
    client.add_camera_callback(example_camera_callback)
    
    client.run()

if __name__ == '__main__':
    main() 