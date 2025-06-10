#!/usr/bin/env python3
"""
Data Server Node - Simple sensor data and camera feed provider
"""

import rclpy
from rclpy.node import Node
import json
import time
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# Optional imports
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class DataServerNode(Node):
    def __init__(self):
        super().__init__('data_server_node')
        
        # Parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        
        self.camera_index = self.get_parameter('camera_index').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        
        # Setup camera
        self.setup_camera()
        
        # Publishers
        self.sensor_pub = self.create_publisher(String, 'robot/sensor_data', 10)
        self.image_pub = self.create_publisher(CompressedImage, 'robot/camera/compressed', 10)
        self.status_pub = self.create_publisher(String, 'robot/status', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'robot/command', self.command_callback, 10)
        
        # Timers
        self.create_timer(0.1, self.publish_sensor_data)  # 10Hz
        self.create_timer(0.1, self.publish_image_data)   # 10Hz
        
        self.get_logger().info("📡 Data Server Node Started")
        self.get_logger().info(f"   Camera: {'✅' if self.camera else '❌'}")
        
    def setup_camera(self):
        """Setup camera or use mock"""
        if not CV2_AVAILABLE:
            self.get_logger().warning("OpenCV not available - using mock camera")
            self.camera = None
            return
            
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
                
            self.get_logger().info(f"Camera ready: {self.image_width}x{self.image_height}")
        except Exception as e:
            self.get_logger().warning(f"Camera failed: {e} - using mock")
            self.camera = None
    
    def get_fake_sensor_data(self):
        """Generate fake sensor data since no Arduino"""
        return {
            "imu": {"x": 0.1, "y": -0.05, "z": 9.8},
            "encoders": {"left": 1234, "right": 1240},
            "battery": 12.3,
            "temperature": 24.5,
            "timestamp": time.time(),
            "fake": True
        }
    
    def capture_image(self):
        """Capture image from camera or create mock"""
        if not self.camera:
            # Create mock image
            if CV2_AVAILABLE:
                mock_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                # Add some pattern so it's not completely black
                cv2.rectangle(mock_image, (100, 100), (540, 380), (50, 50, 50), -1)
                cv2.putText(mock_image, "MOCK CAMERA", (200, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', mock_image)
                return buffer.tobytes()
            return b""
        
        try:
            ret, frame = self.camera.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()
            return None
        except Exception as e:
            self.get_logger().warning(f"Camera capture failed: {e}")
            return None
    
    def publish_sensor_data(self):
        """Publish fake sensor data"""
        data = self.get_fake_sensor_data()
        msg = String()
        msg.data = json.dumps(data)
        self.sensor_pub.publish(msg)
    
    def publish_image_data(self):
        """Publish camera images"""
        image_data = self.capture_image()
        if image_data:
            msg = CompressedImage()
            msg.data = image_data
            msg.format = "jpeg"
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(msg)
    
    def command_callback(self, msg):
        """Handle robot commands"""
        command = msg.data.strip()
        self.get_logger().info(f"🤖 Command: {command}")
        
        # Simple command responses (no actual hardware)
        if command.startswith("TURN"):
            self.get_logger().info("   🔄 Simulating turn...")
        elif command.startswith("FORWARD"):
            self.get_logger().info("   ⬆️ Simulating forward...")
        elif command == "STOP":
            self.get_logger().info("   🛑 Simulating stop...")

def main(args=None):
    rclpy.init(args=args)
    node = DataServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 