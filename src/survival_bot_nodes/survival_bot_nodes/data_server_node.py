#!/usr/bin/env python3
"""
Simple Data Server Node for Pi - Arduino interface and basic camera
"""

import rclpy
from rclpy.node import Node
import json
import time
import serial
import glob
from std_msgs.msg import String

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sensor_msgs.msg import CompressedImage
    COMPRESSED_IMAGE_AVAILABLE = True
except ImportError:
    COMPRESSED_IMAGE_AVAILABLE = False

class DataServerNode(Node):
    def __init__(self):
        super().__init__('data_server_node')
        
        # Parameters
        self.declare_parameter('enable_arduino', True)
        self.declare_parameter('enable_camera', True)
        self.enable_arduino = self.get_parameter('enable_arduino').value
        self.enable_camera = self.get_parameter('enable_camera').value
        
        # Setup Arduino connection
        self.arduino = None
        self.baudrate = 115200
        self.latest_sensor_data = {}
        self.pending_command = None
        self.command_start_time = None
        self.movement_detected = False
        self.encoder_start_left = 0
        self.encoder_start_right = 0
        self.last_encoder_check_time = 0
        
        if self.enable_arduino:
            self.setup_arduino()
        
        # Setup camera
        self.camera = None
        if self.enable_camera and CV2_AVAILABLE and COMPRESSED_IMAGE_AVAILABLE:
            self.setup_camera()
        
        # Publishers
        self.sensor_pub = self.create_publisher(String, 'robot/sensor_data', 10)
        self.status_pub = self.create_publisher(String, 'robot/status', 10)
        
        if self.camera is not None:
            self.camera_pub = self.create_publisher(CompressedImage, 'robot/camera/compressed', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'robot/command', self.command_callback, 10)
        
        # Timers
        self.create_timer(0.05, self.read_arduino_data)   # 20Hz Arduino reading
        
        if self.camera is not None:
            self.create_timer(0.1, self.capture_camera)   # 10Hz camera capture
        
        self.get_logger().info("📡 Data Server Node Started")
        self.get_logger().info(f"   Arduino: {'✅' if self.arduino else '❌'}")
        self.get_logger().info(f"   Camera: {'✅' if self.camera else '❌'}")
        if self.arduino:
            self.get_logger().info(f"   Arduino Port: {self.arduino.port}")
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(1)
            
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 10)
                self.get_logger().info("✅ Camera initialized")
            else:
                self.camera = None
                self.get_logger().warning("⚠️ No camera detected")
                
        except Exception as e:
            self.get_logger().error(f"❌ Camera failed: {e}")
            self.camera = None
    
    def capture_camera(self):
        """Capture and publish camera images"""
        if self.camera is None:
            return
            
        try:
            ret, frame = self.camera.read()
            if ret:
                # Compress image
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                # Create compressed image message
                img_msg = CompressedImage()
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.format = "jpeg"
                img_msg.data = buffer.tobytes()
                
                self.camera_pub.publish(img_msg)
                
        except Exception as e:
            self.get_logger().warning(f"Camera capture failed: {e}")
        
    def find_arduino_port(self):
        """Auto-detect Arduino serial port"""
        possible_ports = [
            '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
            '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2',
        ]
        
        # Also search for any USB/ACM devices
        usb_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        possible_ports.extend(usb_ports)
        possible_ports = sorted(list(set(possible_ports)))
        
        self.get_logger().info(f"🔍 Searching Arduino: {possible_ports}")
        
        for port in possible_ports:
            try:
                test_serial = serial.Serial(port=port, baudrate=self.baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino boot
                
                test_serial.write(b"STATUS\n")
                time.sleep(0.5)
                
                if test_serial.in_waiting > 0:
                    response = test_serial.readline().decode('utf-8', errors='ignore').strip()
                    test_serial.close()
                    self.get_logger().info(f"✅ Found Arduino on {port}")
                    return port
                
                test_serial.close()
                
            except Exception:
                continue
        
        return None
    
    def setup_arduino(self):
        """Initialize Arduino serial connection"""
        port = self.find_arduino_port()
        
        if not port:
            self.get_logger().error("❌ No Arduino found!")
            return
        
        try:
            self.arduino = serial.Serial(port=port, baudrate=self.baudrate, timeout=0.1)
            time.sleep(3)  # Wait for Arduino boot
            
            # Clear startup messages
            while self.arduino.in_waiting:
                self.arduino.readline()
                
            self.get_logger().info(f"✅ Arduino connected on {port}")
        except Exception as e:
            self.get_logger().error(f"❌ Arduino failed: {e}")
            self.arduino = None

    def read_arduino_data(self):
        """Read sensor data from Arduino"""
        if not self.arduino:
            # Use fake data if no Arduino
            self.latest_sensor_data = {
                "mode": "idle",
                "timestamp": time.time(),
                "fake": True
            }
            self.publish_sensor_data()
            return
        
        try:
            while self.arduino.in_waiting:
                line = self.arduino.readline().decode('utf-8').strip()
                
                if not line:
                    continue
                    
                # Check if it's JSON sensor data
                if line.startswith('{') and line.endswith('}'):
                    try:
                        sensor_data = json.loads(line)
                        self.latest_sensor_data = sensor_data
                        self.latest_sensor_data['timestamp'] = time.time()
                        self.publish_sensor_data()
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            self.get_logger().warning(f"Arduino read error: {e}")
    
    def publish_sensor_data(self):
        """Publish latest sensor data and check for movement completion"""
        if self.latest_sensor_data:
            msg = String()
            msg.data = json.dumps(self.latest_sensor_data)
            self.sensor_pub.publish(msg)
            
            # Check for command completion based on actual robot state
            self.check_movement_completion()
    
    def check_movement_completion(self):
        """Check if robot has completed movement based on sensor data"""
        if not self.pending_command or not self.latest_sensor_data:
            return
            
        try:
            # Get current robot state
            mode = self.latest_sensor_data.get('mode', 'idle')
            encoders = self.latest_sensor_data.get('encoders', {})
            current_left = encoders.get('left', 0)
            current_right = encoders.get('right', 0)
            
            # Calculate encoder differences from start position
            left_diff = abs(current_left - self.encoder_start_left)
            right_diff = abs(current_right - self.encoder_start_right)
            
            # Detect if robot has moved significantly (threshold of 10 encoder ticks)
            if left_diff > 10 or right_diff > 10:
                self.movement_detected = True
            
            # Check if movement has completed
            command_age = time.time() - self.command_start_time if self.command_start_time else 0
            
            # Movement complete if:
            # 1. Robot mode is 'idle' (Arduino finished) AND we detected movement
            # 2. Or timeout reached
            
            movement_stopped = (mode == 'idle' and self.movement_detected)
            timeout_reached = command_age > 30.0  # Safety timeout
            
            if movement_stopped or timeout_reached:
                completion_reason = "movement stopped" if movement_stopped else "timeout"
                self.get_logger().info(f"✅ Command completed ({completion_reason}): {self.pending_command}")
                self.get_logger().info(f"   Encoder changes: L={left_diff}, R={right_diff}")
                
                # Send completion
                status_msg = String()
                status_msg.data = f"COMPLETED:{self.pending_command}"
                self.status_pub.publish(status_msg)
                
                # Reset tracking
                self.pending_command = None
                self.command_start_time = None
                self.movement_detected = False
                
        except Exception as e:
            self.get_logger().warning(f"Movement detection error: {e}")
    
    def command_callback(self, msg):
        """Handle robot commands"""
        command = msg.data.strip()
        self.get_logger().info(f"🤖 Command: {command}")
        
        if not self.arduino:
            self.get_logger().warning("⚠️ No Arduino - simulating")
            # Better simulation timing for VLM
            if "TURN" in command:
                time.sleep(4.0)  # Turn simulation
            elif "FORWARD" in command or "BACKWARD" in command:
                time.sleep(5.0)  # Movement simulation  
            elif "STOP" in command:
                time.sleep(0.5)  # Stop simulation
            else:
                time.sleep(1.0)  # Default
                
            status_msg = String()
            status_msg.data = f"COMPLETED:{command}"
            self.status_pub.publish(status_msg)
            return
        
        try:
            arduino_command = self.convert_command(command)
            if arduino_command:
                # Set encoder baseline from current position
                encoders = self.latest_sensor_data.get('encoders', {})
                self.encoder_start_left = encoders.get('left', 0)
                self.encoder_start_right = encoders.get('right', 0)
                
                # Start tracking this command
                self.pending_command = command
                self.command_start_time = time.time()
                self.movement_detected = False
                
                # Send to Arduino
                self.arduino.write(f"{arduino_command}\n".encode())
                self.get_logger().info(f"   📡 Sent: {arduino_command}")
                self.get_logger().info(f"   ⏳ Waiting for movement completion...")
                self.get_logger().info(f"   Encoder baseline: L={self.encoder_start_left}, R={self.encoder_start_right}")
                
                # For STOP commands, send immediate completion since they don't involve movement
                if "STOP" in arduino_command:
                    time.sleep(0.5)  # Brief delay for Arduino to process
                    status_msg = String()
                    status_msg.data = f"COMPLETED:{command}"
                    self.status_pub.publish(status_msg)
                    self.pending_command = None
                    self.command_start_time = None
            
        except Exception as e:
            self.get_logger().error(f"❌ Command failed: {e}")
            # Clear pending command on error
            self.pending_command = None
            self.command_start_time = None
    
    def convert_command(self, ros_command):
        """Convert ROS2 command to Arduino format"""
        if ros_command.startswith("FORWARD"):
            parts = ros_command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,{distance}"
            
        elif ros_command.startswith("BACKWARD"):
            parts = ros_command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,-{distance}"
            
        elif ros_command.startswith("TURN"):
            return ros_command
            
        elif ros_command == "STOP":
            return "STOP"
            
        elif ros_command.startswith("PWM"):
            return ros_command
            
        else:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = DataServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Data Server stopped")
    finally:
        if hasattr(node, 'arduino') and node.arduino:
            node.arduino.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 