#!/usr/bin/env python3
"""
Data Server Node - Camera feed and Arduino interface for 4WD Robot
Interfaces with DEMO.ino Arduino code for motor control and sensor data
"""

import rclpy
from rclpy.node import Node
import json
import time
import serial
import glob
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
        self.declare_parameter('enable_arduino', True)  # Enable/disable Arduino
        
        self.camera_index = self.get_parameter('camera_index').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.enable_arduino = self.get_parameter('enable_arduino').value
        
        # Setup camera
        self.setup_camera()
        
        # Setup Arduino connection with auto-detection
        self.arduino = None
        self.serial_port = None
        self.baudrate = 115200  # Match Arduino baudrate
        self.latest_sensor_data = {}
        if self.enable_arduino:
            self.setup_arduino()
        
        # Publishers
        self.sensor_pub = self.create_publisher(String, 'robot/sensor_data', 10)
        self.image_pub = self.create_publisher(CompressedImage, 'robot/camera/compressed', 10)
        self.status_pub = self.create_publisher(String, 'robot/status', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'robot/command', self.command_callback, 10)
        
        # Timers
        self.create_timer(0.1, self.publish_image_data)   # 10Hz camera
        self.create_timer(0.05, self.read_arduino_data)   # 20Hz Arduino reading
        
        self.get_logger().info("📡 Data Server Node Started")
        self.get_logger().info(f"   Camera: {'✅' if self.camera else '❌'}")
        self.get_logger().info(f"   Arduino: {'✅' if self.arduino else '❌'}")
        if self.arduino:
            self.get_logger().info(f"   Port: {self.serial_port} @ {self.baudrate}")
        
    def find_arduino_port(self):
        """Auto-detect Arduino serial port"""
        # Common Arduino ports on Linux
        possible_ports = [
            '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
            '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2',
        ]
        
        # Also search for any USB/ACM devices
        usb_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        possible_ports.extend(usb_ports)
        
        # Remove duplicates and sort
        possible_ports = sorted(list(set(possible_ports)))
        
        self.get_logger().info(f"🔍 Searching for Arduino on ports: {possible_ports}")
        
        for port in possible_ports:
            try:
                # Try to open the port
                test_serial = serial.Serial(port=port, baudrate=self.baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino boot
                
                # Send a test command and look for response
                test_serial.write(b"STATUS\n")
                time.sleep(0.5)
                
                # Check if we get some response (Arduino should respond)
                if test_serial.in_waiting > 0:
                    response = test_serial.readline().decode('utf-8', errors='ignore').strip()
                    test_serial.close()
                    
                    self.get_logger().info(f"✅ Found Arduino on {port} (response: {response[:20]}...)")
                    return port
                
                test_serial.close()
                
            except Exception as e:
                # Port not available or not Arduino
                continue
        
        return None
    
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
    
    def setup_arduino(self):
        """Initialize Arduino serial connection with auto-detection"""
        # Try to find Arduino port automatically
        self.serial_port = self.find_arduino_port()
        
        if not self.serial_port:
            self.get_logger().error("❌ No Arduino found!")
            self.get_logger().info("   Check:")
            self.get_logger().info("   - Arduino is plugged in and powered")
            self.get_logger().info("   - User has permission: sudo usermod -a -G dialout $USER")
            self.get_logger().info("   - Try: ls /dev/tty* | grep -E '(USB|ACM)'")
            self.arduino = None
            return
        
        try:
            self.arduino = serial.Serial(port=self.serial_port, baudrate=self.baudrate, timeout=0.1)
            time.sleep(3)  # Wait for Arduino to boot up completely
            
            # Clear any startup messages
            while self.arduino.in_waiting:
                self.arduino.readline()
                
            self.get_logger().info(f"✅ Arduino connected on {self.serial_port}")
        except Exception as e:
            self.get_logger().error(f"❌ Arduino connection failed: {e}")
            self.arduino = None

    def read_arduino_data(self):
        """Read sensor data from Arduino"""
        if not self.arduino:
            # Use fake data if no Arduino
            self.latest_sensor_data = self.get_fake_sensor_data()
            self.publish_sensor_data()
            return
        
        try:
            # Read all available lines
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
                        # Not valid JSON, might be status message
                        if "STATUS:" in line:
                            self.get_logger().info(f"Arduino status: {line}")
                        elif line == "READY":
                            self.get_logger().info("🤖 Arduino ready!")
                        
        except Exception as e:
            self.get_logger().warning(f"Arduino read error: {e}")

    def get_fake_sensor_data(self):
        """Generate fake sensor data when no Arduino"""
        return {
            "imu": {"roll": 0.1, "pitch": -0.05, "yaw": 0.0, "working": False},
            "encoders": {"left": 1234, "right": 1240, "distance": 1237},
            "current": 0.5,
            "ldr": {"left": 512, "right": 498},
            "environment": {"temperature": 24.5, "humidity": 60.0, "pressure": 1013.2, "working": False},
            "bumpers": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "mode": "idle",
            "timestamp": time.time(),
            "fake": True
        }
    
    def publish_sensor_data(self):
        """Publish latest sensor data"""
        if self.latest_sensor_data:
            msg = String()
            msg.data = json.dumps(self.latest_sensor_data)
            self.sensor_pub.publish(msg)
    
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
        """Handle robot commands and send to Arduino"""
        command = msg.data.strip()
        self.get_logger().info(f"🤖 Command: {command}")
        
        if not self.arduino:
            self.get_logger().warning("⚠️ No Arduino connection - simulating")
            self.simulate_command(command)
            return
        
        try:
            arduino_command = self.convert_to_arduino_command(command)
            if arduino_command:
                self.arduino.write(f"{arduino_command}\n".encode())
                self.get_logger().info(f"   📡 Sent to Arduino: {arduino_command}")
                
                # Send completion status after a delay
                self.create_timer(1.0, lambda: self.send_completion_status(command), oneshot=True)
            
        except Exception as e:
            self.get_logger().error(f"❌ Command execution failed: {e}")
    
    def convert_to_arduino_command(self, ros_command):
        """Convert ROS2 command to Arduino command format"""
        if ros_command.startswith("FORWARD"):
            # Convert FORWARD,1.0 to MOVE,1.0
            parts = ros_command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,{distance}"
            
        elif ros_command.startswith("BACKWARD"):
            # Convert BACKWARD,1.0 to MOVE,-1.0 (negative distance)
            parts = ros_command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,-{distance}"
            
        elif ros_command.startswith("TURN"):
            # TURN commands are the same format
            return ros_command
            
        elif ros_command == "STOP":
            # STOP command is the same
            return "STOP"
            
        elif ros_command.startswith("PWM"):
            # PWM commands pass through (for joystick manual control)
            return ros_command
            
        elif ros_command == "TURN_LEFT":
            # Direct Arduino command for joystick
            return "TURN_LEFT"
            
        elif ros_command == "TURN_RIGHT":
            # Direct Arduino command for joystick
            return "TURN_RIGHT"
            
        else:
            self.get_logger().warning(f"Unknown command: {ros_command}")
            return None
    
    def send_completion_status(self, original_command):
        """Send completion status back to ROS2"""
        status_msg = String()
        status_msg.data = f"COMPLETED:{original_command}"
        self.status_pub.publish(status_msg)
        self.get_logger().info(f"✅ Command completed: {original_command}")
    
    def simulate_command(self, command):
        """Simulate command execution without Arduino"""
        if command.startswith("TURN"):
            parts = command.split(",")
            angle = parts[1] if len(parts) > 1 else "0"
            self.get_logger().info(f"   🔄 Simulating turn {angle}°...")
            time.sleep(2.0)
            
        elif command.startswith("FORWARD"):
            parts = command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            self.get_logger().info(f"   ⬆️ Simulating forward {distance}m...")
            time.sleep(3.0)
            
        elif command.startswith("BACKWARD"):
            parts = command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            self.get_logger().info(f"   ⬇️ Simulating backward {distance}m...")
            time.sleep(3.0)
            
        elif command == "STOP":
            self.get_logger().info("   🛑 Simulating stop...")
            time.sleep(0.5)
            
        elif command.startswith("PWM"):
            self.get_logger().info("   🎮 Simulating PWM control...")
            time.sleep(0.1)
            
        elif command in ["TURN_LEFT", "TURN_RIGHT"]:
            self.get_logger().info(f"   🔄 Simulating {command.lower()}...")
            time.sleep(1.0)
        
        # Send simulated completion
        status_msg = String()
        status_msg.data = f"COMPLETED:{command}"
        self.status_pub.publish(status_msg)
        self.get_logger().info(f"   ✅ Simulated: {command}")

def main(args=None):
    rclpy.init(args=args)
    node = DataServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Data Server stopped")
    finally:
        # Close Arduino connection
        if hasattr(node, 'arduino') and node.arduino:
            node.arduino.close()
            print("🔌 Arduino connection closed")
        if hasattr(node, 'camera') and node.camera:
            node.camera.release()
            print("📷 Camera released")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 