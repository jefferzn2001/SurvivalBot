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

class DataServerNode(Node):
    def __init__(self):
        super().__init__('data_server_node')
        
        # Parameters
        self.declare_parameter('enable_arduino', True)
        self.enable_arduino = self.get_parameter('enable_arduino').value
        
        # Setup Arduino connection
        self.arduino = None
        self.baudrate = 115200
        self.latest_sensor_data = {}
        
        if self.enable_arduino:
            self.setup_arduino()
        
        # Publishers
        self.sensor_pub = self.create_publisher(String, 'robot/sensor_data', 10)
        self.status_pub = self.create_publisher(String, 'robot/status', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'robot/command', self.command_callback, 10)
        
        # Timers
        self.create_timer(0.05, self.read_arduino_data)   # 20Hz Arduino reading
        
        self.get_logger().info("📡 Data Server Node Started")
        self.get_logger().info(f"   Arduino: {'✅' if self.arduino else '❌'}")
        if self.arduino:
            self.get_logger().info(f"   Port: {self.arduino.port}")
        
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
        """Publish latest sensor data"""
        if self.latest_sensor_data:
            msg = String()
            msg.data = json.dumps(self.latest_sensor_data)
            self.sensor_pub.publish(msg)
    
    def command_callback(self, msg):
        """Handle robot commands"""
        command = msg.data.strip()
        self.get_logger().info(f"🤖 Command: {command}")
        
        if not self.arduino:
            self.get_logger().warning("⚠️ No Arduino - simulating")
            time.sleep(1)
            status_msg = String()
            status_msg.data = f"COMPLETED:{command}"
            self.status_pub.publish(status_msg)
            return
        
        try:
            arduino_command = self.convert_command(command)
            if arduino_command:
                self.arduino.write(f"{arduino_command}\n".encode())
                self.get_logger().info(f"   📡 Sent: {arduino_command}")
                
                # Send completion after delay
                self.create_timer(1.0, lambda: self.send_completion(command), oneshot=True)
            
        except Exception as e:
            self.get_logger().error(f"❌ Command failed: {e}")
    
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
    
    def send_completion(self, command):
        """Send completion status"""
        status_msg = String()
        status_msg.data = f"COMPLETED:{command}"
        self.status_pub.publish(status_msg)

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