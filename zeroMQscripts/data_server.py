#!/usr/bin/env python3
"""
Data Server for Pi - Arduino interface and basic camera using ZeroMQ
"""

import json
import time
import serial
import glob
import zmq
import cv2
import numpy as np
from PIL import Image
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataServer:
    """
    ZeroMQ-based data server that handles:
    - Arduino sensor data
    - Camera feed
    - Command processing
    """
    
    def __init__(self, bind_ip="*", pub_port=5555, cmd_port=5556):
        self.bind_ip = bind_ip
        self.pub_port = pub_port
        self.cmd_port = cmd_port
        
        # ZeroMQ setup
        self.context = zmq.Context()
        
        # Publisher for sensor data and camera
        self.pub_socket = self.context.socket(zmq.PUB)
        bind_addr = f"tcp://{bind_ip}:{pub_port}"
        self.pub_socket.bind(bind_addr)
        logger.info(f"Publisher bound to {bind_addr}")
        
        # Subscriber for commands
        self.cmd_socket = self.context.socket(zmq.PULL)
        cmd_addr = f"tcp://{bind_ip}:{cmd_port}"
        self.cmd_socket.bind(cmd_addr)
        logger.info(f"Command receiver bound to {cmd_addr}")
        
        # Parameters
        self.enable_arduino = True
        self.enable_camera = True
        
        # Setup Arduino connection
        self.arduino = None
        self.baudrate = 115200
        self.latest_sensor_data = {}
        self.command_in_progress = False  # Flag to prevent read interference
        
        if self.enable_arduino:
            self.setup_arduino()
        
        # Setup camera
        self.camera = None
        if self.enable_camera:
            self.setup_camera()
        
        logger.info("📡 Data Server Started")
        logger.info(f"   Arduino: {'✅' if self.arduino else '❌'}")
        logger.info(f"   Camera: {'✅' if self.camera else '❌'}")
        if self.arduino:
            logger.info(f"   Arduino Port: {self.arduino.port}")
    
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
                logger.info("✅ Camera initialized")
            else:
                self.camera = None
                logger.warning("⚠️ No camera detected")
                
        except Exception as e:
            logger.error(f"❌ Camera failed: {e}")
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
                
                # Send compressed image
                self.pub_socket.send_multipart([
                    b"camera",
                    buffer.tobytes()
                ])
                
        except Exception as e:
            logger.warning(f"Camera capture failed: {e}")
        
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
        
        logger.info(f"🔍 Searching Arduino: {possible_ports}")
        
        for port in possible_ports:
            try:
                test_serial = serial.Serial(port=port, baudrate=self.baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino boot
                
                test_serial.write(b"STATUS\n")
                time.sleep(0.5)
                
                if test_serial.in_waiting > 0:
                    response = test_serial.readline().decode('utf-8', errors='ignore').strip()
                    test_serial.close()
                    logger.info(f"✅ Found Arduino on {port}")
                    return port
                
                test_serial.close()
                
            except Exception:
                continue
        
        return None
    
    def setup_arduino(self):
        """Initialize Arduino serial connection"""
        port = self.find_arduino_port()
        
        if not port:
            logger.error("❌ No Arduino found!")
            return
        
        try:
            self.arduino = serial.Serial(port=port, baudrate=self.baudrate, timeout=0.1)
            time.sleep(3)  # Wait for Arduino boot
            
            # Clear startup messages
            while self.arduino.in_waiting:
                self.arduino.readline()
                
            logger.info(f"✅ Arduino connected on {port}")
        except Exception as e:
            logger.error(f"❌ Arduino failed: {e}")
            self.arduino = None

    def read_arduino_data(self):
        """Read sensor data from Arduino"""
        if not self.arduino or self.command_in_progress:
            # Use fake data if no Arduino or command in progress
            if not self.arduino:
                self.latest_sensor_data = {
                    "imu": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "encoders": {"left": 0, "right": 0},
                    "power": {
                        "in": {"voltage": 0.0, "current": 0.0},
                        "out": {"voltage": 0.0, "current": 0.0}
                    },
                    "ldr": {"left": 512, "right": 512},
                    "environment": {"temperature": 25.0, "humidity": 50.0, "pressure": 1013.25},
                    "bumpers": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                    "motion": "stop",
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
            logger.warning(f"Arduino read error: {e}")
    
    def publish_sensor_data(self):
        """Publish latest sensor data"""
        if self.latest_sensor_data:
            self.pub_socket.send_multipart([
                b"sensor_data",
                json.dumps(self.latest_sensor_data).encode()
            ])
    
    def handle_command(self, command):
        """Handle robot commands"""
        command = command.strip()
        logger.info(f"🤖 Command: {command}")
        
        # Debug Arduino status
        if not self.arduino:
            logger.warning("⚠️ No Arduino - command ignored")
            logger.info(f"🔍 Debug: Arduino object is None. Enable Arduino: {self.enable_arduino}")
            if self.enable_arduino:
                logger.info("🔄 Attempting to reconnect Arduino...")
                self.setup_arduino()
            return
        
        # Check if Arduino connection is still valid
        try:
            if not self.arduino.is_open:
                logger.warning("⚠️ Arduino connection closed - attempting reconnect")
                self.setup_arduino()
                return
        except Exception:
            logger.warning("⚠️ Arduino connection invalid - attempting reconnect")
            self.arduino = None
            self.setup_arduino()
            return
        
        try:
            arduino_command = self.convert_command(command)
            if arduino_command:
                # Send to Arduino with proper flushing
                self.arduino.write(f"{arduino_command}\n".encode())
                self.arduino.flush()  # Ensure command is sent immediately
                logger.info(f"   📡 Sent: {arduino_command}")
                
                # Small delay to ensure Arduino processes the command
                time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
            # Try to reconnect on command failure
            self.arduino = None
            self.setup_arduino()
    
    def convert_command(self, command):
        """Convert high-level command to Arduino format"""
        if command.startswith("FORWARD"):
            parts = command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,{distance}"
            
        elif command.startswith("BACKWARD"):
            parts = command.split(",")
            distance = parts[1] if len(parts) > 1 else "1.0"
            return f"MOVE,-{distance}"
            
        elif command.startswith("TURN"):
            return command
            
        elif command == "STOP":
            return "STOP"
            
        elif command.startswith("PWM"):
            return command
            
        else:
            return None

    def run(self):
        """Main run loop"""
        last_sensor_read = 0
        last_camera_capture = 0
        
        try:
            while True:
                # Check for commands (non-blocking)
                try:
                    command = self.cmd_socket.recv_string(flags=zmq.NOBLOCK)
                    self.handle_command(command)
                except zmq.Again:
                    pass  # No command waiting
                
                # Read Arduino data (20Hz)
                if time.time() - last_sensor_read >= 0.05:
                    self.read_arduino_data()
                    last_sensor_read = time.time()
                
                # Capture camera (10Hz)
                if self.camera and time.time() - last_camera_capture >= 0.1:
                    self.capture_camera()
                    last_camera_capture = time.time()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n🛑 Data Server stopped")
        finally:
            if self.arduino:
                self.arduino.close()
            if self.camera:
                self.camera.release()
            self.pub_socket.close()
            self.cmd_socket.close()
            self.context.term()

def main():
    parser = argparse.ArgumentParser(description='Robot Data Server')
    parser.add_argument('--bind-ip', type=str, default='*',
                      help='IP address to bind to (* for all interfaces)')
    parser.add_argument('--pub-port', type=int, default=5555,
                      help='Port number for publishing data')
    parser.add_argument('--cmd-port', type=int, default=5556,
                      help='Port number for receiving commands')
    args = parser.parse_args()
    
    server = DataServer(
        bind_ip=args.bind_ip,
        pub_port=args.pub_port,
        cmd_port=args.cmd_port
    )
    server.run()

if __name__ == '__main__':
    main() 