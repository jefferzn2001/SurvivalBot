#!/usr/bin/env python3
"""
Command Client for sending control commands to the robot
"""

import zmq
import time
import logging

logger = logging.getLogger(__name__)

class CommandClient:
    """
    ZeroMQ-based client for sending commands to the robot
    """
    
    def __init__(self, server_ip="10.102.244.88", cmd_port=5556):
        self.server_ip = server_ip
        self.cmd_port = cmd_port
        
        # ZeroMQ setup
        self.context = zmq.Context()
        
        # Publisher for commands
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{server_ip}:{cmd_port}")
        
        # Small delay to allow connection to establish
        time.sleep(0.1)
        
        logger.info(f"📡 Command Client connected to {server_ip}:{cmd_port}")
    
    def send_command(self, command):
        """Send a command to the robot"""
        try:
            self.pub_socket.send_string(command)
            logger.info(f"Sent command: {command}")
        except Exception as e:
            logger.error(f"Error sending command: {e}")
    
    def move_forward(self, distance=1.0):
        """Move robot forward by distance (meters)"""
        self.send_command(f"FORWARD,{distance}")
    
    def move_backward(self, distance=1.0):
        """Move robot backward by distance (meters)"""
        self.send_command(f"BACKWARD,{distance}")
    
    def turn(self, angle):
        """Turn robot by angle (degrees)"""
        self.send_command(f"TURN,{angle}")
    
    def stop(self):
        """Emergency stop"""
        self.send_command("STOP")
    
    def set_pwm(self, left, right):
        """Set direct PWM values (-255 to 255)"""
        self.send_command(f"PWM,{left},{right}")
    
    def close(self):
        """Close connection"""
        self.pub_socket.close()
        self.context.term()

def main():
    # Example usage
    client = CommandClient()
    
    try:
        while True:
            command = input("Enter command (or 'quit' to exit): ")
            if command.lower() == 'quit':
                break
            client.send_command(command)
            
    except KeyboardInterrupt:
        print("\n🛑 Command Client stopped")
    finally:
        client.close()

if __name__ == '__main__':
    main() 