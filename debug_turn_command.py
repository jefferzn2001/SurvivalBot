#!/usr/bin/env python3
"""
Debug script to test TURN command processing in data_server.py
"""

import sys
import os
sys.path.append('zeroMQscripts')

# Import the DataServer class
from data_server import DataServer

def test_convert_command():
    """Test the convert_command function directly"""
    print("🔧 Testing convert_command function...")
    
    # Create a DataServer instance (won't initialize hardware)
    server = DataServer()
    server.enable_arduino = False  # Disable hardware
    
    # Test various TURN commands
    test_commands = [
        "TURN,60",
        "TURN,-30", 
        "TURN,90",
        "TURN,180",
        "FORWARD,1.0",
        "STOP"
    ]
    
    print("\n📊 Command conversion results:")
    for cmd in test_commands:
        converted = server.convert_command(cmd)
        print(f"   Input:  '{cmd}'")
        print(f"   Output: '{converted}'")
        print(f"   Match:  {cmd == converted if converted else 'None'}")
        print()

def simulate_command_flow():
    """Simulate the full command handling flow"""
    print("🔄 Simulating command flow...")
    
    server = DataServer()
    server.enable_arduino = False
    server.arduino = None  # No Arduino connected
    
    # Test what happens when we send a TURN command
    test_command = "TURN,60"
    print(f"\n📤 Simulating command: '{test_command}'")
    
    # This will show debug output but won't actually send to Arduino
    server.handle_command(test_command)

if __name__ == "__main__":
    test_convert_command()
    simulate_command_flow() 