#!/usr/bin/env python3
"""
Launch file for Current Collection Node
Records current and LDR sensor data at 10Hz
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='survival_bot_nodes',
            executable='current_collection_node.py',
            name='current_collection_node',
            output='screen',
            parameters=[
                # No parameters needed - uses defaults
            ]
        )
    ]) 