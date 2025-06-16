#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Comprehensive Sensor Collection Node - Records ALL sensors at 10Hz
        Node(
            package='survival_bot_nodes',
            executable='current_collection_node.py',
            name='current_collection_node',
            output='screen',
            parameters=[{
                # No parameters needed - starts recording immediately
            }]
        )
    ]) 