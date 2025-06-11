#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Joystick Controller Node
        Node(
            package='survival_bot_nodes',
            executable='joystick_controller_node.py',
            name='joystick_controller_node',
            output='screen'
        )
    ]) 