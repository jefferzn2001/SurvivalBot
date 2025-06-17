#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='survival_bot_nodes',
            executable='simple_command_node.py',
            name='simple_command_node',
            output='screen',
            emulate_tty=True,
        ),
    ]) 