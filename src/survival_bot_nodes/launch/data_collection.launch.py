#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Data Collection Node
        Node(
            package='survival_bot_nodes',
            executable='data_collection_node.py',
            name='data_collection_node',
            output='screen'
        )
    ]) 