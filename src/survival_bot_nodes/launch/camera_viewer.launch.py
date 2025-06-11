#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera Viewer Node
        Node(
            package='survival_bot_nodes',
            executable='camera_viewer_node.py',
            name='camera_viewer_node',
            output='screen'
        )
    ]) 