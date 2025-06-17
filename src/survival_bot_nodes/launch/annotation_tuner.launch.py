#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Simple Annotation Viewer - camera with original annotation + terminal commands
        # (Data server runs on Raspberry Pi)
        Node(
            package='survival_bot_nodes',
            executable='annotation_tuner_node.py',
            name='annotation_tuner_node',
            output='screen'
        )
    ]) 