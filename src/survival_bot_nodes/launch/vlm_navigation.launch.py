#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # VLM Navigation Node Only
        Node(
            package='survival_bot_nodes',
            executable='vlm_navigation_node',
            name='vlm_navigation_node',
            output='screen'
        )
    ]) 