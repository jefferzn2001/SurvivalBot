#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Data Server Node
        Node(
            package='survival_bot_nodes',
            executable='data_server_node',
            name='data_server_node',
            output='screen',
            parameters=[
                {'camera_index': 0},
                {'image_width': 640},
                {'image_height': 480}
            ]
        ),
        
        # VLM Navigation Node
        Node(
            package='survival_bot_nodes',
            executable='vlm_navigation_node',
            name='vlm_navigation_node',
            output='screen',
            parameters=[
                {'goal': 'white shoe'},
                {'max_iterations': 3},
                {'navigation_interval': 10.0}
            ]
        )
    ]) 