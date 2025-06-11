#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Data Server - handles Arduino and camera
        Node(
            package='survival_bot_nodes',
            executable='data_server_node.py',
            name='data_server_node',
            output='screen',
            parameters=[{
                'enable_arduino': True,
                'enable_camera': True
            }]
        ),
        
        # VLM Navigation Random Node
        Node(
            package='survival_bot_nodes',
            executable='vlm_navigation_random_node.py',
            name='vlm_navigation_random_node',
            output='screen',
            parameters=[{
                'goal': 'Explore and find interesting objects',
                'max_iterations': 5.0,
                'navigation_interval': 8.0,
                'random_movements': True,
                'use_fake_data': False
            }]
        ),
        
        # Camera Viewer
        Node(
            package='survival_bot_nodes', 
            executable='camera_viewer_node.py',
            name='camera_viewer_node',
            output='screen'
        )
    ]) 