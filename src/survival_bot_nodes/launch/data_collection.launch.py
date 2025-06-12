#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # VLM-Triggered Data Collection Node
        Node(
            package='survival_bot_nodes',
            executable='data_collection_node.py',
            name='data_collection_node',
            output='screen',
            parameters=[{
                'vlm_triggered_mode': True,
                'output_dir': './train',
                'session_name': 'vlm_session'
            }]
        )
    ]) 