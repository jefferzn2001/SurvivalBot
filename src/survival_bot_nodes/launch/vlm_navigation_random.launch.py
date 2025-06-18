#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from datetime import datetime

def generate_launch_description():
    # Generate single session directory for all data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"./train/data/data_random_{timestamp}"
    
    return LaunchDescription([
        # VLM-Triggered Data Collection Node
        Node(
            package='survival_bot_nodes',
            executable='data_collection_node.py',
            name='data_collection_node',
            output='screen',
            parameters=[{
                'vlm_triggered_mode': True,
                'session_name': 'random',
                'vlm_session_dir': session_dir
            }]
        ),
        
        # VLM Random Navigation Node
        Node(
            package='survival_bot_nodes',
            executable='vlm_navigation_random_node.py',
            name='vlm_navigation_random_node',
            output='screen',
            parameters=[{
                'goal': 'Max Sunlight Location',
                'max_iterations': 10.0,
                'navigation_interval': 15.0,
                'vlm_session_dir': session_dir
            }]
        )
    ])