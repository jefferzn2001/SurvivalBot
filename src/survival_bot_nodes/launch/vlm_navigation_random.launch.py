#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from datetime import datetime

def generate_launch_description():
    # Generate single session directory for all data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"./train/data/data_random_{timestamp}"
    
    return LaunchDescription([
        # VLM-Triggered Data Collection Node (receives data from remote Pi only)
        Node(
            package='survival_bot_nodes',
            executable='data_collection_node.py',
            name='data_collection_node',
            output='screen',
            parameters=[{
                'vlm_triggered_mode': True,
                'session_name': 'random',
                'vlm_session_dir': session_dir
            }],
            env={'ROS_DOMAIN_ID': '0', 'ROBOT_HARDWARE_MODE': 'remote'}
        ),
        
        # VLM Random Navigation Node Only (sends commands to remote Pi)
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
            }],
            env={'ROS_DOMAIN_ID': '0', 'ROBOT_HARDWARE_MODE': 'remote'}
        )
    ])