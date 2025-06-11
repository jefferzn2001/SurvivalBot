#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'enable_arduino',
            default_value='true',
            description='Enable Arduino connection with auto-detection'
        ),
        
        # Simple Data Server Node
        Node(
            package='survival_bot_nodes',
            executable='data_server_node.py',
            name='data_server_node',
            output='screen',
            parameters=[{
                'enable_arduino': LaunchConfiguration('enable_arduino'),
            }]
        )
    ]) 