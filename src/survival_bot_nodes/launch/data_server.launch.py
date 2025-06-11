#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'camera_index',
            default_value='0',
            description='Camera device index'
        ),
        
        DeclareLaunchArgument(
            'enable_arduino',
            default_value='true',
            description='Enable Arduino connection with auto-detection'
        ),
        
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='Camera image width'
        ),
        
        DeclareLaunchArgument(
            'image_height',
            default_value='480',
            description='Camera image height'
        ),
        
        # Data Server Node with Arduino auto-detection
        Node(
            package='survival_bot_nodes',
            executable='data_server_node',
            name='data_server_node',
            output='screen',
            parameters=[{
                'camera_index': LaunchConfiguration('camera_index'),
                'enable_arduino': LaunchConfiguration('enable_arduino'),
                'image_width': LaunchConfiguration('image_width'),
                'image_height': LaunchConfiguration('image_height'),
            }],
            remappings=[
                # Add any topic remappings if needed
            ]
        )
    ]) 