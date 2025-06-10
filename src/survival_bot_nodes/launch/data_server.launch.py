#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Get install path
    pkg_share = FindPackageShare('survival_bot_nodes').find('survival_bot_nodes')
    install_dir = os.path.dirname(os.path.dirname(pkg_share))
    
    return LaunchDescription([
        # Data Server Node Only
        ExecuteProcess(
            cmd=[os.path.join(install_dir, 'bin', 'data_server_node')],
            name='data_server_node',
            output='screen'
        )
    ]) 