from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'survival_bot_nodes'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jeff',
    maintainer_email='jefferzn@seas.upenn.edu',
    description='SurvivalBot navigation and control nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_server_node = survival_bot_nodes.data_server_node:main',
            'vlm_navigation_node = survival_bot_nodes.vlm_navigation_node:main',
            'joystick_controller_node = survival_bot_nodes.joystick_controller_node:main',
            'camera_viewer_node = survival_bot_nodes.camera_viewer_node:main',
        ],
    },
) 