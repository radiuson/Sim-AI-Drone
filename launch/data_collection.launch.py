#!/usr/bin/env python3
"""
ROS2 Launch file for Liftoff Data Collection System

Launches:
1. liftoff_bridge_ros2 - ROS2 bridge for Liftoff telemetry
2. liftoff_capture - Data collection node with OBS capture

Usage:
    ros2 launch ai_drone data_collection.launch.py

Optional arguments:
    output_dir:=/path/to/dataset
    fps:=30
    image_size:=224
    enable_gamepad:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
import os
from pathlib import Path


def generate_launch_description():
    # Get the package directory
    pkg_dir = Path(__file__).parent.parent.absolute()

    # Declare launch arguments
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=str(pkg_dir / 'dataset' / 'liftoff_data'),
        description='Output directory for dataset'
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Data collection FPS'
    )

    image_size_arg = DeclareLaunchArgument(
        'image_size',
        default_value='224',
        description='Image size (square)'
    )

    capture_method_arg = DeclareLaunchArgument(
        'capture_method',
        default_value='obs',
        description='Screen capture method (obs or mss)'
    )

    obs_device_arg = DeclareLaunchArgument(
        'obs_device',
        default_value='/dev/video10',
        description='OBS virtual camera device path'
    )

    enable_gamepad_arg = DeclareLaunchArgument(
        'enable_gamepad',
        default_value='true',
        description='Enable RadioMaster gamepad control'
    )

    bindings_file_arg = DeclareLaunchArgument(
        'bindings_file',
        default_value=str(pkg_dir / 'record' / 'control_bindings.json'),
        description='Gamepad control bindings file'
    )

    # Launch liftoff_bridge_ros2 node
    bridge_node = Node(
        package='ai_drone',  # This will need to be adjusted based on your ROS2 package name
        executable='liftoff_bridge_ros2.py',
        name='liftoff_bridge',
        output='screen',
        parameters=[
            {'host': '127.0.0.1'},
            {'port': 30001},
            {'print_rate_hz': 2.0}
        ]
    )

    # Alternative: Launch bridge as Python script directly
    bridge_process = ExecuteProcess(
        cmd=[
            'python3',
            str(pkg_dir / 'liftoff_bridge_ros2.py')
        ],
        name='liftoff_bridge',
        output='screen',
        shell=False
    )

    # Launch data collection node
    capture_process = ExecuteProcess(
        cmd=[
            'python3', '-m', 'record.liftoff_capture',
            '--output-dir', LaunchConfiguration('output_dir'),
            '--fps', LaunchConfiguration('fps'),
            '--image-size', LaunchConfiguration('image_size'),
            '--capture-method', LaunchConfiguration('capture_method'),
            '--obs-device', LaunchConfiguration('obs_device'),
            '--bindings-file', LaunchConfiguration('bindings_file')
        ],
        cwd=str(pkg_dir),
        name='liftoff_capture',
        output='screen',
        shell=False
    )

    # Launch info message
    launch_info = LogInfo(
        msg=[
            '\n',
            '='*70, '\n',
            'Liftoff Data Collection System Started\n',
            '='*70, '\n',
            'Components:\n',
            '  - ROS2 Bridge: Listening on UDP port 30001\n',
            '  - Data Capture: Saving to ', LaunchConfiguration('output_dir'), '\n',
            '\n',
            'Controls (RadioMaster):\n',
            '  - SH switch UP: Start recording\n',
            '  - SA switch UP: Stop recording\n',
            '  - BTN_SOUTH: Emergency stop\n',
            '\n',
            'Press Ctrl+C to stop all nodes\n',
            '='*70, '\n'
        ]
    )

    return LaunchDescription([
        # Declare arguments
        output_dir_arg,
        fps_arg,
        image_size_arg,
        capture_method_arg,
        obs_device_arg,
        enable_gamepad_arg,
        bindings_file_arg,

        # Launch info
        launch_info,

        # Launch nodes
        bridge_process,  # Use this instead of bridge_node if not using ROS2 package
        capture_process
    ])
