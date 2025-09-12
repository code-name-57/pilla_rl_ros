import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # Node(
        #     package="joy",
        #     executable="joy_node",
        #     name="joy_node",
        # ),
        # Node(
        #     package="pilla_rl_ros",
        #     executable="teleop_node",
        #     name="teleop_node",
        # ),
        Node(
            package="pilla_rl_ros",
            executable="rl_policy_node",
            name="rl_policy_node",
        ),
        Node(
            package="pilla_rl_ros",
            executable="genesis_sim_node",
            name="genesis_sim_node",
        ),
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter_madgwick",
            parameters=[
                {"use_mag": False},
                {"fixed_frame": 'base'},
                {"publish_tf": False},
            ]
        ),
        
])