# !/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import pickle
from importlib import metadata
import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        print(metadata.version("rsl-rl-lib"))
        if metadata.version("rsl-rl-lib") != "2.3.3":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# from go2_env import Go2Env
import sys
import select
import termios
import tty


import math
import rclpy
import numpy as np

# from champ_msgs.msg import Pose as PoseLite
from geometry_msgs.msg import Pose as Pose
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy')
        
        # Use simpler QoS profile for better reliability
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        
        self.velocity_subscriber = self.create_subscription(
            Twist, 
            'cmd_vel', 
            self.velocity_callback, 
            qos_profile=qos_profile
        )

        self.joint_command_publisher = self.create_publisher(
            JointState, 
            '/joint_commands', 
            qos_profile=qos_profile
        )

        self.joint_state_subscriber = Subscriber(self,
            JointState, 
            '/joint_states', 
            qos_profile=qos_profile
        )

        self.imu_subscriber = Subscriber(self,
            Float32MultiArray, 
            '/imu/data', 
            qos_profile=qos_profile
        )

        queue_size = 10
        subscribers = [self.joint_state_subscriber, self.imu_subscriber]

        # Time synchronizer to ensure joint state and IMU data are processed
        # together
        self.sync = ApproximateTimeSynchronizer(subscribers, queue_size, slop=0.1, allow_headerless=True)
        self.sync.registerCallback(self._tick)


        
        self.last_velocity = None
        self.last_observation = None
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.action_count = 0
        self.policy = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load policy after setting up publishers/subscribers
        self.load_policy('src/pilla_rl_ros/pilla_rl_ros/policy.jit')

    def velocity_callback(self, msg):
        # Currently not used, but could be integrated into observation if needed
        self.get_logger().debug(f'Received velocity command: linear_x={msg.linear.x}, linear_y={msg.linear.y}, angular_z={msg.angular.z}')
        self.last_velocity = msg

    def _tick(self, joint_state_msg, imu: Float32MultiArray):
        # This function is called when both joint state and IMU messages are received
        # It can be used to update internal state if needed
        pass
        if self.policy is None:
            self.get_logger().warn('Policy not loaded yet, skipping observation')
            return
            
        self.last_observation = np.zeros(45, dtype=np.float32)

        self.last_observation[0:6] = imu.data[0:6]  # Assuming imu.data is a list of floats

        self.last_observation[9:21] = joint_state_msg.position[:12]  # First 12 joint positions
        self.last_observation[21:33] = joint_state_msg.velocity[:12]  # First 12 joint velocities

        self.action_count += 1

        # Include velocity commands in observation if available
        if self.last_velocity is not None:
            # Append velocity commands to observation
            velocity_data = [
                self.last_velocity.linear.x * 2.0,
                self.last_velocity.linear.y * 2.0, 
                self.last_velocity.angular.z * 0.25
            ]
            # velocity_data = np.array(velocity_data, dtype=np.float32)
            self.last_observation[6] = velocity_data[0]
            self.last_observation[7] = velocity_data[1]
            self.last_observation[8] = velocity_data[2]

        self.last_observation[33:45] = self.last_actions

        try:
            obs_tensor = torch.tensor(self.last_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Check observation size
            if obs_tensor.shape[1] != 45:
                self.get_logger().error(f'Wrong observation size: expected 45, got {obs_tensor.shape[1]}')
                return
                
            with torch.no_grad():
                action = self.policy(obs_tensor).squeeze(0).cpu().numpy()  # Move to CPU for ROS message
            
            self.last_actions = action
            # Check action size  
            if len(action) != 12:
                self.get_logger().error(f'Wrong action size: expected 12, got {len(action)}')
                return
                
            # Log less frequently to reduce noise
            if self.action_count % 50 == 0:  # Log every 50th action
                self.get_logger().info(f'Action #{self.action_count}: [{action[0]:.3f}, {action[1]:.3f}, ...] obs_size: {obs_tensor.shape[1]}')
            
            # action_msg = Float32MultiArray()
            # action_msg.data = action.tolist()
            # self.action_publisher.publish(action_msg)

            joint_msg = JointState()
            joint_msg.name = [f'joint_{i+1}' for i in range(len(action))]
            joint_msg.position = action.tolist()
            self.joint_command_publisher.publish(joint_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in observation callback: {str(e)}')
            

    def load_policy(self, policy_path):
        try:
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy = self.policy.to(self.device)
            self.get_logger().info(f'Successfully loaded policy from {policy_path} on device: {self.device}')
            
            # Test the policy with a dummy input to verify it works
            dummy_obs = torch.zeros(1, 45, device=self.device)  # Expected observation size
            with torch.no_grad():
                dummy_action = self.policy(dummy_obs)
            self.get_logger().info(f'Policy test successful - input shape: {dummy_obs.shape}, output shape: {dummy_action.shape}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load policy from {policy_path}: {str(e)}')
            self.policy = None



def main():
    rclpy.init()
    rl_policy = RLPolicyNode()
    try:
        rclpy.spin(rl_policy)
    except KeyboardInterrupt:
        pass
    finally:
        rl_policy.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    