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


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 2000.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": False,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy')
        
        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = get_cfgs()

        self.default_dof_pos = np.array(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=np.float32)
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
            '/imu/data2', 
            qos_profile=qos_profile
        )

        self.imu_raw_subscriber = Subscriber(self,
            Imu, 
            '/imu/data', 
            qos_profile=qos_profile
        )

        queue_size = 10
        subscribers = [self.joint_state_subscriber, self.imu_raw_subscriber]

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
        self.decimate_factor = 2  # Not used currently
        # Load policy after setting up publishers/subscribers
        self.load_policy('src/pilla_rl_ros/pilla_rl_ros/policy.jit')

    def velocity_callback(self, msg):
        # Currently not used, but could be integrated into observation if needed
        self.get_logger().debug(f'Received velocity command: linear_x={msg.linear.x}, linear_y={msg.linear.y}, angular_z={msg.angular.z}')
        self.last_velocity = msg
    
    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Input quaternion (w, x, y, z).

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def imu_to_rl_obs(self, imu: Imu):
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix
        # (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z
        ])
        
        # Simple integration to estimate velocity
        # self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b
        
        # Extract angular velocity
        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])
        
        # Calculate gravity direction in body frame
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Initialize observation vector
        obs = np.zeros(6, dtype=np.float32)
        
        # Fill observation vector components:
        # Base linear velocity (3)
        # obs[:3] = self._lin_vel_b

        # Base angular velocity (3)
        obs[0:3] = ang_vel_b

        # Gravity direction (3)
        obs[3:6] = gravity_b

        return obs

    def compile_observation(self, joint_state_msg, imu_raw: Imu):
        # This function is called when both joint state and IMU messages are received
        # It can be used to update internal state if needed

        self.last_observation = np.zeros(45, dtype=np.float32)
        # self.last_observation[0:6] = imu.data[0:6]
        self.last_observation[0:6] = self.imu_to_rl_obs(imu_raw)
        # self.get_logger().debug(f'Raw IMU data: {imu_raw.data[0:6]}')
        self.last_observation[0:3] = self.last_observation[0:3] * self.obs_cfg["obs_scales"]["ang_vel"]  # 6,7,8
        self.last_observation[9:21] = joint_state_msg.position[:12]  # First 12 joint positions
        self.last_observation[21:33] = joint_state_msg.velocity[:12]  # First 12 joint velocities
        self.last_observation[9:21] = (self.last_observation[9:21] - self.default_dof_pos) * self.obs_cfg["obs_scales"]["dof_pos"]
        self.last_observation[21:33] = self.last_observation[21:33] * self.obs_cfg["obs_scales"]["dof_vel"]
        # Include velocity commands in observation if available
        if self.last_velocity is not None:
            # Append velocity commands to observation
            velocity_data = [
                self.last_velocity.linear.x * self.obs_cfg["obs_scales"]["lin_vel"],
                self.last_velocity.linear.y * self.obs_cfg["obs_scales"]["lin_vel"],
                self.last_velocity.angular.z * self.obs_cfg["obs_scales"]["ang_vel"]
            ]
            # velocity_data = np.array(velocity_data, dtype=np.float32)
            self.last_observation[6] = velocity_data[0]
            self.last_observation[7] = velocity_data[1]
            self.last_observation[8] = velocity_data[2]
        self.last_observation[33:45] = self.last_actions  # Last actions taken

        return self.last_observation

    def publish_action(self, action):
        exec_actions = np.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = (exec_actions * self.env_cfg["action_scale"]) + self.default_dof_pos
        target_dof_pos = target_dof_pos
        joint_msg = JointState()
        joint_msg.name = [f'joint_{i+1}' for i in range(len(target_dof_pos))]
        joint_msg.position = target_dof_pos.tolist()
        self.joint_command_publisher.publish(joint_msg)


    def _tick(self, joint_state_msg, imu_raw: Imu):
        # This function is called when both joint state and IMU messages are received
        # It can be used to update internal state if needed
        if self.policy is None:
            self.get_logger().warn('Policy not loaded yet, skipping observation')
            return

        self.compile_observation(joint_state_msg, imu_raw)

        try:
            obs_tensor = torch.tensor(self.last_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Check observation size
            if obs_tensor.shape[1] != 45:
                self.get_logger().error(f'Wrong observation size: expected 45, got {obs_tensor.shape[1]}')
                return

            if self.action_count % self.decimate_factor == 0:  
                with torch.no_grad():
                    action = self.policy(obs_tensor).squeeze(0).cpu().numpy()
            
                    self.last_actions = action

                self.publish_action(action)
            else:
                self.publish_action(self.last_actions)

            self.action_count += 1
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
    