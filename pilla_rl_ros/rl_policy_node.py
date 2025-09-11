# !/usr/bin/env python3

from __future__ import print_function

import torch
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
        
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        
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
            Imu, 
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

    def _tick(self, joint_state_msg, imu: Imu):
        # This function is called when both joint state and IMU messages are received
        # It can be used to update internal state if needed
        
        if self.policy is None:
            self.get_logger().warn('Policy not loaded yet, skipping observation')
            return
            
        self.last_observation = np.zeros(45, dtype=np.float32)

        # self.last_observation[0:6] = imu.data[0:6]  # Assuming imu.data is a list of floats
        base_ang_vel = imu.angular_velocity
        self.last_observation[0] = base_ang_vel.x * self.obs_cfg["obs_scales"]["ang_vel"]
        self.last_observation[1] = base_ang_vel.y * self.obs_cfg["obs_scales"]["ang_vel"]
        self.last_observation[2] = base_ang_vel.z * self.obs_cfg["obs_scales"]["ang_vel"]


        projected_gravity = imu.linear_acceleration
        self.last_observation[3:6] = projected_gravity.x, projected_gravity.y, projected_gravity.z


        # Calculate default joint positions in the correct order
        default_positions = []
        for joint_name in self.env_cfg["joint_names"]:
            default_positions.append(self.env_cfg["default_joint_angles"][joint_name])
        default_positions = np.array(default_positions, dtype=np.float32)
        
        # Apply the calculation: (dof_pos - default_dof_pos) * obs_scale
        dof_pos = np.array(joint_state_msg.position[:12], dtype=np.float32)
        self.last_observation[9:21] = (dof_pos - default_positions) * self.obs_cfg["obs_scales"]["dof_pos"]
        dof_vel = np.array(joint_state_msg.velocity[:12], dtype=np.float32)
        self.last_observation[21:33] = dof_vel * self.obs_cfg["obs_scales"]["dof_vel"]

        self.action_count += 1

        # Include velocity commands in observation if available
        if self.last_velocity is not None:
            # Clip velocity commands to their respective ranges
            clipped_linear_x = np.clip(self.last_velocity.linear.x, 
                                      self.command_cfg["lin_vel_x_range"][0], 
                                      self.command_cfg["lin_vel_x_range"][1])
            clipped_linear_y = np.clip(self.last_velocity.linear.y,
                                      self.command_cfg["lin_vel_y_range"][0],
                                      self.command_cfg["lin_vel_y_range"][1])
            clipped_angular_z = np.clip(self.last_velocity.angular.z,
                                       self.command_cfg["ang_vel_range"][0],
                                       self.command_cfg["ang_vel_range"][1])
            # Append velocity commands to observation
            velocity_data = [
                clipped_linear_x * self.obs_cfg["obs_scales"]["lin_vel"],
                clipped_linear_y * self.obs_cfg["obs_scales"]["lin_vel"],
                clipped_angular_z * self.obs_cfg["obs_scales"]["ang_vel"]
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
                actions = self.policy(obs_tensor)   
                # Scale actions according to following calculations
                clipped_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]).squeeze(0).cpu().numpy()
                target_dof_pos_cpu = clipped_actions * self.env_cfg["action_scale"] + default_positions

            # Check action size
            if len(target_dof_pos_cpu) != 12:
                self.get_logger().error(f'Wrong action size: expected 12, got {len(target_dof_pos_cpu)}')
                return
                    
            # Log less frequently to reduce noise
            if self.action_count % 50 == 0:  # Log every 50th action
                self.get_logger().info(f'Action #{self.action_count}: [{target_dof_pos_cpu[0]:.3f}, {target_dof_pos_cpu[1]:.3f}, ...] obs_size: {obs_tensor.shape[1]}')
            


            # Update last_actions for next iteration
            self.last_actions = target_dof_pos_cpu
            joint_msg = JointState()
            joint_msg.name = [f'joint_{i+1}' for i in range(len(target_dof_pos_cpu))]
            joint_msg.position = target_dof_pos_cpu.tolist()
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
    