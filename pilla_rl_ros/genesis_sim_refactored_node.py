# !/usr/bin/env python3

from __future__ import print_function

import torch

import genesis as gs

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import math
import rclpy

# from champ_msgs.msg import Pose as PoseLite
from geometry_msgs.msg import Pose as Pose
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


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


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = False  # Disable for better policy testing
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)

        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def reset(self):
        """Reset the environment to initial state"""
        # Reset robot to default position and joint angles
        self.robot.set_pos(self.base_init_pos.unsqueeze(0), zero_velocity=True, envs_idx=torch.tensor([0], device=gs.device))
        self.robot.set_quat(self.base_init_quat.unsqueeze(0), zero_velocity=True, envs_idx=torch.tensor([0], device=gs.device))
        
        # Set joints to default positions
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0),
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=torch.tensor([0], device=gs.device),
        )
        
        # Reset state variables
        self.actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.episode_length_buf[:] = 0
        self.commands[:] = 0.0  # Set to zero initially
        
        # Ensure robot starts in a stable position
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0),
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=torch.tensor([0], device=gs.device),
        )
        
        # Step once to settle the robot
        self.scene.step()
        
        # Get initial observations
        obs, info = self.get_observations()
        return obs, info

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1

        return self.get_observations()

    def get_observations(self):
        # Update robot state first
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3 - 0,1,2
                self.projected_gravity,  # 3 - 3,4,5
                self.commands * self.commands_scale,  # 3 - 6,7,8
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12 - 9-20
                self.dof_vel * self.obs_scales["dof_vel"],  # 12 - 21-32
                self.actions,  # 12 - 33-44
            ],
            axis=-1,
        )
        
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_imu_data(self):
        orientation =  self.base_quat[0].cpu().numpy().tolist()  # [w, x, y, z]
        angular_velocity = self.base_ang_vel.cpu().numpy().tolist()  # [x, y, z]
        linear_acceleration = self.projected_gravity.cpu().numpy().tolist()  # [x, y, z]

        return orientation, angular_velocity, linear_acceleration

    def get_privileged_observations(self):
        return None

class GenesisSimNode(Node):
    def __init__(self):
        super().__init__('genesis_sim')
        
        # Use simpler QoS profile
        qos_profile = rclpy.qos.QoSProfile(depth=10)

        self.joint_command_subscriber = self.create_subscription(
            JointState,
            '/joint_commands',
            self.joint_command_callback,
            qos_profile=qos_profile
        )

        self.joint_state_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            qos_profile=qos_profile
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            '/imu/data',
            qos_profile=qos_profile
        )

        gs.init()
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

        self.env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )


        self.last_desired_velocity = Twist()
        self.step_count = 0
        
        # Reset environment and get initial observations
        obs, _ = self.env.reset()

        self.publish_observations(obs)

    def joint_command_callback(self, msg):
        # Handle incoming joint commands (if needed)
        if len(msg.position) != 12:
            self.get_logger().error(f'Expected 12 joint positions, but got {len(msg.position)}')
            return
        joint_positions = torch.tensor(msg.position, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        self.step_count += 1
        obs, info = self.env.step(joint_positions)
        self.publish_observations(obs)
    
    def publish_observations(self, obs):
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.env.env_cfg["joint_names"]
        joint_state_msg.position = obs[0, 9:21].cpu().numpy().tolist()  # Joint positions from observation
        joint_state_msg.velocity = obs[0, 21:33].cpu().numpy().tolist()  # Joint velocities from observation
        joint_state_msg.effort = []  # Effort not provided
        self.joint_state_publisher.publish(joint_state_msg)

        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.orientation.w = self.env.base_quat[0,0].item()
        imu_msg.orientation.x = self.env.base_quat[0,1].item()
        imu_msg.orientation.y = self.env.base_quat[0,2].item()
        imu_msg.orientation.z = self.env.base_quat[0,3].item()
        imu_msg.angular_velocity.x = self.env.base_ang_vel[0,0].item()
        imu_msg.angular_velocity.y = self.env.base_ang_vel[0,1].item()
        imu_msg.angular_velocity.z = self.env.base_ang_vel[0,2].item()
        imu_msg.linear_acceleration.x = self.env.projected_gravity[0,0].item()
        imu_msg.linear_acceleration.y = self.env.projected_gravity[0,1].item()
        imu_msg.linear_acceleration.z = self.env.projected_gravity[0,2].item()
        self.imu_publisher.publish(imu_msg)
        # Log progress
        if self.step_count % 50 == 0:  # Log every 50 steps
            self.get_logger().info(f'Step {self.step_count}: obs_size={len(obs[0].cpu().numpy().tolist())})')

def main():
    rclpy.init()
    genesis_sim = GenesisSimNode()
    try:
        rclpy.spin(genesis_sim)
    except KeyboardInterrupt:
        pass
    finally:
        genesis_sim.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    