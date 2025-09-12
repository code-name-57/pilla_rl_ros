# !/usr/bin/env python3

from __future__ import print_function

import torch

import genesis as gs

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.sensors.imu import IMUOptions

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu

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
    def __init__(self, num_envs, env_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]

        self.dt = 0.02  # control frequency on real robot is 50hz

        self.env_cfg = env_cfg

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2, gravity=(0.0, 0.0, -9.81)),
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
        base_link = self.robot.get_link("base")
        self.imu_sensor = self.scene.add_sensor(IMUOptions(entity_idx=self.robot.idx, link_idx_local=base_link.idx_local,read_delay=0.02*2))


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

        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.dof_pos = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

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
        self.episode_length_buf[:] = 0
        
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
        self.get_observations()

    def step(self, actions):

        self.robot.control_dofs_position(actions, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1

        self.get_observations()

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
            '/imu/data_rl',
            qos_profile=qos_profile
        )

        self.imu_raw_publisher = self.create_publisher(
            Imu,
            '/imu/data_raw',
            qos_profile=qos_profile
        )

         # Initialize Genesis and environment

        gs.init()
        env_cfg, _, _, _ = get_cfgs()

        self.env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            show_viewer=True,
        )




        self.step_count = 0
        
        # Reset environment and get initial observations
        self.env.reset()
        while self.step_count < 20:
            self.env.step(torch.zeros((1,12), device=gs.device))
            self.step_count += 1
        
        self.publish_observations()




    def joint_command_callback(self, msg):
        # Handle incoming joint commands (if needed)
        if len(msg.position) != 12:
            self.get_logger().error(f'Expected 12 joint positions, but got {len(msg.position)}')
            return
        # joint_positions = torch.zeros((1,12), device=gs.device)
        joint_positions = torch.tensor(msg.position, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
        self.step_count += 1
        self.env.step(joint_positions)
        self.publish_observations()
    
    def publish_observations(self):
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.env.env_cfg["joint_names"]
        joint_state_msg.position = self.env.dof_pos[0].cpu().numpy().tolist()  # Joint positions from environment
        joint_state_msg.velocity = self.env.dof_vel[0].cpu().numpy().tolist()  # Joint velocities from environment
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

        imu_reading = self.env.imu_sensor.read_ground_truth()
        imu_msg_raw = Imu()
        imu_msg_raw.header.stamp = self.get_clock().now().to_msg()
        imu_msg_raw.angular_velocity.x = imu_reading['ang_vel'][0].item()
        imu_msg_raw.angular_velocity.y = imu_reading['ang_vel'][1].item()
        imu_msg_raw.angular_velocity.z = imu_reading['ang_vel'][2].item()
        imu_msg_raw.linear_acceleration.x = imu_reading['lin_acc'][0].item()
        imu_msg_raw.linear_acceleration.y = imu_reading['lin_acc'][1].item()
        imu_msg_raw.linear_acceleration.z = -imu_reading['lin_acc'][2].item()
        self.imu_raw_publisher.publish(imu_msg_raw)

        if self.step_count % 50 == 0:  # Log every 50 steps
            self.get_logger().info(f'Step {self.step_count}')
            self.get_logger().info(f'IMU: {self.env.imu_sensor.read()}')

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
    