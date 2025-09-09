#!/usr/bin/env python3

"""
Simple script to test the JIT policy with Go2 robot in Genesis simulator.
This script loads a trained policy and runs it on the simulated robot.
"""

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def get_env_config():
    """Get environment configuration for Go2 robot."""
    return {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0, "kd": 0.5,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_scale": 0.25,
        "clip_actions": 100.0,
    }


def get_obs_config():
    """Get observation configuration."""
    return {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25, 
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }


class Go2Simulator:
    """Simplified Go2 robot simulator for policy testing."""
    
    def __init__(self, show_viewer=True):
        self.device = gs.device
        self.dt = 0.02  # 50Hz control frequency
        
        # Get configurations
        self.env_cfg = get_env_config()
        self.obs_cfg = get_obs_config()
        self.obs_scales = self.obs_cfg["obs_scales"]
        
        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        
        # Add ground plane and robot
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # Build scene
        self.scene.build(n_envs=1)
        
        # Setup motor control
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * 12, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * 12, self.motors_dof_idx)
        
        # Initialize state variables
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device, dtype=gs.tc_float,
        )
        self.commands = torch.zeros(3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self.actions = torch.zeros(12, device=self.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        
        # Scale factors for commands in observations
        self.commands_scale = torch.tensor([
            self.obs_scales["lin_vel"], 
            self.obs_scales["lin_vel"], 
            self.obs_scales["ang_vel"]
        ], device=self.device)
        
    def reset(self):
        """Reset robot to initial position."""
        # Reset robot pose
        self.robot.set_pos(self.base_init_pos.unsqueeze(0), zero_velocity=True, 
                          envs_idx=torch.tensor([0], device=self.device))
        self.robot.set_quat(self.base_init_quat.unsqueeze(0), zero_velocity=True,
                           envs_idx=torch.tensor([0], device=self.device))
        
        # Reset joint positions
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0),
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=torch.tensor([0], device=self.device),
        )
        
        # Reset state
        self.actions[:] = 0.0
        self.commands[:] = 0.0
        
        # Step once to settle
        self.scene.step()
        return self.get_observation()
    
    def step(self, actions):
        """Execute one simulation step with given actions."""
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()
        return self.get_observation()
    
    def get_observation(self):
        """Get current robot observation."""
        # Get robot state
        base_quat = self.robot.get_quat()[0]
        inv_base_quat = inv_quat(base_quat)
        
        # Transform velocities to robot frame
        base_ang_vel = transform_by_quat(self.robot.get_ang()[0], inv_base_quat)
        projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        
        # Get joint states
        dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)[0]
        dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)[0]
        
        # Build observation vector
        obs = torch.cat([
            base_ang_vel * self.obs_scales["ang_vel"],              # 3
            projected_gravity,                                       # 3
            self.commands * self.commands_scale,                    # 3
            (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            dof_vel * self.obs_scales["dof_vel"],                   # 12
            self.actions,                                           # 12
        ])
        
        return obs
    
    def set_commands(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """Set velocity commands for the robot."""
        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y  
        self.commands[2] = ang_vel_z


def load_policy(policy_path, device):
    """Load JIT policy from file."""
    try:
        policy = torch.jit.load(policy_path, map_location=device)
        policy = policy.to(device)
        print(f"✓ Policy loaded successfully from {policy_path} on device: {device}")
        
        # Test policy with dummy input
        dummy_obs = torch.zeros(1, 45, device=device)
        with torch.no_grad():
            dummy_action = policy(dummy_obs)
        print(f"✓ Policy test passed - input: {dummy_obs.shape}, output: {dummy_action.shape}")
        return policy
        
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return None


def main():
    """Main function to run policy checker."""
    print("=== Go2 Policy Checker with Genesis Simulator ===\n")
    
    # Initialize Genesis
    gs.init()
    
    # Create simulator
    print("Initializing simulator...")
    sim = Go2Simulator(show_viewer=True)
    
    # Load policy
    policy_path = "policy.jit"  # Adjust path as needed
    policy = load_policy(policy_path, sim.device)
    if policy is None:
        print("Cannot proceed without policy. Exiting.")
        return
    
    # Reset environment
    print("Resetting robot...")
    obs = sim.reset()
    print(f"✓ Initial observation shape: {obs.shape}")
    
    # Run simulation
    print("\nStarting policy evaluation...")
    print("Commands: Forward=1.0, Stationary=0.0")
    print("Press Ctrl+C to stop\n")
    
    step_count = 0
    try:
        while True:
            # Set commands (you can modify these)
            if step_count < 200:  # First 4 seconds: stationary
                sim.set_commands(0.0, 0.0, 0.0)
            elif step_count < 500:  # Next 6 seconds: forward
                sim.set_commands(1.0, 0.0, 0.0) 
            elif step_count < 650:  # Next 3 seconds: turn
                sim.set_commands(0.5, 0.0, 0.5)
            else:  # Rest: stationary
                sim.set_commands(0.0, 0.0, 0.0)
            
            # Get policy action
            with torch.no_grad():
                action = policy(obs.unsqueeze(0)).squeeze(0)
            
            # Step simulation
            obs = sim.step(action)
            step_count += 1
            
            # Log progress
            if step_count % 50 == 0:
                print(f"Step {step_count}: Action norm = {torch.norm(action):.3f}")
                
    except KeyboardInterrupt:
        print(f"\nStopped after {step_count} steps")
    
    print("Policy evaluation completed!")


if __name__ == "__main__":
    main()
