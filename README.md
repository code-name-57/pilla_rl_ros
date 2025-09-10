# Pilla RL ROS

A ROS 2 package for reinforcement learning-based robotic control, supporting quadruped (Go2) and humanoid (H1) robots with physics simulation and real robot deployment.

## Overview

This package implements a complete RL-based control system that bridges reinforcement learning policies with ROS 2 for robotic control. The system uses physics simulation for development and testing, then deploys to real robots for locomotion control.

### System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Joystick   │───▶│ Teleop Node │───▶│ RL Policy   │───▶│ Genesis Sim │
│   Input     │    │             │    │    Node     │    │    Node     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │                   │
                          │                   ▼                   ▼
                          │           ┌─────────────┐    ┌─────────────┐
                          │           │ Joint Cmds  │    │Robot State  │
                          │           │ Publisher   │    │ Feedback    │
                          │           └─────────────┘    └─────────────┘
                          │                   │                   │
                          ▼                   ▼                   ▼
                  ┌─────────────────────────────────────────────────────┐
                  │              Real Robot / Hardware              │
                  └─────────────────────────────────────────────────────┘
```

## Nodes Description

### 1. Teleop Node (`teleop_node.py`)
**Purpose**: Manual robot control via joystick input

**Subscriptions**:
- `/joy` (sensor_msgs/Joy) - Joystick input for manual control

**Publications**:
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands (linear and angular)
- `/body_pose` (geometry_msgs/Pose) - Body pose commands

**Service Calls**:
- `pilla/arm_motors` (std_srvs/SetBool) - Enable/disable motor control
- `pilla/go_to_zero_pos` (std_srvs/Trigger) - Move robot to zero position
- `pilla/engage` (std_srvs/SetBool) - Engage/disengage robot control

**Functionality**:
- Converts joystick input to robot movement commands
- Button mapping: A=arm/disarm motors, X=go to zero position, Y=engage/disengage
- Supports forward/backward, strafing, and turning motions
- Provides hardware interface for motor control and safety features

### 2. RL Policy Node (`rl_policy_node.py`)
**Purpose**: Neural network policy inference for generating robot actions

**Subscriptions**:
- `/cmd_vel` (geometry_msgs/Twist) - High-level velocity commands
- `/robot_observations` (std_msgs/Float32MultiArray) - Robot state observations (45 dimensions)

**Publications**:
- `/robot_actions` (std_msgs/Float32MultiArray) - Policy actions (12 dimensions for Go2)
- `/joint_commands` (sensor_msgs/JointState) - Joint position commands

**Functionality**:
- Loads pre-trained PyTorch JIT policy (`policy.jit`)
- Combines velocity commands with robot observations
- Runs neural network inference to generate joint actions
- Publishes both raw actions and formatted joint commands
- Runs at 50Hz control frequency

**Observation Vector (45 dimensions)**:
- Base angular velocity (3) 
- Gravity direction (3)
- Velocity commands (3)
- Joint positions relative to default (12)
- Joint velocities (12) 
- Previous actions (12)

### 3. Genesis Sim Node (`genesis_sim_node.py`)
**Purpose**: Physics simulation of Go2 robot using Genesis simulator

**Subscriptions**:
- `/robot_actions` (std_msgs/Float32MultiArray) - Actions from RL policy

**Publications**:
- `/robot_observations` (std_msgs/Float32MultiArray) - Robot state observations

**Functionality**:
- Simulates Go2 quadruped robot with Genesis physics engine
- Processes incoming actions and applies them to simulated robot
- Computes and publishes robot observations (joint states, IMU, base velocity)
- Provides visual simulation environment for development and testing
- Runs PD controllers on robot joints with kp=20.0, kd=0.5

**Robot Configuration**:
- 12 actuated joints (3 per leg: hip, thigh, calf)
- Control frequency: 50Hz (dt=0.02s)
- Joint order: FR, FL, RR, RL (Front-Right, Front-Left, Rear-Right, Rear-Left)

### 4. Genesis Sim Refactored Node (`genesis_sim_refactored_node.py`) 
**Purpose**: Alternative simulation node with joint command interface

**Subscriptions**:
- `/joint_commands` (sensor_msgs/JointState) - Direct joint position commands

**Publications**:
- `/robot_observations` (std_msgs/Float32MultiArray) - Robot state observations

**Functionality**:
- Similar to genesis_sim_node but accepts direct joint commands
- Useful for bypassing RL policy and testing direct joint control
- Provides the same simulation environment with different input interface

### 5. H1 Fullbody Node (`h1fullbodyNode.py`)
**Purpose**: Control node for H1 humanoid robot

**Subscriptions**:
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
- `/imu` (sensor_msgs/Imu) - IMU sensor data (synchronized)
- `/joint_states` (sensor_msgs/JointState) - Current joint states (synchronized)

**Publications**:
- `/joint_command` (sensor_msgs/JointState) - Joint position commands

**Functionality**:
- Specialized controller for H1 humanoid robot (19 DOF)
- Uses time-synchronized IMU and joint state data
- Implements fullbody control with balance and locomotion
- Runs policy at reduced frequency (decimation=4) for computational efficiency

**Observation Vector (69 dimensions)**:
- Base linear velocity (3)
- Base angular velocity (3)  
- Gravity direction (3)
- Velocity commands (3)
- Joint positions relative to default (19)
- Joint velocities (19)
- Previous actions (19)

## Supporting Files

### Launch File (`launch/pilla_rl.launch.py`)
Launches the complete system with all necessary nodes:
- Joy node for joystick input
- Teleop node for manual control
- RL policy node for neural network inference  
- Genesis sim node for physics simulation

### Test Scripts

**`policy_checker_sim.py`**: Standalone policy testing script
- Tests neural network policies without ROS
- Useful for debugging and policy validation
- Runs predefined command sequences

**`test_ros_nodes.py`**: ROS system integration testing
- Publishes test velocity commands to verify node communication
- Helps debug topic connections and data flow

## Topic Flow and Data Relationships

### Main Control Loop:
1. **Joystick Input** → `joy` topic → **Teleop Node**
2. **Teleop Node** → `cmd_vel` topic → **RL Policy Node**  
3. **Genesis Sim Node** → `robot_observations` topic → **RL Policy Node**
4. **RL Policy Node** → `robot_actions` topic → **Genesis Sim Node**
5. **RL Policy Node** → `joint_commands` topic → **Real Robot/Hardware**

### Alternative Paths:
- **Direct Joint Control**: `joint_commands` → **Genesis Sim Refactored Node**
- **H1 Robot**: `cmd_vel` + `imu` + `joint_states` → **H1 Fullbody Node** → `joint_command`

## Usage

### Launch Complete System
```bash
ros2 launch pilla_rl_ros pilla_rl.launch.py
```

### Launch Individual Nodes
```bash
# Terminal 1: Simulation
ros2 run pilla_rl_ros genesis_sim_node

# Terminal 2: RL Policy  
ros2 run pilla_rl_ros rl_policy_node

# Terminal 3: Teleop (with joystick connected)
ros2 run pilla_rl_ros teleop_node

# Terminal 4: Joy node
ros2 run joy joy_node
```

### Test the System
```bash
# Test policy without ROS
python3 src/pilla_rl_ros/pilla_rl_ros/policy_checker_sim.py

# Test ROS node communication
python3 src/pilla_rl_ros/pilla_rl_ros/test_ros_nodes.py
```

## Dependencies

### Core Requirements:
- ROS 2 (Humble or newer)
- Python 3.8+
- PyTorch (with CUDA support recommended)
- Genesis Physics Simulator
- rsl-rl-lib==2.3.3

### ROS 2 Dependencies:
- `rclpy` - ROS 2 Python client library
- `std_msgs` - Standard message types
- `geometry_msgs` - Geometry message types
- `sensor_msgs` - Sensor message types  
- `message_filters` - Message synchronization
- `joy` - Joystick driver package

### Hardware Requirements:
- NVIDIA GPU (recommended for PyTorch inference)
- Compatible joystick/gamepad for manual control
- Robot hardware (Go2 or H1) for real deployment

## Robot Support

### Go2 Quadruped Robot:
- 12 DOF (3 joints per leg)
- Uses `genesis_sim_node` and `rl_policy_node`
- Policy trained for locomotion tasks
- Supports walking, turning, and basic maneuvers

### H1 Humanoid Robot:
- 19 DOF fullbody control
- Uses `h1fullbodyNode`
- Requires IMU and joint state feedback
- Supports balance and walking behaviors

## Development Notes

- The system is designed for both simulation and real robot deployment
- Neural network policies are trained offline and loaded as JIT models
- Genesis simulator provides accurate physics for policy development
- All nodes use QoS profiles optimized for real-time control
- Hardware interfaces include safety features (arm/disarm, emergency stop)