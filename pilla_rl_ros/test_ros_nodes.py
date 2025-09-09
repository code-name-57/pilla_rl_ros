#!/usr/bin/env python3

"""
Test script to verify that the ROS nodes work together properly.
This script will publish some test commands to the genesis sim node.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class TestCommandPublisher(Node):
    def __init__(self):
        super().__init__('test_command_publisher')
        
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(2.0, self.publish_commands)  # Every 2 seconds
        self.command_sequence = [
            (0.0, 0.0, 0.0),   # Stationary
            (1.0, 0.0, 0.0),   # Forward
            (0.5, 0.0, 0.5),   # Forward + Turn
            (0.0, 0.0, 0.0),   # Stationary
            (-0.5, 0.0, 0.0),  # Backward
            (0.0, 0.5, 0.0),   # Sideways
        ]
        self.command_index = 0
        
    def publish_commands(self):
        if self.command_index < len(self.command_sequence):
            cmd = Twist()
            lin_x, lin_y, ang_z = self.command_sequence[self.command_index]
            cmd.linear.x = lin_x
            cmd.linear.y = lin_y
            cmd.angular.z = ang_z
            
            self.cmd_vel_publisher.publish(cmd)
            self.get_logger().info(f'Published command {self.command_index + 1}: linear_x={lin_x}, linear_y={lin_y}, angular_z={ang_z}')
            
            self.command_index += 1
        else:
            self.get_logger().info('All test commands published. Repeating...')
            self.command_index = 0


def main():
    rclpy.init()
    test_publisher = TestCommandPublisher()
    
    print("=== ROS Nodes Test Publisher ===")
    print("This will publish test velocity commands to /cmd_vel")
    print("Make sure to run:")
    print("1. python3 genesis_sim_node.py")
    print("2. python3 rl_policy_node.py") 
    print("3. Then run this script")
    print("Press Ctrl+C to stop\n")
    
    try:
        rclpy.spin(test_publisher)
    except KeyboardInterrupt:
        print("\nTest publisher stopped")
    finally:
        test_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
