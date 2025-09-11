#!/usr/bin/env python3

from __future__ import print_function

import math
import rclpy

# from champ_msgs.msg import Pose as PoseLite
from geometry_msgs.msg import Pose as Pose
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool, Trigger

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

class Teleop(Node):
    def __init__(self):
        super().__init__('pilla_teleop')
        
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 1)
        # self.pose_lite_publisher = self.create_publisher(PoseLite, 'body_pose/raw', 1)
        self.pose_publisher = self.create_publisher(Pose, 'body_pose', 1)
        
        self.joy_subscriber = self.create_subscription(Joy, 'joy', self.joy_callback, 1)

        # Service clients for hardware interface
        self.arm_motors_client = self.create_client(SetBool, 'pilla/arm_motors')
        self.go_to_zero_pos_client = self.create_client(Trigger, 'pilla/go_to_zero_pos')
        self.engage_client = self.create_client(SetBool, 'pilla/engage')

        # Button state tracking to prevent multiple calls
        self.button_states = {
            'A': False,  # Button 0
            'X': False,  # Button 2
            'Y': False   # Button 3
        }

        # State tracking for toggle functionality
        self.arm_state = False
        self.engage_state = False

        self.declare_parameter("gait/swing_height", 0)
        self.declare_parameter("gait/nominal_height", 0)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("turn", 1.0)
        
        self.swing_height = self.get_parameter("gait/swing_height").value
        self.nominal_height = self.get_parameter("gait/nominal_height").value
        self.speed = self.get_parameter("speed").value
        self.turn = self.get_parameter("turn").value

        self.get_logger().info('Pilla Teleop initialized')
        self.get_logger().info('Button mapping: A=arm/disarm toggle, X=go_to_zero_pos, Y=engage/disengage toggle')

    def joy_callback(self, data):
        # Handle button presses for hardware interface services
        self.handle_service_buttons(data)

        twist = Twist()
        twist.linear.x = data.axes[1] * self.speed
        twist.linear.y = data.buttons[4] * data.axes[0] * self.speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = (not data.buttons[4]) * data.axes[0] * self.turn
        self.velocity_publisher.publish(twist)

        # body_pose_lite = PoseLite()
        # body_pose_lite.x = 0.0
        # body_pose_lite.y = 0.0
        # body_pose_lite.roll = (not data.buttons[5]) *-data.axes[3] * 0.349066
        # body_pose_lite.pitch = data.axes[4] * 0.174533
        # body_pose_lite.yaw = data.buttons[5] * data.axes[3] * 0.436332
        # if data.axes[5] < 0:
        #     body_pose_lite.z = data.axes[5] * 0.5

        # self.pose_lite_publisher.publish(body_pose_lite)

        # body_pose = Pose()
        # body_pose.position.z = body_pose_lite.z

        # quaternion = quaternion_from_euler(body_pose_lite.roll, body_pose_lite.pitch, body_pose_lite.yaw)
        # body_pose.orientation.x = quaternion[0]
        # body_pose.orientation.y = quaternion[1]
        # body_pose.orientation.z = quaternion[2]
        # body_pose.orientation.w = quaternion[3]

        # self.pose_publisher.publish(body_pose)

    def handle_service_buttons(self, data):
        """Handle button presses for service calls with debouncing."""
        # A button (button 0) - arm/disarm motors toggle
        if data.buttons[0] and not self.button_states['A']:
            self.button_states['A'] = True
            self.arm_state = not self.arm_state
            self.call_arm_motors_service(self.arm_state)
        elif not data.buttons[0] and self.button_states['A']:
            self.button_states['A'] = False

        # X button (button 2) - go to zero position
        if data.buttons[2] and not self.button_states['X']:
            self.button_states['X'] = True
            self.call_go_to_zero_pos_service()
        elif not data.buttons[2] and self.button_states['X']:
            self.button_states['X'] = False

        # Y button (button 3) - engage/disengage toggle
        if data.buttons[3] and not self.button_states['Y']:
            self.button_states['Y'] = True
            self.engage_state = not self.engage_state
            self.call_engage_service(self.engage_state)
        elif not data.buttons[3] and self.button_states['Y']:
            self.button_states['Y'] = False

    def call_arm_motors_service(self, arm_state):
        """Call the arm_motors service."""
        if not self.arm_motors_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('arm_motors service not available')
            return

        request = SetBool.Request()
        request.data = arm_state
        
        future = self.arm_motors_client.call_async(request)
        future.add_done_callback(
            lambda f: self.handle_arm_motors_response(f, arm_state)
        )

    def handle_arm_motors_response(self, future, arm_state):
        """Handle response from arm_motors service."""
        try:
            response = future.result()
            if response.success:
                action = "Armed" if arm_state else "Disarmed"
                self.get_logger().info(f'{action} motors: {response.message}')
            else:
                # Revert state on failure
                self.arm_state = not self.arm_state
                self.get_logger().error(f'Failed to arm/disarm motors: {response.message}')
        except Exception as e:
            # Revert state on failure
            self.arm_state = not self.arm_state
            self.get_logger().error(f'Service call failed: {e}')

    def call_go_to_zero_pos_service(self):
        """Call the go_to_zero_pos service."""
        if not self.go_to_zero_pos_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('go_to_zero_pos service not available')
            return

        request = Trigger.Request()
        
        future = self.go_to_zero_pos_client.call_async(request)
        future.add_done_callback(self.handle_go_to_zero_pos_response)

    def handle_go_to_zero_pos_response(self, future):
        """Handle response from go_to_zero_pos service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Go to zero position: {response.message}')
            else:
                self.get_logger().error(f'Failed to go to zero position: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def call_engage_service(self, engage_state):
        """Call the engage service."""
        if not self.engage_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('engage service not available')
            return

        request = SetBool.Request()
        request.data = engage_state
        
        future = self.engage_client.call_async(request)
        future.add_done_callback(
            lambda f: self.handle_engage_response(f, engage_state)
        )

    def handle_engage_response(self, future, engage_state):
        """Handle response from engage service."""
        try:
            response = future.result()
            if response.success:
                action = "Engaged" if engage_state else "Disengaged"
                self.get_logger().info(f'{action}: {response.message}')
            else:
                # Revert state on failure
                self.engage_state = not self.engage_state
                self.get_logger().error(f'Failed to engage/disengage: {response.message}')
        except Exception as e:
            # Revert state on failure
            self.engage_state = not self.engage_state
            self.get_logger().error(f'Service call failed: {e}')

def main():
    rclpy.init()
    teleop = Teleop()
    
    try:
        rclpy.spin(teleop)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()