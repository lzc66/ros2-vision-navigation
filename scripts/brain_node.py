#!/usr/bin/env python3
"""
Embodied AI Brain: State-machine driven patrol + visual servo + return.

States:
  EXPLORE   -> send random Nav2 goals for patrol
  LOCK      -> cancel nav, visual-servo toward red object via /cmd_vel
  RETURN    -> navigate back to spawn point
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Point, Twist
import math
import random
import time

# Spawn point (map==world coords)
SPAWN_X, SPAWN_Y = -2.0, -0.5

# Patrol waypoints (safe, explored in Stage 2)
PATROL_POINTS = [
    (1.5, -0.5), (1.0, 0.5), (1.5, 0.5), (1.5, 1.5),
    (0.0, 1.8), (2.0, 0.0), (-1.5, 1.5), (0.5, -1.8)
]

# Vision servo params
KP = 0.0008         # P-controller gain (angular) - slower for 1920px
FORWARD_SPEED = 0.08  # very slow approach
CAPTURE_AREA = 600000  # area threshold: object <0.5m (waffle 1920x1080 cam)
ERR_THRESHOLD = 20.0   # pixels, dead zone


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.state = 'EXPLORE'

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None

        # Command velocity publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Vision subscriber
        self.create_subscription(Point, '/red_object', self.vision_cb, 10)

        # State
        self._red_error_x = 0.0
        self._red_area = 0.0
        self._red_seen = False
        self._last_vision_time = 0.0
        self._nav_active = False
        self._nav_done = False
        self._return_cooldown_until = 0.0

        self.get_logger().info('Brain Node ready. State: EXPLORE')
        self._start_explore()

    # === Vision Callback ===
    def vision_cb(self, msg):
        self._red_error_x = msg.x
        self._red_area = msg.y
        self._red_seen = True
        self._last_vision_time = time.time()

        if self.state == 'EXPLORE' and self._red_area > 2000:
            if time.time() < self._return_cooldown_until:
                return
            self._enter_lock()

    # === State: EXPLORE ===
    def _start_explore(self):
        self.state = 'EXPLORE'
        self._nav_active = True
        goal = random.choice(PATROL_POINTS)
        self.get_logger().info(f'[EXPLORE] Patrolling to ({goal[0]:.1f}, {goal[1]:.1f})')

        self.nav_client.wait_for_server()
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose.header.frame_id = 'map'
        nav_goal.pose.pose.position.x = goal[0]
        nav_goal.pose.pose.position.y = goal[1]
        nav_goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(nav_goal)
        future.add_done_callback(self._nav_response_cb)

    def _nav_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle is None:
            self.get_logger().error('[EXPLORE] Goal rejected')
            self._nav_active = False
            self._start_return()
            return
        self._goal_handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        self._nav_active = False
        if self.state == 'EXPLORE':
            self.get_logger().info('[EXPLORE] Waypoint reached, picking next...')
            self._start_explore()

    # === State: LOCK & SERVO ===
    def _enter_lock(self):
        if self.state == 'LOCK':
            return
        self.state = 'LOCK'
        self.get_logger().info('[LOCK] Red object detected! Cancelling navigation...')

        # Cancel current navigation
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._nav_active = False
            self._goal_handle = None

        # Start servo loop via timer
        self._servo_timer = self.create_timer(0.1, self._servo_loop)

    def _servo_loop(self):
        if self.state != 'LOCK':
            return

        # Timeout: if vision lost for > 5s, return to explore
        if time.time() - self._last_vision_time > 5.0:
            self.get_logger().info('[LOCK] Vision lost, returning to explore')
            self._servo_timer.cancel()
            self._start_explore()
            return

        area = self._red_area
        err = self._red_error_x

        twist = Twist()

        # Check if target is close enough
        if area > CAPTURE_AREA:
            self.get_logger().info(f'[TARGET ACQUIRED] Area={area:.0f} — object captured!')
            self._servo_timer.cancel()
            self._start_return()
            return

        # P-controller: steer toward center
        if abs(err) > ERR_THRESHOLD:
            twist.angular.z = -KP * err
        else:
            twist.angular.z = 0.0

        # Low-speed forward approach
        twist.linear.x = FORWARD_SPEED
        self.cmd_pub.publish(twist)

    # === State: RETURN ===
    def _start_return(self):
        self.state = 'RETURN'
        self.get_logger().info(f'[RETURN] Navigating back to spawn ({SPAWN_X:.1f}, {SPAWN_Y:.1f})')

        self.nav_client.wait_for_server()
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose.header.frame_id = 'map'
        nav_goal.pose.pose.position.x = SPAWN_X
        nav_goal.pose.pose.position.y = SPAWN_Y
        nav_goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(nav_goal)
        future.add_done_callback(self._return_response_cb)

    def _return_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle is None:
            self.get_logger().error('[RETURN] Goal rejected')
            return
        self._goal_handle.get_result_async().add_done_callback(self._return_result_cb)

    def _return_result_cb(self, future):
        self.get_logger().info('[RETURN] Arrived at spawn! Mission complete. Restarting EXPLORE.')
        self._return_cooldown_until = time.time() + 10.0  # 10s cooldown
        self._start_explore()


def main():
    rclpy.init()
    node = BrainNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
