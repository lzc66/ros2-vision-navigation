#!/usr/bin/env python3
"""
Embodied AI Brain: Dynamic Semantic Waypoint Navigation.

States:
  EXPLORE -> random Nav2 patrol waypoints
  LOCK    -> compute red-box MAP coords via camera projection,
             send dynamic Nav2 goals every 2s (Nav2 handles obstacle avoidance)
  RETURN  -> navigate back to spawn point
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Twist
import math
import random
import time

SPAWN_X, SPAWN_Y = -2.0, -0.5

PATROL_POINTS = [
    (1.5, -0.5), (1.0, 0.5), (1.5, 0.5), (1.5, 1.5),
    (0.0, 1.8), (2.0, 0.0), (-1.5, 1.5), (0.5, -1.8)
]

# Camera intrinsic params (Waffle Intel RealSense R200)
CAM_FOV = 1.089       # horizontal FOV (radians)
CAM_W = 1920.0        # image width (pixels)
CAM_CX = CAM_W / 2.0  # 960.0

# Distance estimation: K_DIST / sqrt(area)
K_DIST = 600.0        # tunable calibration constant

# LOCK termination
CAPTURE_AREA = 350000 # box occupies ~17% of image -> very close
CENTER_THRESHOLD = 80.0  # pixels: object must be near image center

# LOCK dynamic goal update rate
GOAL_UPDATE_INTERVAL = 2.0  # seconds

# Cooldown after RETURN
RETURN_COOLDOWN = 10.0  # seconds


def quat_to_yaw(q):
    """Convert quaternion (x,y,z,w) to yaw angle."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.state = 'EXPLORE'

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None

        # Vision subscriber
        self.create_subscription(Point, '/red_object', self.vision_cb, 10)

        # AMCL pose subscriber
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        # Robot pose (updated from AMCL)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0

        # Red object state
        self._red_error_x = 0.0
        self._red_area = 0.0
        self._red_seen = False
        self._last_vision_time = 0.0

        # LOCK state
        self._lock_timer = None
        self._return_cooldown_until = 0.0

        # Velocity for emergency stop
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Brain ready. State: EXPLORE (Dynamic Semantic Nav)')
        self._start_explore()

    # === AMCL Callback ===
    def amcl_cb(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        self._robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

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

    # === Camera to Map Projection ===
    def _project_target(self):
        """Project red box from camera coords to MAP coordinates."""
        error_x = self._red_error_x
        area = self._red_area

        # Optical angle: negative error_x = object to the LEFT of center
        theta = -(error_x / CAM_CX) * (CAM_FOV / 2.0)

        # Distance estimate from area
        if area < 100:
            dist = 8.0  # far away, fallback
        else:
            dist = K_DIST / math.sqrt(area)

        # Project to map coordinates
        tx = self._robot_x + dist * math.cos(self._robot_yaw + theta)
        ty = self._robot_y + dist * math.sin(self._robot_yaw + theta)
        return tx, ty, dist, theta

    # === State: EXPLORE ===
    def _start_explore(self):
        self.state = 'EXPLORE'
        goal = random.choice(PATROL_POINTS)
        self.get_logger().info(f'[EXPLORE] Patrol -> ({goal[0]:.1f}, {goal[1]:.1f})')
        self._send_nav_goal(goal[0], goal[1])

    # === State: LOCK (Dynamic Semantic Waypoint) ===
    def _enter_lock(self):
        if self.state == 'LOCK':
            return
        self.state = 'LOCK'
        self.get_logger().info('[LOCK] Red object detected! Starting dynamic semantic tracking...')

        # Stop any current Nav2 goal (will be replaced by our dynamic goal)
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

        # Start dynamic goal update timer
        self._lock_timer = self.create_timer(GOAL_UPDATE_INTERVAL, self._lock_update)

    def _lock_update(self):
        """Periodic: project target -> send Nav2 goal with obstacle avoidance."""
        if self.state != 'LOCK':
            return

        # Timeout: vision lost too long
        if time.time() - self._last_vision_time > 5.0:
            self.get_logger().info('[LOCK] Vision lost, returning to EXPLORE')
            self._lock_timer.cancel()
            self._lock_timer = None
            self._start_explore()
            return

        error_x = self._red_error_x
        area = self._red_area

        # Check termination: close enough AND centered
        if area > CAPTURE_AREA and abs(error_x) < CENTER_THRESHOLD:
            self.get_logger().info(f'[TARGET ACQUIRED] Area={area:.0f} err={error_x:.0f}px')
            self._lock_timer.cancel()
            self._lock_timer = None
            self._start_return()
            return

        # Project target to map
        tx, ty, dist, theta = self._project_target()
        self.get_logger().info(
            f'[LOCK] err={error_x:.0f}px area={area:.0f} '
            f'dist={dist:.2f}m theta={theta:.3f}rad -> target_map=({tx:.2f}, {ty:.2f})'
        )

        # Send dynamic Nav2 goal (planner handles obstacle avoidance!)
        self._send_nav_goal(tx, ty)

    # === State: RETURN ===
    def _start_return(self):
        self.state = 'RETURN'
        self.get_logger().info(f'[RETURN] Navigating to spawn ({SPAWN_X:.1f}, {SPAWN_Y:.1f})')
        self._send_nav_goal(SPAWN_X, SPAWN_Y)

    # === Shared Nav2 Goal Sender ===
    def _send_nav_goal(self, gx, gy):
        self.nav_client.wait_for_server()
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose.header.frame_id = 'map'
        nav_goal.pose.pose.position.x = gx
        nav_goal.pose.pose.position.y = gy
        nav_goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(nav_goal)
        future.add_done_callback(self._nav_response_cb)

    def _nav_response_cb(self, future):
        self._goal_handle = future.result()
        if self._goal_handle is None:
            self.get_logger().error('Nav2 goal rejected')
            return
        self._goal_handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        result = future.result()
        status = result.status if result else -1
        if status != 4:  # 4 = SUCCEEDED
            if self.state in ('EXPLORE', 'LOCK'):
                pass  # will retry
            return
        if self.state == 'EXPLORE':
            self.get_logger().info('[EXPLORE] Waypoint reached, next...')
            self._start_explore()
        elif self.state == 'RETURN':
            self.get_logger().info('[RETURN] Arrived at spawn! Mission complete.')
            self._return_cooldown_until = time.time() + RETURN_COOLDOWN
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
