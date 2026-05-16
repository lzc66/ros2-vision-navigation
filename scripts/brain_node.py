#!/usr/bin/env python3
"""
Embodied AI Brain: Continuous Logistics Loop.

States: EXPLORE -> LOCK -> GRAB -> RETURN -> DROP -> EXPLORE ...

Stage 4 Fixes:
  1. Approach Margin: safe_dist = max(0.2, dist - APPROACH_MARGIN) to avoid lethal cost
  2. Goal Filtering: only re-send Nav2 goal if target drifts >0.2m
  3. GRAB (3s load) + DROP (3s unload) states with state cache reset
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Twist
import math, random, time

SPAWN_X, SPAWN_Y = -2.0, -0.5

PATROL_POINTS = [
    (1.5, -0.5), (1.0, 0.5), (1.5, 0.5), (1.5, 1.5),
    (0.0, 1.8), (2.0, 0.0), (-1.5, 1.5), (0.5, -1.8)
]

# Camera (Waffle R200)
CAM_FOV = 1.089
CAM_W = 1920.0
CAM_CX = CAM_W / 2.0
K_DIST = 600.0

# Termination
CAPTURE_AREA = 350000
CENTER_THRESHOLD = 80.0

# Stage 4: Approach Margin (prevent lethal-cost goal rejection)
APPROACH_MARGIN = 0.45  # meters: stop this far in front of the box

# Stage 4: Goal Filtering
GOAL_FILTER_EPS = 0.2   # meters: min drift to re-send Nav2 goal
GOAL_UPDATE_PERIOD = 2.0 # seconds between projection updates

# Timing
GRAB_DURATION = 3.0   # seconds: loading cargo
DROP_DURATION = 3.0   # seconds: unloading cargo
VISION_TIMEOUT = 5.0  # seconds: lost vision -> back to explore


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.state = 'EXPLORE'
        self.round = 0  # logistics round counter

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None

        # Subscribers
        self.create_subscription(Point, '/red_object', self.vision_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        # Robot pose
        self._rx = 0.0; self._ry = 0.0; self._ryaw = 0.0

        # Red object
        self._red_error_x = 0.0; self._red_area = 0.0
        self._red_seen = False; self._last_vision_t = 0.0

        # LOCK dynamic goals
        self._lock_timer = None

        # Stage 4: Goal filter cache
        self._last_sent_tx = None
        self._last_sent_ty = None

        # CmdVel (emergency stop only)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Brain ready. State: EXPLORE (Continuous Logistics)')
        self._start_explore()

    # ============ Callbacks ============
    def amcl_cb(self, msg):
        self._rx = msg.pose.pose.position.x
        self._ry = msg.pose.pose.position.y
        self._ryaw = quat_to_yaw(msg.pose.pose.orientation)

    def vision_cb(self, msg):
        self._red_error_x = msg.x
        self._red_area = msg.y
        self._red_seen = True
        self._last_vision_t = time.time()

        if self.state == 'EXPLORE' and self._red_area > 2000:
            self._enter_lock()

    # ============ Camera Projection (Stage 4: Approach Margin) ============
    def _project_target(self):
        err = self._red_error_x
        area = self._red_area
        theta = -(err / CAM_CX) * (CAM_FOV / 2.0)

        if area < 100:
            raw_dist = 8.0
        else:
            raw_dist = K_DIST / math.sqrt(area)

        # Stage 4 Fix 1: safe approach distance (stop 0.45m in front of box)
        safe_dist = max(0.2, raw_dist - APPROACH_MARGIN)

        tx = self._rx + safe_dist * math.cos(self._ryaw + theta)
        ty = self._ry + safe_dist * math.sin(self._ryaw + theta)
        return tx, ty, raw_dist, safe_dist, theta

    # ============ EXPLORE ============
    def _start_explore(self):
        # Reset red-object state cache
        self._red_seen = False
        self._red_error_x = 0.0
        self._red_area = 0.0
        self._last_sent_tx = None
        self._last_sent_ty = None
        self._lock_timer = None

        self.state = 'EXPLORE'
        goal = random.choice(PATROL_POINTS)
        self.round += 1
        self.get_logger().info(f'[EXPLORE #{self.round}] Patrol -> ({goal[0]:.1f}, {goal[1]:.1f})')
        self._send_nav_goal(goal[0], goal[1])

    # ============ LOCK ============
    def _enter_lock(self):
        if self.state == 'LOCK':
            return
        self.state = 'LOCK'
        self.get_logger().info('[LOCK] Red object detected! Starting dynamic semantic tracking...')

        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

        # Reset goal filter cache for new target
        self._last_sent_tx = None
        self._last_sent_ty = None

        self._lock_timer = self.create_timer(GOAL_UPDATE_PERIOD, self._lock_update)

    def _lock_update(self):
        if self.state != 'LOCK':
            return

        # Vision timeout
        if time.time() - self._last_vision_t > VISION_TIMEOUT:
            self.get_logger().info('[LOCK] Vision lost, back to EXPLORE')
            self._lock_timer.cancel()
            self._start_explore()
            return

        err = self._red_error_x
        area = self._red_area

        # Termination check
        if area > CAPTURE_AREA and abs(err) < CENTER_THRESHOLD:
            self.get_logger().info(f'[TARGET ACQUIRED] Area={area:.0f} err={err:.0f}px')
            self._lock_timer.cancel()
            self._enter_grab()
            return

        # Project target
        tx, ty, raw_dist, safe_dist, theta = self._project_target()

        # Stage 4 Fix 2: Goal filtering — only send if drifted >0.2m
        if self._last_sent_tx is not None:
            drift = math.sqrt((tx - self._last_sent_tx)**2 + (ty - self._last_sent_ty)**2)
            if drift < GOAL_FILTER_EPS:
                return  # target stable, skip re-send

        self._last_sent_tx = tx
        self._last_sent_ty = ty

        self.get_logger().info(
            f'[LOCK] err={err:.0f}px area={area:.0f} raw_dist={raw_dist:.2f}m '
            f'safe_dist={safe_dist:.2f}m theta={theta:.3f}rad -> ({tx:.2f}, {ty:.2f})'
        )
        self._send_nav_goal(tx, ty)

    # ============ GRAB (Stage 4 Fix 3: loading pause) ============
    def _enter_grab(self):
        self.state = 'GRAB'
        self.get_logger().info('[GRAB] Loading cargo... (3s)')
        self._stop_robot()
        self.create_timer(GRAB_DURATION, self._grab_done, one_shot=True)

    def _grab_done(self):
        self.get_logger().info('[GRAB] Cargo loaded! Returning to base.')
        self._start_return()

    # ============ RETURN ============
    def _start_return(self):
        self.state = 'RETURN'
        self.get_logger().info(f'[RETURN] Navigating to spawn ({SPAWN_X:.1f}, {SPAWN_Y:.1f})')
        self._send_nav_goal(SPAWN_X, SPAWN_Y)

    # ============ DROP (Stage 4 Fix 3: unloading pause) ============
    def _enter_drop(self):
        self.state = 'DROP'
        self.get_logger().info('[DROP] Unloading cargo... (3s)')
        self._stop_robot()
        self.create_timer(DROP_DURATION, self._drop_done, one_shot=True)

    def _drop_done(self):
        self.get_logger().info('[DROP] Cargo unloaded. Starting next round!')
        self._start_explore()

    # ============ Helpers ============
    def _stop_robot(self):
        stop = Twist()
        stop.linear.x = 0.0; stop.angular.z = 0.0
        self.cmd_pub.publish(stop)

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
        if status != 4:
            return  # 4 = SUCCEEDED; ignore aborts/cancels

        if self.state == 'EXPLORE':
            self.get_logger().info('[EXPLORE] Waypoint reached, next...')
            self._start_explore()
        elif self.state == 'RETURN':
            self.get_logger().info('[RETURN] Arrived at spawn.')
            self._enter_drop()
        # In LOCK, the timer handles goal completion


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
