#!/usr/bin/env python3
"""
Embodied AI Brain: Multi-color sort + self-healing + continuous logistics.

States: EXPLORE -> LOCK -> GRAB -> RETURN -> DROP -> EXPLORE ...

Stage 5:
  - Relative capture ratio (CAPTURE_RATIO = 0.15)
  - Dual-color semantic routing: red -> RED_ZONE, blue -> BLUE_ZONE
  - Self-healing nav: EXPLORE failure -> next WP; RETURN failure -> retry 3s
  - /target_object with z=1.0 (red) or z=2.0 (blue)
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Twist
import math, random, time

SPAWN_X, SPAWN_Y = -2.0, -0.5
RED_DROP_ZONE = (-2.0, -0.5)
BLUE_DROP_ZONE = (2.0, 2.0)

def _type_color(z):
    return 'RED' if z < 1.5 else 'BLUE/BRN'

PATROL_POINTS = [
    (1.5, -0.5), (1.0, 0.5), (1.5, 0.5), (1.5, 1.5),
    (0.0, 1.8), (2.0, 0.0), (-1.5, 1.5), (0.5, -1.8)
]

# Camera (Waffle R200)
CAM_FOV = 1.089
K_DIST = 600.0

# Stage 5: relative capture
CAPTURE_RATIO = 0.12  # 12% of image = close enough
CENTER_THRESHOLD = 0.08  # normalized error_x deadzone

APPROACH_MARGIN = 0.45
GOAL_FILTER_EPS = 0.2
GOAL_UPDATE_PERIOD = 2.0
GRAB_DURATION = 3.0
DROP_DURATION = 3.0
VISION_TIMEOUT = 5.0


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.state = 'EXPLORE'
        self.round = 0

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None

        # Subscribers
        self.create_subscription(Point, '/target_object', self.vision_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        # Robot pose
        self._rx = 0.0; self._ry = 0.0; self._ryaw = 0.0

        # Target object (Stage 5: type-aware)
        self._target_err_norm = 0.0   # normalized [-1, 1]
        self._target_ratio = 0.0      # 0..1
        self._target_type = 0.0       # 1.0=red, 2.0=blue
        self._target_seen = False
        self._last_vision_t = 0.0

        # LOCK state
        self._lock_timer = None
        self._last_tx = None; self._last_ty = None

        # CmdVel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Brain ready. EXPLORE (Multi-Color Sort + Self-Healing)')
        self._start_explore()

    # ============ Callbacks ============
    def amcl_cb(self, msg):
        self._rx = msg.pose.pose.position.x
        self._ry = msg.pose.pose.position.y
        self._ryaw = quat_to_yaw(msg.pose.pose.orientation)

    def vision_cb(self, msg):
        self._target_err_norm = msg.x    # normalized [-1, 1]
        self._target_ratio = msg.y       # 0..1
        self._target_type = msg.z        # 1.0=red, 2.0=blue
        self._target_seen = True
        self._last_vision_t = time.time()

        if self.state == 'EXPLORE' and self._target_ratio > 0.002:
            self._enter_lock()

    # ============ Camera Projection ============
    def _project_target(self):
        err = self._target_err_norm  # normalized
        ratio = self._target_ratio

        # Convert normalized error back to pixel-equivalent for projection
        theta = -(err) * (CAM_FOV / 2.0)

        if ratio < 0.001:
            raw_dist = 8.0
        else:
            # K_DIST calibrated on pixel area; normalize for ratio
            raw_dist = K_DIST / math.sqrt(ratio * 1920 * 1080)

        safe_dist = max(0.2, raw_dist - APPROACH_MARGIN)
        tx = self._rx + safe_dist * math.cos(self._ryaw + theta)
        ty = self._ry + safe_dist * math.sin(self._ryaw + theta)
        return tx, ty, raw_dist, safe_dist, theta

    # ============ EXPLORE ============
    def _start_explore(self):
        self._target_seen = False
        self._target_err_norm = 0.0
        self._target_ratio = 0.0
        self._target_type = 0.0
        self._last_tx = None; self._last_ty = None
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
        tcolor = 'RED' if self._target_type < 1.5 else 'BLUE'
        self.get_logger().info(f'[LOCK] {tcolor} target detected! Tracking...')

        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self._last_tx = None; self._last_ty = None
        self._lock_timer = self.create_timer(GOAL_UPDATE_PERIOD, self._lock_update)

    def _lock_update(self):
        if self.state != 'LOCK':
            return
        if time.time() - self._last_vision_t > VISION_TIMEOUT:
            self.get_logger().info('[LOCK] Vision lost, back to EXPLORE')
            self._lock_timer.cancel()
            self._start_explore()
            return

        err = self._target_err_norm
        ratio = self._target_ratio

        # Stage 5: relative capture check
        if ratio > CAPTURE_RATIO and abs(err) < CENTER_THRESHOLD:
            cname = 'RED' if self._target_type < 1.5 else 'BLUE'
            self.get_logger().info(f'[TARGET ACQUIRED] {cname} ratio={ratio:.3f} err={err:.3f}')
            self._lock_timer.cancel()
            self._enter_grab()
            return

        tx, ty, raw_dist, safe_dist, theta = self._project_target()
        if self._last_tx is not None:
            drift = math.sqrt((tx - self._last_tx)**2 + (ty - self._last_ty)**2)
            if drift < GOAL_FILTER_EPS:
                return
        self._last_tx = tx; self._last_ty = ty

        self.get_logger().info(
            f'[LOCK] err={err:.3f} ratio={ratio:.4f} rdist={raw_dist:.2f}m sdist={safe_dist:.2f}m -> ({tx:.2f},{ty:.2f})'
        )
        self._send_nav_goal(tx, ty)

    # ============ GRAB ============
    def _enter_grab(self):
        self.state = 'GRAB'
        cname = 'RED' if self._target_type < 1.5 else 'BLUE'
        self.get_logger().info(f'[GRAB] Loading {cname} cargo... (3s)')
        self._stop_robot()
        self.create_timer(GRAB_DURATION, self._grab_done, one_shot=True)

    def _grab_done(self):
        self.get_logger().info('[GRAB] Cargo loaded!')
        self._start_return()

    # ============ RETURN (Stage 5: semantic routing) ============
    def _start_return(self):
        self.state = 'RETURN'
        is_red = self._target_type < 1.5
        zone = RED_DROP_ZONE if is_red else BLUE_DROP_ZONE
        cname = 'RED zone' if is_red else 'BLUE zone'
        self.get_logger().info(f'[RETURN] Navigating to {cname} ({zone[0]:.1f}, {zone[1]:.1f})')
        self._send_nav_goal(zone[0], zone[1])

    # ============ DROP ============
    def _enter_drop(self):
        self.state = 'DROP'
        cname = 'RED' if self._target_type < 1.5 else 'BLUE'
        self.get_logger().info(f'[DROP] Unloading {cname} cargo... (3s)')
        self._stop_robot()
        self.create_timer(DROP_DURATION, self._drop_done, one_shot=True)

    def _drop_done(self):
        self.get_logger().info('[DROP] Done! Starting next round.')
        self._start_explore()

    # ============ Helpers ============
    def _stop_robot(self):
        stop = Twist(); stop.linear.x = 0.0; stop.angular.z = 0.0
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
            self._nav_failed()
            return
        self._goal_handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        result = future.result()
        status = result.status if result else -1

        if status == 4:  # SUCCEEDED
            if self.state == 'EXPLORE':
                self.get_logger().info('[EXPLORE] Waypoint reached, next...')
                self._start_explore()
            elif self.state == 'RETURN':
                self.get_logger().info('[RETURN] Arrived!')
                self._enter_drop()
            # LOCK timer handles its own completion
            return

        # Stage 5: Self-Healing — status != 4
        self.get_logger().warn(f'Nav2 failed (status={status}), self-healing...')
        self._nav_failed()

    def _nav_failed(self):
        """Self-healing: EXPLORE->next WP, RETURN->retry after 3s."""
        if self.state == 'EXPLORE':
            self.get_logger().info('[HEAL] EXPLORE goal failed, trying next waypoint')
            self._start_explore()
        elif self.state == 'RETURN':
            is_red = self._target_type < 1.5
            zone = RED_DROP_ZONE if is_red else BLUE_DROP_ZONE
            self.get_logger().info(f'[HEAL] RETURN failed, retrying in 3s...')
            self.create_timer(3.0, lambda: self._send_nav_goal(zone[0], zone[1]), one_shot=True)
        # LOCK: timer handles recovery via next projection


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
