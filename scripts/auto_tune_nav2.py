#!/usr/bin/env python3
"""
Auto-tune Nav2 parameters across 5 iterations of closed-loop simulation.
Tunes: max_vel_x (DWA) and inflation_radius (local + global costmap).
Heuristic dynamically adapts based on previous iteration score.

Data Dictionary (Hard Constraints):
  - Spawn: world(-2.0, -0.5, 0.1) = map(3.0, 4.5)
  - Cylinders at world(±1.1, ±1.1), (±1.1, 0), (0, ±1.1), (0,0) radius=0.15m
  - Safe bounding box: world x,y ∈ [-2.3, 2.3]
  - Goals must be >0.5m from any cylinder center
"""
import os
import re
import sys
import time
import signal
import subprocess
import math
import random
from dataclasses import dataclass

# Paths
PKG_DIR = os.path.join(os.path.dirname(__file__), '..')
NAV2_PARAMS = os.path.join(PKG_DIR, 'params', 'nav2_params.yaml')
WS_DIR = '/home/lzc/ros2_ws'

# Tunable ranges
MAX_VEL_X_MIN, MAX_VEL_X_MAX = 0.20, 0.60
INFLATION_MIN, INFLATION_MAX = 0.20, 0.50

# Simulation limits
MAX_TIME = 90.0

# === Data Dictionary: Hard Constraints ===
# Map origin at world(-5.0, -5.0). map(x,y) = world(x+5, y+5)
MAP_ORIGIN_X, MAP_ORIGIN_Y = -5.0, -5.0

# Robot spawn at world(-2.0, -0.5) = map(3.0, 4.5)
SPAWN_WX, SPAWN_WY = -2.0, -0.5
INIT_POSE_MAP_X = SPAWN_WX - MAP_ORIGIN_X  # 3.0
INIT_POSE_MAP_Y = SPAWN_WY - MAP_ORIGIN_Y  # 4.5

# Cylinder obstacle centers (world coords, from SDF)
CYLINDER_GRID = [-1.1, 0.0, 1.1]
CYLINDER_RADIUS = 0.15
CYLINDERS = [(x, y) for x in CYLINDER_GRID for y in CYLINDER_GRID]

# Hexagon obstacle centers (world coords, from SDF)
HEXAGONS = [
    (3.5, 0.0), (1.8, 2.7), (1.8, -2.7), (-1.8, 2.7), (-1.8, -2.7)
]
HEXAGON_RADIUS = 0.45

# Safe bounding box (world coords)
BOUND_X_MIN, BOUND_X_MAX = -2.3, 2.3
BOUND_Y_MIN, BOUND_Y_MAX = -2.3, 2.3

# Minimum distance from goal to any obstacle center
MIN_OBSTACLE_DIST = 0.5

# Goal tolerance
GOAL_TOLERANCE = 0.4
COLLISION_CHECK_INTERVAL = 1.0


@dataclass
class IterResult:
    iteration: int
    max_vel_x: float
    inflation_radius: float
    score: float
    elapsed_time: float
    reached_goal: bool
    collision: bool
    final_x: float
    final_y: float


def world_to_map(wx, wy):
    return wx - MAP_ORIGIN_X, wy - MAP_ORIGIN_Y


def is_goal_safe(wx, wy):
    """Check if a world-coord goal is within bounds and away from obstacles."""
    if not (BOUND_X_MIN <= wx <= BOUND_X_MAX and BOUND_Y_MIN <= wy <= BOUND_Y_MAX):
        return False
    # Check cylinders
    for cx, cy in CYLINDERS:
        dist = math.sqrt((wx - cx)**2 + (wy - cy)**2)
        if dist < CYLINDER_RADIUS + MIN_OBSTACLE_DIST:
            return False
    # Check hexagons
    for hx, hy in HEXAGONS:
        dist = math.sqrt((wx - hx)**2 + (wy - hy)**2)
        if dist < HEXAGON_RADIUS + MIN_OBSTACLE_DIST:
            return False
    return True


def generate_safe_goal():
    """Generate a random goal within bounds and >0.5m from obstacles."""
    safe_candidates = [
        (2.0, -0.5),   # straight east, ~4m clear path
        (2.0, 0.5),    # east-north
        (1.5, 1.5),    # diagonal
        (-1.5, 1.5),   # north-west
        (0.0, 2.0),    # north
        (-2.0, 0.5),   # west
        (1.0, -1.5),   # south-east
    ]
    # Filter and shuffle
    valid = [g for g in safe_candidates if is_goal_safe(g[0], g[1])]
    if valid:
        return random.choice(valid)
    # Fallback: random search within bounds
    for _ in range(100):
        wx = random.uniform(BOUND_X_MIN, BOUND_X_MAX)
        wy = random.uniform(BOUND_Y_MIN, BOUND_Y_MAX)
        if is_goal_safe(wx, wy):
            return wx, wy
    return 2.0, -0.5  # Last resort: safe straight line


def modify_params(max_vel_x, inflation_radius):
    """Modify nav2_params.yaml with new tuning values."""
    with open(NAV2_PARAMS, 'r') as f:
        content = f.read()

    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}max_vel_x:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}max_speed_xy:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    content = re.sub(
        r'(velocity_smoother:\n(?:.*\n)*?\s{4,8}max_velocity:\s*\[)[0-9.]+,',
        f'\\g<1>{max_vel_x},', content, count=1)
    content = re.sub(
        r'(local_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
        f'\\g<1>{inflation_radius}', content, count=1)
    content = re.sub(
        r'(global_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
        f'\\g<1>{inflation_radius}', content, count=1)

    with open(NAV2_PARAMS, 'w') as f:
        f.write(content)
    return True


def kill_all():
    """Force kill all Gazebo and ROS 2 processes."""
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient', 'ruby'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'navigation.launch'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzserver'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzclient'], capture_output=True)
    time.sleep(3)


def _send_initial_pose():
    """Send initial pose to AMCL at map(3.0, 4.5) = world(-2.0, -0.5)."""
    msg = (
        '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "map"}, '
        f'pose: {{pose: {{position: {{x: {INIT_POSE_MAP_X}, y: {INIT_POSE_MAP_Y}, z: 0.0}}, '
        'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, '
        'covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.25, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '
        '0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]}}'
    )
    try:
        subprocess.run(
            ['ros2', 'topic', 'pub', '--once', '/initialpose',
             'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
            timeout=5, capture_output=True, env=os.environ
        )
        print(f"[INIT] Initial pose at map({INIT_POSE_MAP_X}, {INIT_POSE_MAP_Y}) = world({SPAWN_WX}, {SPAWN_WY})")
    except Exception as e:
        print(f"[INIT] Failed to send initial pose: {e}")


def _unpause_physics():
    """Unpause Gazebo physics after all setup is done."""
    try:
        subprocess.run(
            ['ros2', 'service', 'call', '/unpause_physics', 'std_srvs/srv/Empty', '{}'],
            timeout=5, capture_output=True, env=os.environ
        )
        print("[INIT] Physics unpaused")
    except Exception as e:
        print(f"[INIT] Unpause failed: {e}")


def _get_robot_odom():
    """Get current robot position from /odom topic."""
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/odom', '--once', '--field', 'pose.pose.position'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        match = re.search(r'x:\s*([\d.-]+)\s*\n\s*y:\s*([\d.-]+)', result.stdout)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None


def _send_nav_goal(goal_wx, goal_wy, timeout_sec):
    """Send navigation goal and monitor progress."""
    goal_mx, goal_my = world_to_map(goal_wx, goal_wy)

    result = {'reached': False, 'collision': False, 'elapsed': timeout_sec,
              'final_x': 0.0, 'final_y': 0.0}
    start_time = time.time()

    goal_cmd = [
        'ros2', 'action', 'send_goal', '/navigate_to_pose',
        'nav2_msgs/action/NavigateToPose',
        f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {goal_mx}, y: {goal_my}, z: 0.0}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}}}'
    ]
    proc = subprocess.Popen(goal_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=os.environ)

    # The odom frame tracks robot movement. Robot starts at world(SPAWN_WX, SPAWN_WY).
    # Goal distance in world coords = offset from spawn to goal.
    goal_odom_x = goal_wx - SPAWN_WX
    goal_odom_y = goal_wy - SPAWN_WY

    previous_x, previous_y = 0.0, 0.0
    stuck_count = 0
    collision_detected = False

    while time.time() - start_time < timeout_sec:
        odom_x, odom_y = _get_robot_odom()
        if odom_x is not None:
            result['final_x'] = odom_x
            result['final_y'] = odom_y

            dist_to_goal = math.sqrt((odom_x - goal_odom_x)**2 + (odom_y - goal_odom_y)**2)
            if dist_to_goal < GOAL_TOLERANCE:
                result['reached'] = True
                result['elapsed'] = time.time() - start_time
                break

            movement = math.sqrt((odom_x - previous_x)**2 + (odom_y - previous_y)**2)
            if movement < 0.02:
                stuck_count += 1
            else:
                stuck_count = 0
            previous_x, previous_y = odom_x, odom_y

            if stuck_count > 10:
                print(f"  [WARN] Robot stuck at odom({odom_x:.2f}, {odom_y:.2f})")
                collision_detected = True
                result['collision'] = True
                result['elapsed'] = time.time() - start_time
                break

        if proc.poll() is not None:
            break
        time.sleep(COLLISION_CHECK_INTERVAL)

    if not result['reached'] and not collision_detected:
        result['elapsed'] = timeout_sec

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return result


def run_iteration(iter_num, max_vel_x, inflation_radius) -> IterResult:
    """Run a single tuning iteration."""
    modify_params(max_vel_x, inflation_radius)

    # Pick a safe goal
    goal_wx, goal_wy = generate_safe_goal()
    goal_mx, goal_my = world_to_map(goal_wx, goal_wy)

    print(f"\n{'='*60}")
    print(f"[Iter {iter_num}] max_vel_x={max_vel_x:.2f}, inflation_radius={inflation_radius:.2f}")
    print(f"  Goal: world({goal_wx:.1f}, {goal_wy:.1f}) = map({goal_mx:.1f}, {goal_my:.1f})")
    print(f"{'='*60}")

    # Build
    print("[BUILD] Building...")
    build_result = subprocess.run(
        ['colcon', 'build', '--packages-select', 'project', '--symlink-install'],
        cwd=WS_DIR, capture_output=True, text=True, timeout=120
    )
    if 'failed' in build_result.stderr.lower():
        print(f"[WARN] Build errors: {build_result.stderr[-200:]}")

    # Launch
    print("[LAUNCH] Starting navigation simulation...")
    env = os.environ.copy()
    env['GAZEBO_MODEL_PATH'] = (
        f"{PKG_DIR}/models:"
        f"{os.path.expanduser('~/.gazebo/models')}:"
        "/opt/ros/humble/share/turtlebot3_gazebo/models"
    )
    proc = subprocess.Popen(
        ['ros2', 'launch', 'project', 'navigation.launch.py'],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    # Wait for system init
    print("[INIT] Waiting for system (25s)...")
    time.sleep(25)

    # Send initial pose and unpause
    _send_initial_pose()
    _unpause_physics()
    time.sleep(5)

    # Send goal
    print(f"[GOAL] Sending goal...")
    goal_result = _send_nav_goal(goal_wx, goal_wy, MAX_TIME)

    reached = goal_result['reached']
    collision = goal_result['collision']
    elapsed = goal_result['elapsed']
    final_x = goal_result['final_x']
    final_y = goal_result['final_y']

    if collision:
        score = 0.0
        print(f"[RESULT] COLLISION! Score: {score}")
    elif reached:
        score = (MAX_TIME - elapsed) * 10.0
        print(f"[RESULT] GOAL REACHED in {elapsed:.1f}s! Score: {score:.1f}")
    else:
        score = 10.0
        goal_odom_x = goal_wx - SPAWN_WX
        goal_odom_y = goal_wy - SPAWN_WY
        dist = math.sqrt((final_x - goal_odom_x)**2 + (final_y - goal_odom_y)**2)
        print(f"[RESULT] TIMEOUT. Dist to goal: {dist:.2f}m. Score: {score}")

    # Cleanup
    print("[CLEANUP] Killing all processes...")
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)
    kill_all()

    return IterResult(
        iteration=iter_num, max_vel_x=max_vel_x, inflation_radius=inflation_radius,
        score=score, elapsed_time=elapsed, reached_goal=reached, collision=collision,
        final_x=final_x, final_y=final_y
    )


def heuristic_next_params(iter_num, all_results):
    """Dynamically generate next parameters based on previous results."""
    if iter_num == 1:
        return 0.26, 0.40

    best = max(all_results, key=lambda r: r.score)
    last = all_results[-1]
    candidates = []

    if last.collision:
        candidates.append((max(0.20, last.max_vel_x - 0.08), min(0.50, last.inflation_radius + 0.05)))
        candidates.append((max(0.20, last.max_vel_x - 0.04), min(0.50, last.inflation_radius + 0.08)))
    elif last.reached_goal:
        if last.max_vel_x < MAX_VEL_X_MAX - 0.05:
            candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.08), last.inflation_radius))
            candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.05),
                              max(0.20, last.inflation_radius - 0.05)))
        else:
            candidates.append((last.max_vel_x, max(0.20, last.inflation_radius - 0.08)))
    else:
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.06), last.inflation_radius))
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.06),
                          max(0.20, last.inflation_radius - 0.05)))

    candidates.append((best.max_vel_x + 0.04 if best.max_vel_x < 0.50 else best.max_vel_x - 0.03,
                       best.inflation_radius - 0.04 if best.inflation_radius > 0.24 else best.inflation_radius + 0.03))

    valid = [(v, r) for v, r in candidates
             if MAX_VEL_X_MIN <= v <= MAX_VEL_X_MAX and INFLATION_MIN <= r <= INFLATION_MAX]
    used = {(r.max_vel_x, r.inflation_radius) for r in all_results}
    for v, r in valid:
        if (v, r) not in used:
            return round(v, 2), round(r, 2)

    while True:
        v = round(random.uniform(MAX_VEL_X_MIN, MAX_VEL_X_MAX), 2)
        r = round(random.uniform(INFLATION_MIN, INFLATION_MAX), 2)
        if (v, r) not in used:
            return v, r


def git_commit(iter_num, max_vel_x, radius, score):
    """Git commit the params file."""
    try:
        subprocess.run(['git', 'add', '-f', NAV2_PARAMS], cwd=PKG_DIR, capture_output=True)
        msg = f"[Auto-Tune] Iter {iter_num}: max_vel={max_vel_x:.2f}, radius={radius:.2f} | Score: {score:.1f}"
        subprocess.run(['git', 'commit', '-m', msg], cwd=PKG_DIR, capture_output=True)
        print(f"[GIT] Committed: {msg}")
    except Exception as e:
        print(f"[GIT] Commit failed: {e}")


def main():
    print("=" * 70)
    print("Nav2 Auto-Tuning: 5 Iterations of Closed-Loop Optimization")
    print(f"  Spawn: world({SPAWN_WX}, {SPAWN_WY})  |  Bounds: {[BOUND_X_MIN, BOUND_X_MAX]} x {[BOUND_Y_MIN, BOUND_Y_MAX]}")
    print("=" * 70)

    print("\n[MAP] Generating 2D map...")
    subprocess.run([sys.executable, os.path.join(PKG_DIR, 'scripts', 'generate_map.py')],
                   cwd=PKG_DIR, capture_output=True)

    kill_all()
    all_results = []

    for iteration in range(1, 6):
        max_vel, inflation = heuristic_next_params(iteration, all_results)
        result = run_iteration(iteration, max_vel, inflation)
        all_results.append(result)
        git_commit(iteration, max_vel, inflation, result.score)
        print(f"\n[ITER {iteration} SUMMARY] v={max_vel:.2f} r={inflation:.2f} "
              f"score={result.score:.1f} reached={result.reached_goal} "
              f"collision={result.collision} time={result.elapsed_time:.1f}s")

    # Best params
    best = max(all_results, key=lambda r: r.score)
    modify_params(best.max_vel_x, best.inflation_radius)
    print(f"\n{'='*70}")
    print("BEST PARAMETERS:")
    print(f"  max_vel_x = {best.max_vel_x:.2f}")
    print(f"  inflation_radius = {best.inflation_radius:.2f}")
    print(f"  Score = {best.score:.1f}")
    print(f"{'='*70}")

    # Report
    print(f"\n{'Iter':<6} {'max_vel_x':<12} {'inflation':<12} {'Score':<10} "
          f"{'Time':<8} {'Reached':<8} {'Collision':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.iteration:<6} {r.max_vel_x:<12.2f} {r.inflation_radius:<12.2f} "
              f"{r.score:<10.1f} {r.elapsed_time:<8.1f} {str(r.reached_goal):<8} "
              f"{str(r.collision):<10}")

    print(f"\nFinal parameters written to: {NAV2_PARAMS}")


if __name__ == '__main__':
    main()
