#!/usr/bin/env python3
"""
Auto-tune Nav2 parameters across 5 iterations of closed-loop simulation.
Tunes: max_vel_x (DWA) and inflation_radius (local + global costmap).

Stage 2 Debug Fixes:
  - inflation_radius range: [0.10, 0.25] (was [0.20, 0.50] - blocked 0.8m passages)
  - Motion model: MOORE (was DUBIN - TB3 can rotate in place, no turning radius needed)
  - /cmd_vel monitor: fail if no non-zero velocity within 10s of goal
  - amcl_pose verification before sending goal
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

# === Debug-Fixed Tunable Ranges ===
MAX_VEL_X_MIN, MAX_VEL_X_MAX = 0.20, 0.60
INFLATION_MIN, INFLATION_MAX = 0.10, 0.25  # Fixed: was [0.20, 0.50], now respects 0.8m passages

MAX_TIME = 90.0
CMD_VEL_TIMEOUT = 12.0  # Fail if no non-zero cmd_vel within this time

# === Data Dictionary: Hard Constraints ===
MAP_ORIGIN_X, MAP_ORIGIN_Y = -5.0, -5.0
SPAWN_WX, SPAWN_WY = -2.0, -0.5
INIT_POSE_MAP_X = SPAWN_WX - MAP_ORIGIN_X  # 3.0
INIT_POSE_MAP_Y = SPAWN_WY - MAP_ORIGIN_Y  # 4.5

CYLINDERS = [(x, y) for x in [-1.1, 0.0, 1.1] for y in [-1.1, 0.0, 1.1]]
CYLINDER_RADIUS = 0.15
HEXAGONS = [(3.5, 0.0), (1.8, 2.7), (1.8, -2.7), (-1.8, 2.7), (-1.8, -2.7)]
HEXAGON_RADIUS = 0.45
BOUND_X_MIN, BOUND_X_MAX = -2.3, 2.3
BOUND_Y_MIN, BOUND_Y_MAX = -2.3, 2.3
MIN_OBSTACLE_DIST = 0.5
GOAL_TOLERANCE = 0.4
COLLISION_CHECK_INTERVAL = 0.5  # Check more frequently


@dataclass
class IterResult:
    iteration: int
    max_vel_x: float
    inflation_radius: float
    score: float
    elapsed_time: float
    reached_goal: bool
    collision: bool
    plan_failure: bool
    final_x: float
    final_y: float


def world_to_map(wx, wy):
    return wx - MAP_ORIGIN_X, wy - MAP_ORIGIN_Y


def is_goal_safe(wx, wy):
    if not (BOUND_X_MIN <= wx <= BOUND_X_MAX and BOUND_Y_MIN <= wy <= BOUND_Y_MAX):
        return False
    for cx, cy in CYLINDERS:
        if math.sqrt((wx - cx)**2 + (wy - cy)**2) < CYLINDER_RADIUS + MIN_OBSTACLE_DIST:
            return False
    for hx, hy in HEXAGONS:
        if math.sqrt((wx - hx)**2 + (wy - hy)**2) < HEXAGON_RADIUS + MIN_OBSTACLE_DIST:
            return False
    return True


def generate_safe_goal():
    safe_candidates = [
        (2.0, -0.5), (2.0, 0.5), (1.5, 1.5), (-1.5, 1.5),
        (0.0, 2.0), (-2.0, 0.5), (1.0, -1.5),
    ]
    valid = [g for g in safe_candidates if is_goal_safe(g[0], g[1])]
    if valid:
        return random.choice(valid)
    for _ in range(200):
        wx = random.uniform(BOUND_X_MIN, BOUND_X_MAX)
        wy = random.uniform(BOUND_Y_MIN, BOUND_Y_MAX)
        if is_goal_safe(wx, wy):
            return wx, wy
    return 2.0, -0.5


def modify_params(max_vel_x, inflation_radius):
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
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient', 'ruby'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'navigation.launch'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzserver'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'gzclient'], capture_output=True)
    time.sleep(3)


def _send_initial_pose():
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
        print(f"[INIT] Initial pose at map({INIT_POSE_MAP_X}, {INIT_POSE_MAP_Y})")
        return True
    except Exception as e:
        print(f"[INIT] Failed to send initial pose: {e}")
        return False


def _verify_amcl_converged(timeout=10.0):
    """Check amcl_pose topic to verify AMCL has converged."""
    print("[AMCL] Verifying AMCL convergence...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run([
                'ros2', 'topic', 'echo', '/amcl_pose', '--once',
                '--field', 'pose.pose.position'
            ], capture_output=True, text=True, timeout=3, env=os.environ)
            if 'x:' in result.stdout and 'y:' in result.stdout:
                match = re.search(r'x:\s*([\d.-]+)\s*\n\s*y:\s*([\d.-]+)', result.stdout)
                if match:
                    amcl_x, amcl_y = float(match.group(1)), float(match.group(2))
                    # Check if AMCL pose is within reasonable range of expected pose
                    dist_from_expected = math.sqrt(
                        (amcl_x - INIT_POSE_MAP_X)**2 + (amcl_y - INIT_POSE_MAP_Y)**2
                    )
                    if dist_from_expected < 2.0:
                        print(f"[AMCL] Converged: map({amcl_x:.2f}, {amcl_y:.2f}) offset={dist_from_expected:.2f}m")
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    print("[AMCL] WARNING: AMCL did not converge, proceeding anyway")
    return False


def _get_robot_odom():
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


def _get_cmd_vel():
    """Check if /cmd_vel has a non-zero velocity recently."""
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/cmd_vel', '--once',
            '--field', 'linear.x'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        match = re.search(r'(-?[\d.]+)', result.stdout)
        if match:
            vel_x = float(match.group(1))
            return abs(vel_x) > 0.001
    except Exception:
        pass
    return False


def _send_nav_goal(goal_wx, goal_wy, timeout_sec):
    """Send navigation goal and monitor progress with /cmd_vel checking."""
    goal_mx, goal_my = world_to_map(goal_wx, goal_wy)
    goal_odom_x = goal_wx - SPAWN_WX
    goal_odom_y = goal_wy - SPAWN_WY

    result = {
        'reached': False, 'collision': False, 'elapsed': timeout_sec,
        'final_x': 0.0, 'final_y': 0.0, 'plan_failure': False
    }
    start_time = time.time()

    # Send navigation goal
    goal_cmd = [
        'ros2', 'action', 'send_goal', '/navigate_to_pose',
        'nav2_msgs/action/NavigateToPose',
        f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {goal_mx}, y: {goal_my}, z: 0.0}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}}}'
    ]
    proc = subprocess.Popen(goal_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=os.environ)

    # Monitor loop
    previous_x, previous_y = 0.0, 0.0
    stuck_count = 0
    cmd_vel_dead_time = 0.0
    collision_detected = False
    plan_failure = False
    first_iteration = True

    while time.time() - start_time < timeout_sec:
        # --- cmd_vel check ---
        has_cmd_vel = _get_cmd_vel()
        if not has_cmd_vel:
            cmd_vel_dead_time += COLLISION_CHECK_INTERVAL
            if cmd_vel_dead_time > CMD_VEL_TIMEOUT and time.time() - start_time > 15.0:
                print(f"  [FAIL] No cmd_vel for {cmd_vel_dead_time:.0f}s - plan failure!")
                plan_failure = True
                result['plan_failure'] = True
                result['elapsed'] = time.time() - start_time
                break
        else:
            cmd_vel_dead_time = 0.0  # Reset on non-zero cmd_vel
            if first_iteration:
                print(f"  [OK] cmd_vel active, robot is moving")
                first_iteration = False

        # --- odom check ---
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
            if movement < 0.01:
                stuck_count += 1
            else:
                stuck_count = 0
            previous_x, previous_y = odom_x, odom_y

            if stuck_count > 20:
                print(f"  [WARN] Robot stuck at odom({odom_x:.2f}, {odom_y:.2f})")
                collision_detected = True
                result['collision'] = True
                result['elapsed'] = time.time() - start_time
                break

        if proc.poll() is not None:
            break
        time.sleep(COLLISION_CHECK_INTERVAL)

    if not result['reached'] and not collision_detected and not plan_failure:
        result['elapsed'] = timeout_sec

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return result


def run_iteration(iter_num, max_vel_x, inflation_radius) -> IterResult:
    modify_params(max_vel_x, inflation_radius)
    goal_wx, goal_wy = generate_safe_goal()
    goal_mx, goal_my = world_to_map(goal_wx, goal_wy)

    print(f"\n{'='*60}")
    print(f"[Iter {iter_num}] max_vel_x={max_vel_x:.2f}, inflation_radius={inflation_radius:.2f}")
    print(f"  Goal: world({goal_wx:.1f}, {goal_wy:.1f}) = map({goal_mx:.1f}, {goal_my:.1f})")
    print(f"{'='*60}")

    print("[BUILD] Building...")
    build_result = subprocess.run(
        ['colcon', 'build', '--packages-select', 'project', '--symlink-install'],
        cwd=WS_DIR, capture_output=True, text=True, timeout=120
    )
    if 'failed' in build_result.stderr.lower():
        print(f"[WARN] Build errors: {build_result.stderr[-200:]}")

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

    print("[INIT] Waiting for system (25s)...")
    time.sleep(25)

    # Send initial pose
    _send_initial_pose()
    time.sleep(2)

    # Verify AMCL convergence
    _verify_amcl_converged(timeout=8.0)

    # Send goal
    print(f"[GOAL] Sending goal...")
    goal_result = _send_nav_goal(goal_wx, goal_wy, MAX_TIME)

    reached = goal_result['reached']
    collision = goal_result['collision']
    plan_failure = goal_result['plan_failure']
    elapsed = goal_result['elapsed']
    final_x = goal_result['final_x']
    final_y = goal_result['final_y']

    if plan_failure:
        score = 0.0
        print(f"[RESULT] PLAN FAILURE (no cmd_vel)! Score: {score}")
    elif collision:
        score = 0.0
        print(f"[RESULT] COLLISION/STUCK! Score: {score}")
    elif reached:
        score = (MAX_TIME - elapsed) * 10.0
        print(f"[RESULT] GOAL REACHED in {elapsed:.1f}s! Score: {score:.1f}")
    else:
        score = 10.0
        goal_odom_x = goal_wx - SPAWN_WX
        goal_odom_y = goal_wy - SPAWN_WY
        dist = math.sqrt((final_x - goal_odom_x)**2 + (final_y - goal_odom_y)**2)
        print(f"[RESULT] TIMEOUT. Dist to goal: {dist:.2f}m. Score: {score}")

    print("[CLEANUP] Killing all processes...")
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)
    kill_all()

    return IterResult(
        iteration=iter_num, max_vel_x=max_vel_x, inflation_radius=inflation_radius,
        score=score, elapsed_time=elapsed, reached_goal=reached, collision=collision,
        plan_failure=plan_failure, final_x=final_x, final_y=final_y
    )


def heuristic_next_params(iter_num, all_results):
    if iter_num == 1:
        return 0.26, 0.15  # Start with small inflation

    best = max(all_results, key=lambda r: r.score)
    last = all_results[-1]
    candidates = []

    if last.plan_failure or last.collision:
        candidates.append((max(0.20, last.max_vel_x - 0.06), min(INFLATION_MAX, last.inflation_radius + 0.04)))
        candidates.append((max(0.20, last.max_vel_x - 0.03), min(INFLATION_MAX, last.inflation_radius + 0.02)))
    elif last.reached_goal:
        if last.max_vel_x < MAX_VEL_X_MAX - 0.05:
            candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.08), last.inflation_radius))
            candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.05),
                              max(INFLATION_MIN, last.inflation_radius - 0.03)))
        else:
            candidates.append((last.max_vel_x, max(INFLATION_MIN, last.inflation_radius - 0.04)))
    else:
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.06), last.inflation_radius))
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.06),
                          max(INFLATION_MIN, last.inflation_radius - 0.03)))
        candidates.append((last.max_vel_x, max(INFLATION_MIN, last.inflation_radius - 0.05)))

    candidates.append((best.max_vel_x + 0.03 if best.max_vel_x < 0.50 else best.max_vel_x - 0.03,
                       best.inflation_radius - 0.02 if best.inflation_radius > 0.12 else best.inflation_radius + 0.02))

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
    try:
        subprocess.run(['git', 'add', '-f', NAV2_PARAMS], cwd=PKG_DIR, capture_output=True)
        msg = f"[Auto-Tune] Iter {iter_num}: max_vel={max_vel_x:.2f}, radius={radius:.2f} | Score: {score:.1f}"
        subprocess.run(['git', 'commit', '-m', msg], cwd=PKG_DIR, capture_output=True)
        print(f"[GIT] Committed: {msg}")
    except Exception as e:
        print(f"[GIT] Commit failed: {e}")


def main():
    print("=" * 70)
    print("Nav2 Auto-Tuning: 5 Iterations [Stage 2 Debug Fixed]")
    print(f"  Spawn: world({SPAWN_WX}, {SPAWN_WY})  |  Bounds: [{BOUND_X_MIN},{BOUND_X_MAX}]x[{BOUND_Y_MIN},{BOUND_Y_MAX}]")
    print(f"  inflation_radius range: [{INFLATION_MIN}, {INFLATION_MAX}]")
    print(f"  Motion model: MOORE  |  cmd_vel timeout: {CMD_VEL_TIMEOUT}s")
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
              f"plan_fail={result.plan_failure} time={result.elapsed_time:.1f}s")

    best = max(all_results, key=lambda r: r.score)
    modify_params(best.max_vel_x, best.inflation_radius)
    print(f"\n{'='*70}")
    print("BEST PARAMETERS:")
    print(f"  max_vel_x = {best.max_vel_x:.2f}")
    print(f"  inflation_radius = {best.inflation_radius:.2f}")
    print(f"  Score = {best.score:.1f}")
    print(f"{'='*70}")

    print(f"\n{'Iter':<6} {'max_vel':<10} {'infl_r':<10} {'Score':<10} {'Time':<8} {'Reached':<8} {'PlanFail':<9}")
    print("-" * 65)
    for r in all_results:
        print(f"{r.iteration:<6} {r.max_vel_x:<10.2f} {r.inflation_radius:<10.2f} "
              f"{r.score:<10.1f} {r.elapsed_time:<8.1f} {str(r.reached_goal):<8} {str(r.plan_failure):<9}")

    print(f"\nFinal parameters written to: {NAV2_PARAMS}")


if __name__ == '__main__':
    main()
