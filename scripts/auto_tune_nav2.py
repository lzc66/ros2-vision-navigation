#!/usr/bin/env python3
"""
Auto-tune Nav2 parameters: 5 iterations of closed-loop simulation.
Tunes: max_vel_x [0.20, 0.60], inflation_radius [0.10, 0.20]

Stage 2 Debug v2 Fixes:
  - Safe goals: world(1.0, 0.5) or world(1.5, -0.5) - clear area between cylinders
  - cost_scaling_factor: 12.0 (sharp cost drop, clear corridors)
  - DWA: sim_time=2.0, xy_tol=0.15, yaw_tol=0.20
  - Fast-fail: if amcl_pose moves <0.1m over 15s with active cmd_vel, fail immediately
"""
import os, re, sys, time, signal, subprocess, math, random
from dataclasses import dataclass

PKG_DIR = os.path.join(os.path.dirname(__file__), '..')
NAV2_PARAMS = os.path.join(PKG_DIR, 'params', 'nav2_params.yaml')
WS_DIR = '/home/lzc/ros2_ws'

# === Tuning Ranges ===
MAX_VEL_X_MIN, MAX_VEL_X_MAX = 0.20, 0.60
INFLATION_MIN, INFLATION_MAX = 0.10, 0.20

# === Simulation ===
MAX_TIME = 90.0
CMD_VEL_TIMEOUT = 12.0
FAST_FAIL_STUCK_TIME = 15.0   # If moves <0.1m in this window with cmd_vel active -> fail
FAST_FAIL_MIN_DIST = 0.10

# === Fixed Safe Goals (between cylinders, away from hexagon boundaries) ===
SAFE_GOALS = [
    (1.5, -0.5),   # Primary: clear path, far from hexagons
    (1.0, 0.5),    # Backup: golden clear area
]
GOAL_TOLERANCE = 0.4
COLLISION_CHECK_INTERVAL = 0.5

# === Hard Constraints ===
MAP_ORIGIN_X, MAP_ORIGIN_Y = -5.0, -5.0
SPAWN_WX, SPAWN_WY = -2.0, -0.5
INIT_POSE_MAP_X = SPAWN_WX - MAP_ORIGIN_X
INIT_POSE_MAP_Y = SPAWN_WY - MAP_ORIGIN_Y

CYLINDERS = [(x, y) for x in [-1.1, 0.0, 1.1] for y in [-1.1, 0.0, 1.1]]
CYLINDER_RADIUS = 0.15
HEXAGONS = [(3.5, 0.0), (1.8, 2.7), (1.8, -2.7), (-1.8, 2.7), (-1.8, -2.7)]
HEXAGON_RADIUS = 0.45
BOUND_X_MIN, BOUND_X_MAX = -2.3, 2.3
BOUND_Y_MIN, BOUND_Y_MAX = -2.3, 2.3
MIN_OBSTACLE_DIST = 0.5


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
    stuck_failure: bool
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


def pick_goal():
    """Pick a verified safe goal from the approved list."""
    for gw in SAFE_GOALS:
        if is_goal_safe(gw[0], gw[1]):
            return gw
    return SAFE_GOALS[0]


def modify_params(max_vel_x, inflation_radius):
    with open(NAV2_PARAMS, 'r') as f:
        content = f.read()
    # RPP controller params
    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}desired_linear_vel:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    content = re.sub(r'(FollowPath:\n(?:.*\n)*?\s{4,8}max_linear_vel:\s*)[0-9.]+',
                     f'\\g<1>{max_vel_x}', content, count=1)
    # velocity_smoother
    content = re.sub(r'(velocity_smoother:\n(?:.*\n)*?\s{4,8}max_velocity:\s*\[)[0-9.]+,',
                     f'\\g<1>{max_vel_x},', content, count=1)
    # inflation_radius in both costmaps
    content = re.sub(r'(local_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
                     f'\\g<1>{inflation_radius}', content, count=1)
    content = re.sub(r'(global_costmap:\n(?:.*\n)*?\s{6,10}inflation_radius:\s*)[0-9.]+',
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
        subprocess.run(['ros2', 'topic', 'pub', '--once', '/initialpose',
                        'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
                       timeout=5, capture_output=True, env=os.environ)
        print(f"[INIT] Initial pose at map({INIT_POSE_MAP_X}, {INIT_POSE_MAP_Y})")
        return True
    except Exception as e:
        print(f"[INIT] Initial pose failed: {e}")
        return False


def _get_amcl_pose():
    """Get AMCL estimated map position."""
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/amcl_pose', '--once', '--field', 'pose.pose.position'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        match = re.search(r'x:\s*([\d.-]+)\s*\n\s*y:\s*([\d.-]+)', result.stdout)
        if match:
            return float(match.group(1)), float(match.group(2))
    except Exception:
        pass
    return None, None


def _verify_amcl_converged(timeout=10.0):
    print("[AMCL] Verifying convergence...")
    start = time.time()
    while time.time() - start < timeout:
        pose = _get_amcl_pose()
        if pose[0] is not None:
            dist = math.sqrt((pose[0] - INIT_POSE_MAP_X)**2 + (pose[1] - INIT_POSE_MAP_Y)**2)
            if dist < 2.0:
                print(f"[AMCL] Converged: map({pose[0]:.2f}, {pose[1]:.2f}) offset={dist:.2f}m")
                return True
        time.sleep(1.0)
    print("[AMCL] WARNING: Not converged")
    return False


def _get_cmd_vel_active():
    """Check if /cmd_vel has non-zero velocity."""
    try:
        result = subprocess.run([
            'ros2', 'topic', 'echo', '/cmd_vel', '--once', '--field', 'linear.x'
        ], capture_output=True, text=True, timeout=3, env=os.environ)
        match = re.search(r'(-?[\d.]+)', result.stdout)
        if match:
            return abs(float(match.group(1))) > 0.001
    except Exception:
        pass
    return False


def _get_odom():
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
    """Send nav goal + monitor with fast-fail on stuck-with-cmd_vel."""
    goal_mx, goal_my = world_to_map(goal_wx, goal_wy)
    # odom reports in world coordinates (Gazebo frame), so use world coords directly
    result = {
        'reached': False, 'collision': False, 'elapsed': timeout_sec,
        'final_x': 0.0, 'final_y': 0.0, 'plan_failure': False, 'stuck_failure': False
    }
    start_time = time.time()

    goal_cmd = [
        'ros2', 'action', 'send_goal', '/navigate_to_pose',
        'nav2_msgs/action/NavigateToPose',
        f'{{pose: {{header: {{frame_id: "map"}}, pose: {{position: {{x: {goal_mx}, y: {goal_my}, z: 0.0}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}}}'
    ]
    proc = subprocess.Popen(goal_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=os.environ)

    cmd_vel_dead_time = 0.0
    initial_amcl_pose = None
    collision_detected = False
    plan_failure = False
    stuck_failure = False
    first_cmd_vel = True

    while time.time() - start_time < timeout_sec:
        # --- cmd_vel monitoring ---
        has_cmd_vel = _get_cmd_vel_active()
        if has_cmd_vel:
            cmd_vel_dead_time = 0.0
            if first_cmd_vel:
                print("  [OK] cmd_vel active, robot moving")
                first_cmd_vel = False
        else:
            cmd_vel_dead_time += COLLISION_CHECK_INTERVAL
            if cmd_vel_dead_time > CMD_VEL_TIMEOUT and time.time() - start_time > 15.0:
                print(f"  [FAIL] No cmd_vel for {cmd_vel_dead_time:.0f}s - plan failure!")
                plan_failure = True
                result['plan_failure'] = True
                break

        # --- AMCL pose tracking for stuck detection ---
        amcl_pos = _get_amcl_pose()
        if amcl_pos[0] is not None:
            if initial_amcl_pose is None:
                initial_amcl_pose = amcl_pos
            # Total cumulative movement since start
            total_dist = math.sqrt(
                (amcl_pos[0] - initial_amcl_pose[0])**2 +
                (amcl_pos[1] - initial_amcl_pose[1])**2
            )
            # After significant time, if barely moved and cmd_vel active -> stuck
            if time.time() - start_time > 30.0 and total_dist < 0.3 and has_cmd_vel:
                print(f"  [STUCK] AMCL pose moved only {total_dist:.2f}m in {time.time()-start_time:.0f}s with active cmd_vel!")
                print(f"  [STUCK] Robot is spinning/blocked - fast fail!")
                stuck_failure = True
                result['stuck_failure'] = True
                break

        # --- Odometry goal check (world coords) ---
        odom_pos = _get_odom()
        if odom_pos[0] is not None:
            result['final_x'] = odom_pos[0]
            result['final_y'] = odom_pos[1]
            # odom reports Gazebo world coordinates, compare directly with world goal
            dist = math.sqrt((odom_pos[0] - goal_wx)**2 + (odom_pos[1] - goal_wy)**2)
            if dist < GOAL_TOLERANCE:
                result['reached'] = True
                result['elapsed'] = time.time() - start_time
                break

        if proc.poll() is not None:
            break
        time.sleep(COLLISION_CHECK_INTERVAL)

    if not result['reached'] and not collision_detected and not plan_failure and not stuck_failure:
        result['elapsed'] = timeout_sec

    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: proc.kill()

    return result


def run_iteration(iter_num, max_vel_x, inflation_radius) -> IterResult:
    modify_params(max_vel_x, inflation_radius)
    goal_wx, goal_wy = pick_goal()
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

    _send_initial_pose()
    time.sleep(2)
    _verify_amcl_converged(timeout=8.0)

    print(f"[GOAL] Sending goal: world({goal_wx:.1f}, {goal_wy:.1f})...")
    goal_result = _send_nav_goal(goal_wx, goal_wy, MAX_TIME)

    reached = goal_result['reached']
    collision = goal_result['collision']
    plan_failure = goal_result['plan_failure']
    stuck_failure = goal_result['stuck_failure']
    elapsed = goal_result['elapsed']
    final_x = goal_result['final_x']
    final_y = goal_result['final_y']

    # Scoring
    if stuck_failure:
        score = 0.0
        print(f"[RESULT] STUCK FAILURE (amcl_pose drift <0.1m)! Score: {score}")
    elif plan_failure:
        score = 0.0
        print(f"[RESULT] PLAN FAILURE (no cmd_vel)! Score: {score}")
    elif collision:
        score = 0.0
        print(f"[RESULT] COLLISION! Score: {score}")
    elif reached:
        score = (MAX_TIME - elapsed) * 10.0
        print(f"[RESULT] GOAL REACHED in {elapsed:.1f}s! Score: {score:.1f}")
    else:
        score = 10.0
        dist = math.sqrt((final_x - goal_wx)**2 + (final_y - goal_wy)**2)
        print(f"[RESULT] TIMEOUT. Dist to goal: {dist:.2f}m. Score: {score}")

    print("[CLEANUP] Killing all processes...")
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)
    kill_all()

    return IterResult(
        iteration=iter_num, max_vel_x=max_vel_x, inflation_radius=inflation_radius,
        score=score, elapsed_time=elapsed, reached_goal=reached, collision=collision,
        plan_failure=plan_failure, stuck_failure=stuck_failure,
        final_x=final_x, final_y=final_y
    )


def heuristic_next_params(iter_num, all_results):
    if iter_num == 1:
        return 0.30, 0.15

    best = max(all_results, key=lambda r: r.score)
    last = all_results[-1]
    candidates = []

    if last.stuck_failure or last.plan_failure:
        # Reduce speed, try different inflation
        candidates.append((max(0.20, last.max_vel_x - 0.06), min(INFLATION_MAX, last.inflation_radius + 0.03)))
        candidates.append((max(0.20, last.max_vel_x - 0.03), min(INFLATION_MAX, last.inflation_radius + 0.02)))
    elif last.reached_goal:
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.08), last.inflation_radius))
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.05), max(INFLATION_MIN, last.inflation_radius - 0.03)))
    else:
        # Timeout - vary speed and inflation
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.06), last.inflation_radius))
        candidates.append((last.max_vel_x, max(INFLATION_MIN, last.inflation_radius - 0.03)))
        candidates.append((min(MAX_VEL_X_MAX, last.max_vel_x + 0.04), max(INFLATION_MIN, last.inflation_radius - 0.02)))
        candidates.append((last.max_vel_x, min(INFLATION_MAX, last.inflation_radius + 0.03)))

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
    print("Nav2 Auto-Tuning: 5 Iterations [Stage 2 Debug v2]")
    print(f"  Spawn: world({SPAWN_WX}, {SPAWN_WY})  |  Bounds: [{BOUND_X_MIN},{BOUND_X_MAX}]")
    print(f"  inflation_radius: [{INFLATION_MIN}, {INFLATION_MAX}]  |  cost_scaling: 12.0")
    print(f"  DWA: sim_time=2.0, xy_tol=0.15, yaw_tol=0.20")
    print(f"  Fast-fail: <{FAST_FAIL_MIN_DIST:.2f}m in {FAST_FAIL_STUCK_TIME:.0f}s = stuck")
    print(f"  Safe goals: {SAFE_GOALS}")
    print("=" * 70)

    print("\n[MAP] Generating map...")
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
              f"plan_fail={result.plan_failure} stuck={result.stuck_failure} t={result.elapsed_time:.1f}s")

    best = max(all_results, key=lambda r: r.score)
    modify_params(best.max_vel_x, best.inflation_radius)
    print(f"\n{'='*70}")
    print(f"BEST: max_vel_x={best.max_vel_x:.2f}  inflation_radius={best.inflation_radius:.2f}  Score={best.score:.1f}")
    print(f"{'='*70}")

    print(f"\n{'Iter':<6} {'v':<8} {'r':<8} {'Score':<10} {'Time':<8} {'Reached':<8} {'PlanFail':<9} {'StuckFail':<10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r.iteration:<6} {r.max_vel_x:<8.2f} {r.inflation_radius:<8.2f} "
              f"{r.score:<10.1f} {r.elapsed_time:<8.1f} {str(r.reached_goal):<8} "
              f"{str(r.plan_failure):<9} {str(r.stuck_failure):<10}")


if __name__ == '__main__':
    main()
