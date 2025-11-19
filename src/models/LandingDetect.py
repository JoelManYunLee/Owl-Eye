import cv2
import numpy as np
from collections import deque
import sys
import os

# Add path for HitDetect import
sys.path.append(os.path.join(os.path.dirname(__file__)))
try:
    from HitDetect import Pose
except ImportError:
    # If HitDetect import fails (e.g., missing tensorflow), define a minimal Pose class
    class Pose:
        def __init__(self):
            self.kp = None
            self.bx = None
            self.by = None
        
        def init_from_kparray(self, kparray):
            kp = np.array(kparray).reshape((17, 2))
            self.kp = kp
            valid_kp = kp[~np.isnan(kp).any(axis=1)]
            if len(valid_kp) > 0:
                self.bx = [np.min(valid_kp[:, 0]), np.max(valid_kp[:, 0])]
                self.by = [np.min(valid_kp[:, 1]), np.max(valid_kp[:, 1])]
            else:
                self.bx = [0, 0]
                self.by = [0, 0]
        
        def can_reach(self, p, epsx=1.5, epsy=1.5):
            if self.bx is None or self.by is None:
                return False
            dx, dy = self.bx[1] - self.bx[0], self.by[1] - self.by[0]
            return self.bx[0] - epsx * dx < p[0] < self.bx[1] + epsx * dx and \
                   self.by[0] - epsy * dy < p[1] < self.by[1] + epsy * dy


class ShuttleState:
    RISING = "RISING"
    FALLING = "FALLING"


def moving_average(values, window_size):
    if len(values) == 0:
        return None
    window = min(window_size, len(values))
    return float(np.mean(values[-window:]))


def is_point_in_box(point, box, buffer_px=10):
    if point is None or box is None:
        return False
    x, y = point
    xmin, ymin, xmax, ymax = box
    return (x >= xmin - buffer_px and x <= xmax + buffer_px and y >= ymin - buffer_px
            and y <= ymax + buffer_px)


def is_point_near_net_keypoints(point, net_points, max_distance=15):
    """
    Check if a point is near the net's actual keypoints (more accurate than bounding box).
    
    Args:
        point: (x, y) position to check
        net_points: List of net keypoint positions
        max_distance: Maximum distance in pixels to be considered "on net"
        
    Returns:
        True if point is within max_distance of any net keypoint, False otherwise
    """
    if point is None or net_points is None or len(net_points) < 4:
        return False
    
    px, py = point
    min_distance = float('inf')
    
    # Check distance to each net keypoint
    for net_pt in net_points:
        if net_pt is None or len(net_pt) < 2:
            continue
        nx, ny = net_pt[0], net_pt[1]
        distance = np.sqrt((px - nx)**2 + (py - ny)**2)
        min_distance = min(min_distance, distance)
    
    return min_distance <= max_distance


def joints_to_bbox(joints, padding=15):
    if joints is None:
        return None
    xs = [p[0] for p in joints if p is not None]
    ys = [p[1] for p in joints if p is not None]
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def joints_to_pose(joints):
    """Convert joint list to Pose object for reachability checking."""
    if joints is None:
        return None
    # Filter out None values and ensure we have valid keypoints
    valid_joints = [j for j in joints if j is not None and len(j) >= 2]
    if len(valid_joints) < 17:
        # Pad to 17 keypoints if needed (Pose expects 17 keypoints)
        valid_joints.extend([[np.nan, np.nan]] * (17 - len(valid_joints)))
    pose = Pose()
    pose.init_from_kparray(valid_joints[:17])
    return pose


def is_point_reachable_by_pose(point, pose, epsx=1.5, epsy=1.5):
    """Check if a point is reachable by a player pose using HitDetect logic."""
    if pose is None or point is None:
        return False
    return pose.can_reach(np.array(point), epsx=epsx, epsy=epsy)


def get_player_lowest_point(joints):
    """Get the lowest y-coordinate (highest value) from player joints (feet/ankles)."""
    if joints is None:
        return None
    # Ankle indices are typically 15 (left_ankle) and 16 (right_ankle) in 17-keypoint format
    # Also check knees (13, 14) and other lower body points
    valid_ys = []
    for joint in joints:
        if joint is not None and len(joint) >= 2:
            valid_ys.append(joint[1])
    if len(valid_ys) == 0:
        return None
    # Return the maximum y (lowest on screen = closest to ground)
    return max(valid_ys)


def is_point_below_player(point, player_joints, threshold_px=50):
    """Check if point is significantly below player's feet (suggesting ground landing)."""
    if point is None or player_joints is None:
        return False
    player_lowest = get_player_lowest_point(player_joints)
    if player_lowest is None:
        return False
    # Point is below player if its y is greater (lower on screen) by threshold
    return point[1] > player_lowest + threshold_px


def has_high_downward_velocity(vy_history, threshold=15.0):
    """Check if velocity history shows high downward velocity (suggesting ground landing)."""
    if len(vy_history) == 0:
        return False
    # Check if average of positive (downward) velocities exceeds threshold
    downward_vels = [v for v in vy_history if v > 0]
    if len(downward_vels) == 0:
        return False
    avg_downward = sum(downward_vels) / len(downward_vels)
    return avg_downward > threshold


def get_distance_from_player_bbox(point, player_joints):
    """Get minimum distance from point to player's actual bounding box."""
    if point is None or player_joints is None:
        return float('inf')
    bbox = joints_to_bbox(player_joints, padding=0)
    if bbox is None:
        return float('inf')
    xmin, ymin, xmax, ymax = bbox
    px, py = point
    
    # Calculate distance to bounding box
    dx = max(xmin - px, 0, px - xmax)
    dy = max(ymin - py, 0, py - ymax)
    return np.sqrt(dx*dx + dy*dy)


def compute_homography(court_corners, model_size=(610.0, 1340.0)):
    # court_corners expected: [tl, tr, bl, br] or 6-pt where [0]=tl, [1]=tr, [4]=bl, [5]=br
    if court_corners is None:
        return None
    pts = None
    if len(court_corners) >= 6:
        pts = np.array([
            court_corners[0],  # tl
            court_corners[1],  # tr
            court_corners[4],  # bl
            court_corners[5],  # br
        ], dtype=np.float32)
    elif len(court_corners) == 4:
        pts = np.array(court_corners, dtype=np.float32)
    else:
        return None

    w, h = model_size
    dst = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts, dst)
    return H


def warp_point(point, H):
    if point is None or H is None:
        return None
    p = np.array([[point]], dtype=np.float32)
    tp = cv2.perspectiveTransform(p, H)
    return (float(tp[0, 0, 0]), float(tp[0, 0, 1]))


def in_model_bounds(point, model_size=(610.0, 1340.0)):
    if point is None:
        return False
    x, y = point
    return (x >= 0.0 and x <= model_size[0] and y >= 0.0 and y <= model_size[1])


def classify_in_out(landing_pos, court_corners, model_size=(610.0, 1340.0)):
    """
    Classify if a landing position is IN or OUT of bounds.
    
    Args:
        landing_pos: (x, y) position in image coordinates
        court_corners: Four (x, y) corners of the court
        model_size: (width, height) of the model court
        
    Returns:
        "IN" or "OUT"
    """
    if landing_pos is None or court_corners is None:
        return "OUT"
    
    H = compute_homography(court_corners, model_size)
    if H is None:
        return "OUT"
    
    top_down_pos = warp_point(landing_pos, H)
    if top_down_pos is None:
        return "OUT"
    
    return "IN" if in_model_bounds(top_down_pos, model_size) else "OUT"


def is_in_player_reach_zone(pos, player_joints_list, epsx=1.5, epsy=2.0):
    """
    Check if a position is in a player's reach zone using generous parameters.
    Used for ambiguity checking in the triage system.
    
    Args:
        pos: (x, y) position to check
        player_joints_list: List of player joint arrays
        epsx: Horizontal reachability multiplier (default: 1.5)
        epsy: Vertical reachability multiplier (default: 2.0)
        
    Returns:
        True if position is in any player's reach zone, False otherwise
    """
    if pos is None or player_joints_list is None:
        return False
    
    for joints in player_joints_list:
        if joints is None:
            continue
        pose = joints_to_pose(joints)
        if pose is not None and is_point_reachable_by_pose(pos, pose, epsx=epsx, epsy=epsy):
            return True
    
    return False


def is_in_clear_ground_zone(pos, player_joints_list, threshold_px=30):
    """
    Check if a position is in a clear ground zone (significantly below player's feet).
    Used for triage to identify clear ground landings.
    
    Args:
        pos: (x, y) position to check
        player_joints_list: List of player joint arrays
        threshold_px: Vertical threshold in pixels (default: 30)
        
    Returns:
        True if position is clearly below any player, False otherwise
    """
    if pos is None or player_joints_list is None:
        return False
    
    for joints in player_joints_list:
        if joints is not None and is_point_below_player(pos, joints, threshold_px=threshold_px):
            return True
    
    return False


class LandingDetector:
    def __init__(self, smooth_window=7, vy_window=10, model_size=(610.0, 1340.0), 
                 reach_epsx=1.5, reach_epsy=2.0, min_serve_rise_velocity=-2.0, 
                 horizontal_hit_threshold=2.5, landing_cluster_window=60):
        """
        Initialize landing detector with tunable parameters.
        
        Args:
            smooth_window: Rolling average window for y-coordinate smoothing (default: 7)
            vy_window: Window size for velocity history (default: 10)
            model_size: Model court size for homography (default: (610.0, 1340.0))
            reach_epsx: Horizontal reachability multiplier for player zone (default: 1.5, tune: 2.0-2.5)
            reach_epsy: Vertical reachability multiplier for player zone (default: 2.0, tune: 2.0-2.5)
            min_serve_rise_velocity: Minimum upward velocity to transition to RISING (default: -2.0)
            horizontal_hit_threshold: Minimum horizontal speed to confirm PLAYER_HIT (default: 2.5 px/frame)
            landing_cluster_window: Frames to look back for first landing in cluster (default: 60, ~2 seconds)
        """
        self.smooth_window = smooth_window
        self.vy_window = vy_window
        self.model_size = model_size
        self.reach_epsx = reach_epsx
        self.reach_epsy = reach_epsy
        self.min_serve_rise_velocity = min_serve_rise_velocity
        self.horizontal_hit_threshold = horizontal_hit_threshold
        self.landing_cluster_window = landing_cluster_window

        self.state = ShuttleState.RISING  # Start in RISING state
        self.y_buffer = deque(maxlen=32)
        self.x_buffer = deque(maxlen=32)  # Track x-coordinate for horizontal velocity
        self.y_smooth_prev = None
        self.x_smooth_prev = None
        self.vy_history = deque(maxlen=vy_window)
        self.vx_history = deque(maxlen=vy_window)  # Track horizontal velocity
        self.H = None
        self.potential_landings = []  # Collect all potential landing events
        self.potential_net_hits = []  # Collect all potential net hit events

    def set_homography_from_court(self, court_corners):
        self.H = compute_homography(court_corners, self.model_size)

    def reset_rally(self):
        """Reset detector for a new rally."""
        self.state = ShuttleState.RISING
        self.y_buffer.clear()
        self.x_buffer.clear()
        self.y_smooth_prev = None
        self.x_smooth_prev = None
        self.vy_history.clear()
        self.vx_history.clear()
        self.potential_landings = []  # Clear potential landings
        self.potential_net_hits = []  # Clear potential net hits

    def update(self, frame_index, shuttle_pos, player_joints_list=None, player_boxes=None, net_box=None, net_points=None, court_corners=None):
        """
        Update landing detector - Step 1: Collect all potential landing events.
        
        This method processes each frame and collects potential landing events.
        After all frames are processed, call get_true_landing() to find the final landing.
        
        Args:
            frame_index: Current frame number
            shuttle_pos: (x, y) position of shuttlecock
            player_joints_list: List of player joint arrays (preferred for accurate detection)
            player_boxes: List of bounding boxes (fallback if joints unavailable)
            net_box: Bounding box for net (fallback)
            net_points: List of net keypoints (preferred for accurate detection)
            court_corners: Court corner points for homography (not used during collection)
            
        Returns:
            None (events are collected internally, use get_true_landing() after processing all frames)
        """
        # Initialize homography if needed (for later use in get_true_landing)
        if self.H is None and court_corners is not None:
            self.set_homography_from_court(court_corners)

        # If shuttle not visible, no state update
        if shuttle_pos is None:
            return None

        # ============================================
        # STEP 1: Update Velocity and State
        # ============================================
        # Smooth y with rolling average
        self.y_buffer.append(shuttle_pos[1])
        y_smooth = moving_average(list(self.y_buffer), self.smooth_window)
        if y_smooth is None:
            return None

        # Smooth x with rolling average (for horizontal velocity tracking)
        self.x_buffer.append(shuttle_pos[0])
        x_smooth = moving_average(list(self.x_buffer), self.smooth_window)
        if x_smooth is None:
            return None

        # Compute instantaneous vertical velocity
        vy = None
        if self.y_smooth_prev is not None:
            vy = y_smooth - self.y_smooth_prev
            self.vy_history.append(vy)
        self.y_smooth_prev = y_smooth

        # Compute instantaneous horizontal velocity
        vx = None
        if self.x_smooth_prev is not None:
            vx = x_smooth - self.x_smooth_prev
            self.vx_history.append(vx)
        self.x_smooth_prev = x_smooth

        if vy is None or vx is None:
            return None

        # Update State: RISING (if vy < -2.0) or FALLING (if vy > 0)
        prev_state = self.state
        if vy < self.min_serve_rise_velocity:
            self.state = ShuttleState.RISING
        elif vy > 0:
            self.state = ShuttleState.FALLING

        # ============================================
        # STEP 2: Detect Impact and Apply Filters
        # ============================================
        # Impact event: transition vy from >0 to <=0 while in FALLING
        impact_trigger = (prev_state == ShuttleState.FALLING and vy <= 0)
        if not impact_trigger:
            return None

        impact_pos = shuttle_pos
        
        # Get context data
        player_data = player_joints_list if player_joints_list else None
        
        # Filter 1: Net Hit Filter
        # Only classify as net hit if shuttle is actually ON the net (not just near it)
        # Use distance to net keypoints for more accurate detection
        is_on_net = False
        if net_points is not None and len(net_points) >= 4:
            # Check if point is close to actual net keypoints (more accurate than bbox)
            # Use a small distance threshold (15px) to ensure shuttle is actually on net
            if is_point_near_net_keypoints(impact_pos, net_points, max_distance=15):
                # Additional check: verify shuttle is at net height, not clearly below
                net_bbox = compute_net_bbox_from_points(net_points, padding=3)
                if net_bbox is not None:
                    net_y_max = net_bbox[3]  # Bottom of net
                    # If shuttle is more than 30px below net bottom, it's a ground landing
                    if impact_pos[1] > net_y_max + 30:
                        # Shuttle is clearly below net - this is a ground landing, not net hit
                        is_on_net = False
                    else:
                        # Shuttle is near net height - this is a net hit
                        is_on_net = True
        elif net_box is not None:
            # Fallback: use net_box with strict checks
            net_y_min = net_box[1]
            net_y_max = net_box[3]
            net_x_min = net_box[0]
            net_x_max = net_box[2]
            
            is_horizontally_on_net = (net_x_min - 3 <= impact_pos[0] <= net_x_max + 3)
            
            if is_horizontally_on_net:
                # If shuttle is clearly below net bottom, it's a ground landing
                if impact_pos[1] > net_y_max + 30:
                    is_on_net = False
                elif net_y_min - 5 <= impact_pos[1] <= net_y_max + 5:
                    is_on_net = True
        
        if is_on_net:
            # Collect net hit as potential rally-ending event
            self.potential_net_hits.append({
                "frame": frame_index,
                "pos": impact_pos
            })
            return None
        
        # Filter 2: Player Hit Filter
        # If impact is inside player_reach_zone, check horizontal velocity
        in_player_reach_zone = False
        if player_data:
            in_player_reach_zone = is_in_player_reach_zone(impact_pos, player_data, 
                                                           epsx=self.reach_epsx, epsy=self.reach_epsy)
        # Fallback to bounding boxes if joints unavailable
        elif player_boxes:
            for box in player_boxes:
                if box is not None and is_point_in_box(impact_pos, box, buffer_px=15):
                    in_player_reach_zone = True
                    break
        
        if in_player_reach_zone:
            # Check horizontal velocity in last ~5 frames
            recent_vx = list(self.vx_history)[-5:] if len(self.vx_history) >= 5 else list(self.vx_history)
            if len(recent_vx) > 0:
                avg_horizontal_speed = sum(abs(vx) for vx in recent_vx) / len(recent_vx)
                if avg_horizontal_speed > self.horizontal_hit_threshold:
                    # This is a PLAYER_HIT - discard and continue
                    return None
        
        # Event passed all filters - add to potential landings
        self.potential_landings.append({
            "frame": frame_index,
            "pos": impact_pos
        })
        
        return None  # Don't return event yet, will be processed in get_true_landing()

    def get_true_landing(self, court_corners=None):
        """
        Step 2: Find the true landing from collected potential landings.
        
        After processing all frames, this method analyzes the potential_landings and
        potential_net_hits lists to find the first event in the final landing cluster.
        Net hits are prioritized since they end the rally.
        
        Args:
            court_corners: Court corner points for homography (if not already set)
            
        Returns:
            Event dict with GROUND_LANDING or NET_HIT, or None if no landing found
        """
        # Initialize homography if needed
        if self.H is None and court_corners is not None:
            self.set_homography_from_court(court_corners)
        
        # Combine all potential events (net hits and ground landings)
        all_events = []
        for net_hit in self.potential_net_hits:
            all_events.append({**net_hit, "event_type": "NET_HIT"})
        for landing in self.potential_landings:
            all_events.append({**landing, "event_type": "GROUND_LANDING"})
        
        # Sort by frame number
        all_events.sort(key=lambda x: x["frame"])
        
        # Check if empty
        if not all_events:
            return None  # No landing was detected in the video
        
        # Get last event (the very last potential event detected)
        last_event = all_events[-1]
        
        # Find first in cluster
        true_event = last_event
        for event in reversed(all_events[:-1]):
            if (last_event["frame"] - event["frame"]) < self.landing_cluster_window:
                # This event is still in the cluster, update true_event to this earlier event
                true_event = event
            else:
                # This event is too old and not part of the final landing cluster
                break
        
        # Check if the true event is a net hit (prioritize net hits)
        # Look for net hits in the final cluster
        final_cluster_events = [e for e in all_events 
                               if abs(e["frame"] - last_event["frame"]) < self.landing_cluster_window]
        net_hits_in_cluster = [e for e in final_cluster_events if e.get("event_type") == "NET_HIT"]
        
        if net_hits_in_cluster:
            # If there's a net hit in the final cluster, use the earliest one
            net_hit = min(net_hits_in_cluster, key=lambda x: x["frame"])
            return {
                "frame": net_hit["frame"],
                "pos": [int(net_hit["pos"][0]), int(net_hit["pos"][1])],
                "event_type": "NET_HIT",
                "classification": "FAULT",
                "velocity_history": list(self.vy_history),
            }
        
        # Otherwise, return the ground landing
        final_pos = true_event["pos"]
        in_out = classify_in_out(final_pos, court_corners, self.model_size) if court_corners is not None else "OUT"
        
        return {
            "frame": true_event["frame"],
            "pos": [int(final_pos[0]), int(final_pos[1])],
            "event_type": "GROUND_LANDING",
            "classification": "TRUE_POSITIVE",
            "in_out": in_out,
            "velocity_history": list(self.vy_history),
        }


def compute_net_bbox_from_points(net_points, padding=15):
    """Compute bounding box from net keypoints."""
    if net_points is None or len(net_points) == 0:
        return None
    xs = [p[0] for p in net_points if p is not None and len(p) >= 2]
    ys = [p[1] for p in net_points if p is not None and len(p) >= 2]
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(min(xs) - padding), int(min(ys) - padding), int(max(xs) + padding), int(max(ys) + padding)]


