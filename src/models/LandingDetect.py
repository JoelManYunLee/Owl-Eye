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
    AMBIGUOUS = "AMBIGUOUS"
    GROUNDED = "GROUNDED"


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
    def __init__(self, smooth_window=7, vy_window=10, model_size=(610.0, 1340.0), min_frames_between_landings=60, look_ahead_window=5):
        self.smooth_window = smooth_window
        self.vy_window = vy_window
        self.model_size = model_size
        self.min_frames_between_landings = min_frames_between_landings
        self.look_ahead_window = look_ahead_window  # Configurable look-ahead window for ambiguous events

        self.state = ShuttleState.RISING
        self.y_buffer = deque(maxlen=32)
        self.y_smooth_prev = None
        self.vy_history = deque(maxlen=vy_window)
        self.H = None
        self.last_landing_frame = -1  # Track last ground landing frame to prevent duplicates
        self.ambiguous_event_data = None  # Store ambiguous event: {"frame": frame_num, "pos": (x,y)}

    def set_homography_from_court(self, court_corners):
        self.H = compute_homography(court_corners, self.model_size)

    def reset_rally(self):
        self.state = ShuttleState.RISING
        self.y_buffer.clear()
        self.y_smooth_prev = None
        self.vy_history.clear()
        self.last_landing_frame = -1  # Reset landing frame tracking
        self.ambiguous_event_data = None  # Clear ambiguous event data

    def update(self, frame_index, shuttle_pos, player_joints_list=None, player_boxes=None, net_box=None, net_points=None, court_corners=None):
        """
        Update landing detector with frame data using the "Look-Ahead" method.
        
        Algorithm Pipeline:
        1. Update Velocity and State (if not ambiguous)
        2. Check for Ambiguity Resolution (Look-Ahead) - runs BEFORE impact detection
        3. Check for New Impact (The "Triage")
        
        Args:
            frame_index: Current frame number
            shuttle_pos: (x, y) position of shuttlecock
            player_joints_list: List of player joint arrays (preferred for accurate detection)
            player_boxes: List of bounding boxes (fallback if joints unavailable)
            net_box: Bounding box for net (fallback)
            net_points: List of net keypoints (preferred for accurate detection)
            court_corners: Court corner points for homography
            
        Returns:
            Event dict when a landing/net/player hit is detected; otherwise None
        """
        # Initialize homography if needed
        if self.H is None and court_corners is not None:
            self.set_homography_from_court(court_corners)

        # If shuttle not visible, no state update
        if shuttle_pos is None:
            return None

        # ============================================
        # STEP 1: Update Velocity and State
        # ============================================
        # Smooth y with 7-frame rolling average
        self.y_buffer.append(shuttle_pos[1])
        y_smooth = moving_average(list(self.y_buffer), self.smooth_window)
        if y_smooth is None:
            return None

        # Compute instantaneous vertical velocity
        vy = None
        if self.y_smooth_prev is not None:
            vy = y_smooth - self.y_smooth_prev
            self.vy_history.append(vy)
        self.y_smooth_prev = y_smooth

        if vy is None:
            return None

        # Update State (if not ambiguous)
        prev_state = self.state
        if self.state != ShuttleState.AMBIGUOUS:
            if vy < 0:
                self.state = ShuttleState.RISING
            elif vy > 0:
                self.state = ShuttleState.FALLING

        # ============================================
        # STEP 2: Check for Ambiguity Resolution (Look-Ahead)
        # This MUST run before impact detection
        # ============================================
        if self.state == ShuttleState.AMBIGUOUS and self.ambiguous_event_data is not None:
            frames_since_impact = frame_index - self.ambiguous_event_data["frame"]
            
            # Check if look-ahead window has passed
            if frames_since_impact > self.look_ahead_window:
                # Resolve the ambiguous event
                impact_y = self.ambiguous_event_data["pos"][1]
                current_y = y_smooth
                
                # If shuttle has risen significantly (10+ pixels), it was a PLAYER_HIT
                if current_y < (impact_y - 10):
                    # It was a PLAYER_HIT
                    self.state = ShuttleState.RISING
                    original_frame = self.ambiguous_event_data["frame"]
                    original_pos = self.ambiguous_event_data["pos"]
                    self.ambiguous_event_data = None  # Clear ambiguous event
                    return {
                        "frame": original_frame,
                        "pos": [int(original_pos[0]), int(original_pos[1])],
                        "event_type": "PLAYER_HIT",
                        "classification": "FALSE_POSITIVE",
                        "velocity_history": list(self.vy_history),
                    }
                else:
                    # Shuttle stayed at same level, jiggled, or bounced weakly - GROUND_LANDING
                    self.state = ShuttleState.GROUNDED
                    original_frame = self.ambiguous_event_data["frame"]
                    original_pos = self.ambiguous_event_data["pos"]
                    self.ambiguous_event_data = None  # Clear ambiguous event
                    
                    # Check for duplicate landing
                    if self.last_landing_frame >= 0 and original_frame - self.last_landing_frame < self.min_frames_between_landings:
                        return None
                    
                    self.last_landing_frame = original_frame
                    in_out = classify_in_out(original_pos, court_corners, self.model_size)
                    return {
                        "frame": original_frame,
                        "pos": [int(original_pos[0]), int(original_pos[1])],
                        "event_type": "GROUND_LANDING",
                        "classification": "TRUE_POSITIVE",
                        "in_out": in_out,
                        "velocity_history": list(self.vy_history),
                    }

        # ============================================
        # STEP 3: Check for New Impact (The "Triage")
        # ============================================
        # Impact event: transition vy from >0 to <=0 while in FALLING
        impact_trigger = (prev_state == ShuttleState.FALLING and vy <= 0)
        if not impact_trigger:
            return None

        impact_pos = shuttle_pos
        
        # Get context data
        player_data = player_joints_list if player_joints_list else None
        net_data = net_points if net_points is not None else net_box
        
        # Define zones for triage
        in_player_reach_zone = False
        in_clear_ground_zone = False
        
        if player_data:
            # Check if impact is in player reach zone (generous parameters for ambiguity checking)
            in_player_reach_zone = is_in_player_reach_zone(impact_pos, player_data, epsx=1.5, epsy=2.0)
            # Check if impact is in clear ground zone (>30px below player's feet)
            in_clear_ground_zone = is_in_clear_ground_zone(impact_pos, player_data, threshold_px=30)
        
        # Fallback to bounding boxes if joints unavailable
        if not player_data and player_boxes:
            for box in player_boxes:
                if box is not None and is_point_in_box(impact_pos, box, buffer_px=15):
                    in_player_reach_zone = True
                    break
        
        # Check if impact is near net
        is_near_net = False
        if net_points is not None and len(net_points) >= 4:
            net_bbox = compute_net_bbox_from_points(net_points)
            if net_bbox is not None and is_point_in_box(impact_pos, net_bbox, buffer_px=10):
                is_near_net = True
        elif net_box is not None and is_point_in_box(impact_pos, net_box, buffer_px=8):
            is_near_net = True
        
        # Run Triage Logic
        if is_near_net:
            # NET_HIT (Rally over)
            self.state = ShuttleState.GROUNDED
            return {
                "frame": frame_index,
                "pos": [int(impact_pos[0]), int(impact_pos[1])],
                "event_type": "NET_HIT",
                "classification": "FAULT",
                "velocity_history": list(self.vy_history),
            }
        elif in_player_reach_zone and not in_clear_ground_zone:
            # AMBIGUOUS impact - enter ambiguous state and look ahead
            self.state = ShuttleState.AMBIGUOUS
            self.ambiguous_event_data = {
                "frame": frame_index,
                "pos": impact_pos
            }
            return None  # Don't return event yet, wait for resolution
        else:
            # Clear GROUND_LANDING (far from players or clearly below them)
            # Check for duplicate landing
            if self.last_landing_frame >= 0 and frame_index - self.last_landing_frame < self.min_frames_between_landings:
                return None
            
            self.state = ShuttleState.GROUNDED
            self.last_landing_frame = frame_index
            in_out = classify_in_out(impact_pos, court_corners, self.model_size)
            return {
                "frame": frame_index,
                "pos": [int(impact_pos[0]), int(impact_pos[1])],
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


