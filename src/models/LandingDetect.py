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

# TODO: @Gavin change this method if you had a better method of getting homography
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


class LandingDetector:
    def __init__(self, smooth_window=7, vy_window=10, model_size=(610.0, 1340.0)):
        self.smooth_window = smooth_window
        self.vy_window = vy_window
        self.model_size = model_size

        self.state = ShuttleState.RISING
        self.y_buffer = deque(maxlen=32)
        self.y_smooth_prev = None
        self.vy_history = deque(maxlen=vy_window)
        self.H = None

    def set_homography_from_court(self, court_corners):
        self.H = compute_homography(court_corners, self.model_size)

    def reset_rally(self):
        self.state = ShuttleState.RISING
        self.y_buffer.clear()
        self.y_smooth_prev = None
        self.vy_history.clear()

    def update(self, frame_index, shuttle_pos, player_joints_list=None, player_boxes=None, net_box=None, net_points=None, court_corners=None):
        """
        Update landing detector with frame data.
        
        Args:
            frame_index: Current frame number
            shuttle_pos: (x, y) position of shuttlecock
            player_joints_list: List of player joint arrays (preferred for accurate detection)
            player_boxes: List of bounding boxes (fallback if joints unavailable)
            net_box: Bounding box for net (fallback)
            net_points: List of net keypoints (preferred for accurate detection)
            court_corners: Court corner points for homography
        """
        # Returns event dict when a landing/net/player hit is detected; otherwise None
        if self.H is None and court_corners is not None:
            self.set_homography_from_court(court_corners)

        # If shuttle not visible, no state update
        if shuttle_pos is None:
            return None

        # Smooth y
        self.y_buffer.append(shuttle_pos[1])
        y_smooth = moving_average(list(self.y_buffer), self.smooth_window)
        if y_smooth is None:
            return None

        # Compute vy
        vy = None
        if self.y_smooth_prev is not None:
            vy = y_smooth - self.y_smooth_prev
            self.vy_history.append(vy)
        self.y_smooth_prev = y_smooth

        if vy is None:
            return None

        # Determine state
        prev_state = self.state
        if vy < 0:
            self.state = ShuttleState.RISING
        elif vy > 0:
            self.state = ShuttleState.FALLING

        # Impact event: transition vy from >0 to <=0 while in FALLING
        impact_trigger = (prev_state == ShuttleState.FALLING and vy <= 0)
        if not impact_trigger:
            return None

        impact_pos = shuttle_pos

        # Classify against players using Pose.can_reach() (more accurate than bbox)
        if player_joints_list:
            for joints in player_joints_list:
                pose = joints_to_pose(joints)
                if pose is not None and is_point_reachable_by_pose(impact_pos, pose, epsx=1.5, epsy=1.5):
                    self.state = ShuttleState.RISING
                    return {
                        "frame": frame_index,
                        "pos": [int(impact_pos[0]), int(impact_pos[1])],
                        "event_type": "PLAYER_HIT",
                        "classification": "FALSE_POSITIVE",
                        "velocity_history": list(self.vy_history),
                    }
        # Fallback to bounding box check if joints unavailable
        elif player_boxes:
            for box in player_boxes:
                if box is not None and is_point_in_box(impact_pos, box, buffer_px=10):
                    self.state = ShuttleState.RISING
                    return {
                        "frame": frame_index,
                        "pos": [int(impact_pos[0]), int(impact_pos[1])],
                        "event_type": "PLAYER_HIT",
                        "classification": "FALSE_POSITIVE",
                        "velocity_history": list(self.vy_history),
                    }

        # Classify against net using net_points if available (more accurate)
        if net_points is not None and len(net_points) >= 4:
            # Check if point is within net polygon (using bounding box approximation)
            net_bbox = compute_net_bbox_from_points(net_points)
            if net_bbox is not None and is_point_in_box(impact_pos, net_bbox, buffer_px=15):
                self.state = ShuttleState.GROUNDED
                return {
                    "frame": frame_index,
                    "pos": [int(impact_pos[0]), int(impact_pos[1])],
                    "event_type": "NET_HIT",
                    "classification": "FAULT",
                    "velocity_history": list(self.vy_history),
                }
        # Fallback to net_box if net_points unavailable
        elif net_box is not None and is_point_in_box(impact_pos, net_box, buffer_px=10):
            self.state = ShuttleState.GROUNDED
            return {
                "frame": frame_index,
                "pos": [int(impact_pos[0]), int(impact_pos[1])],
                "event_type": "NET_HIT",
                "classification": "FAULT",
                "velocity_history": list(self.vy_history),
            }

        # Ground landing
        self.state = ShuttleState.GROUNDED
        top_down = warp_point(impact_pos, self.H)
        in_out = "IN" if in_model_bounds(top_down, self.model_size) else "OUT"
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


