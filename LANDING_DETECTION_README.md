# Landing Detection Algorithm

This module detects badminton rally-ending events by identifying impact moments and classifying true ground landings as IN/OUT using a perspective transform. The algorithm uses a **two-step approach** to accurately differentiate between player hits (rally continues) and ground landings (rally ends).

## Core Principle: Two-Step Collection and Post-Processing

The algorithm operates in two distinct phases:

1. **Step 1: Collect All Potential Landing Events** (Frame-by-frame processing):
   - Processes each frame to update velocities and state
   - Detects all potential impact events
   - Applies filters (Net Hit Filter, Player Hit Filter) to classify events
   - Collects remaining events in `potential_landings` and `potential_net_hits` lists

2. **Step 2: Find the True Landing** (Post-processing):
   - After all frames are processed, analyzes collected events
   - Identifies the last potential landing event
   - Finds the first event within a cluster window (default: 60 frames)
   - Prioritizes `NET_HIT` events if present in the final cluster
   - Classifies the true landing as IN/OUT

This approach ensures only one definitive rally-ending event per video, eliminating false positives.

### States

- **RISING**: Shuttle is moving up ($v_y < min_serve_rise_velocity$, default: -2.0)
- **FALLING**: Shuttle is moving down ($v_y > 0$)

### Algorithm Pipeline

#### Step 1: Collect All Potential Landing Events (Per Frame)

For each frame, the algorithm:

1. **Update Velocity and State**:
   - **Get y_smooth**: Update the rolling average of the shuttle's $y$-coordinate using `smooth_window` (default: 7)
   - **Get x_smooth**: Update the rolling average of the shuttle's $x$-coordinate for horizontal velocity
   - **Get vy**: Calculate the instantaneous vertical velocity: $v_y = y_{smooth_t} - y_{smooth_{t-1}}$
   - **Get vx**: Calculate the instantaneous horizontal velocity: $v_x = x_{smooth_t} - x_{smooth_{t-1}}$
   - **Update State**:
     - If $v_y < min_serve_rise_velocity$: `state = RISING` (significant upward motion)
     - If $v_y > 0$: `state = FALLING` (downward motion)

2. **Detect Impact and Apply Filters**:
   - **Impact Trigger**: `prev_state == FALLING AND vy <= 0`
   - **Filter 1: Net Hit Filter**:
     - Checks if impact position is within 15px of net keypoints
     - Verifies shuttle is at net height (not more than 30px below net bottom)
     - If net hit detected, adds to `potential_net_hits` list
   - **Filter 2: Player Hit Filter**:
     - Checks if impact is in player reach zone
     - Calculates average horizontal velocity from `vx_history`
     - If `avg_horizontal_speed > horizontal_hit_threshold` (default: 2.5), discards as `PLAYER_HIT`
   - **If event passes all filters**: Adds to `potential_landings` list

#### Step 2: Find the True Landing (Post-Processing)

After all frames are processed, `get_true_landing()` is called:

1. **Combine Events**: Merges `potential_net_hits` and `potential_landings`, sorted by frame
2. **Find Last Event**: Identifies the last potential landing event
3. **Cluster Analysis**: Iterates backward from last event to find first event within `landing_cluster_window` (default: 60 frames)
4. **Net Hit Prioritization**: If any `NET_HIT` events exist in the final cluster, returns the earliest one
5. **Classify Ground Landing**: Otherwise, classifies the true landing event as `GROUND_LANDING` using `classify_in_out()`

### Inputs

- **shuttle_pos**: (x, y) from `res/ball/loca_info(denoise)/{video}/...json`
- **player_joints_list**: List of player joint arrays from `res/players/player_kp/{video}.json` (preferred for accurate detection)
- **player_boxes**: List of bounding boxes computed from player joints (fallback if joints unavailable)
- **net_points**: Net keypoints from `res/courts/court_kp/{video}.json` via `NetDetect` (preferred for accurate detection)
- **net_box**: Bounding box from net keypoints (fallback if keypoints unavailable)
- **court_corners**: 4 corners extracted via `CourtDetect.get_homography_corners()` from the 6-point `court_info` (uses indices [0,1,4,5] for top-left, top-right, bottom-left, bottom-right)

## Algorithm Parameters

The algorithm uses several configurable parameters that can be adjusted for different detection sensitivity:

- **smooth_window**: Rolling average window for y-coordinate and x-coordinate smoothing (default: 7)
- **vy_window**: Window size for velocity history (default: 10)
- **min_serve_rise_velocity**: Minimum upward velocity to transition to RISING state (default: -2.0)
  - Prevents false detections before the serve/rally starts
- **horizontal_hit_threshold**: Threshold for average horizontal velocity to classify as player hit (default: 2.5)
  - Used to filter out player hits that don't show significant vertical rise
- **landing_cluster_window**: Window size in frames for finding true landing from cluster (default: 60)
  - Used in post-processing to identify the first event in a cluster of potential landings
- **Player reachability (for player hit filtering)**: reach_epsx=1.5, reach_epsy=2.0
- **Net detection**:
  - Distance threshold: 15px from net keypoints
  - Vertical check: Shuttle must be within 30px below net bottom to be considered net hit

### Outputs

`res/landings/{video}_landings.json`

```json
{
  "video_name": "test1",
  "total_rallies": 1,
  "total_landings": 3,
  "landings": [
    {
      "frame": 362,
      "pos": [797, 345],
      "event_type": "GROUND_LANDING",
      "classification": "TRUE_POSITIVE",
      "in_out": "OUT",
      "velocity_history": [14.14, 16.71, 18.57, ...]
    },
    {
      "frame": 445,
      "pos": [594, 667],
      "event_type": "GROUND_LANDING",
      "classification": "TRUE_POSITIVE",
      "in_out": "IN",
      "velocity_history": [68.71, 76.86, 80.57, ...]
    }
  ]
}
```

### Event Types

- **GROUND_LANDING**: True positive ground landing with `in_out` classification ("IN" or "OUT")
  - Detected after post-processing identifies the true landing from collected potential events
- **NET_HIT**: Shuttle hit the net (fault) - detected when impact is within 15px of net keypoints and at net height
  - Prioritized in post-processing if present in the final landing cluster
- **PLAYER_HIT**: Filtered out during collection phase
  - Detected using horizontal velocity threshold (avg_horizontal_speed > 2.5)
  - Events are discarded and not added to potential_landings list

### Helper Functions

- **`classify_in_out(landing_pos, court_corners)`**:
  - Defines a model court and calculates homography matrix from court corners
  - Transforms landing position to top-down view and checks if inside court boundaries
  - Returns "IN" or "OUT"

- **`is_in_player_reach_zone(pos, player_joints_list, reach_epsx=1.5, reach_epsy=2.0)`**:
  - Checks if position is in any player's reach zone
  - Used in Player Hit Filter to identify potential player hits

- **`is_point_near_net_keypoints(point, net_points, max_distance=15)`**:
  - Checks if a point is within max_distance pixels of any net keypoint
  - Returns True if point is close to net, False otherwise
  - Used in Net Hit Filter for accurate net hit detection

- **`compute_net_bbox_from_points(net_points, padding=3)`**:
  - Computes bounding box from net keypoints with minimal padding
  - Used to determine net height for vertical position checking

### Key Improvements: Two-Step Collection and Post-Processing

The algorithm has been redesigned using a two-step approach for highly accurate, single-event rally-ending detection:

1. **Collection Phase**: All potential events are collected during frame processing:
   - Net hits are identified using distance to net keypoints (15px threshold)
   - Player hits are filtered using horizontal velocity analysis
   - Remaining events are stored for post-processing

2. **Post-Processing Phase**: True landing is determined from collected events:
   - Analyzes clusters of potential landings (60-frame window)
   - Prioritizes net hits if present in the final cluster
   - Ensures only one definitive rally-ending event per video

3. **Handles Edge Cases**:
   - **Net hits near ground landings**: Net hits are prioritized in post-processing
   - **Player hits with low vertical rise**: Detected using horizontal velocity threshold
   - **Multiple potential landings**: Cluster analysis finds the true landing

4. **Horizontal Velocity Analysis**: Uses `vx` (horizontal velocity) to detect player hits that don't show significant vertical rise, improving accuracy for defensive shots

5. **Single-Event Detection**: Post-processing ensures only one rally-ending event is returned, eliminating false positives and duplicate detections

See `USAGE_LANDING_ANNOTATION.md` for how to run and visualize on video with a timeline.
