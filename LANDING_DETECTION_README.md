## Landing Detection Algorithm

This module detects badminton rally-ending events by identifying impact moments and classifying true ground landings as IN/OUT using a perspective transform. The algorithm uses the **"Look-Ahead" method** to accurately differentiate between player hits (rally continues) and ground landings (rally ends) by analyzing post-impact behavior.

### Core Principle: The "Look-Ahead" Method

Instead of using conflicting heuristics to make an instant decision, this algorithm uses a **triage system**:

1. **Triage**: When an impact occurs, it's first classified:
   - **Clear Impacts** (far from players or net) are classified immediately as `GROUND_LANDING`
   - **Ambiguous Impacts** (near a player) are not classified. The algorithm enters an `AMBIGUOUS` state

2. **Look-Ahead**: For ambiguous impacts, the algorithm "waits" for 5-10 frames (configurable)

3. **Resolve**: It then checks the shuttle's behavior after the impact:
   - If the shuttle **rose**, it was a `PLAYER_HIT`
   - If the shuttle **stayed down**, it was a `GROUND_LANDING`

This method correctly handles both "fast player hits" (which look like ground landings) and "slow ground landings" (which look like player hits).

### States

- **RISING**: Shuttle is moving up ($v_y < 0$)
- **FALLING**: Shuttle is moving down ($v_y > 0$)
- **AMBIGUOUS**: An impact occurred near a player. The system is "looking ahead" to classify it
- **GROUNDED**: The rally is confirmed over (ground or net hit)

### Algorithm Pipeline (Per Frame)

The algorithm runs three steps every frame:

#### Step 1: Update Velocity and State
- **Get y_smooth**: Update the 7-frame rolling average of the shuttle's $y$-coordinate
- **Get vy**: Calculate the instantaneous vertical velocity: $v_y = y_{smooth_t} - y_{smooth_{t-1}}$
- **Update State** (if not ambiguous):
  - If `state != AMBIGUOUS`:
    - If $v_y < 0$: `state = RISING`
    - If $v_y > 0$: `state = FALLING`

#### Step 2: Check for Ambiguity Resolution (Look-Ahead)
**This logic MUST run before impact detection.**

- **Check State**: IF `state == AMBIGUOUS`:
  - **Get Time**: `frames_since_impact = current_frame - ambiguous_event_data["frame"]`
  - **Check Time**: IF `frames_since_impact > look_ahead_window` (default: 5 frames):
    - **Resolve**:
      - Get the original impact y position: `impact_y = ambiguous_event_data["pos"][1]`
      - Get the current smoothed y position: `current_y = y_smooth`
      - **IF** `current_y < (impact_y - 10)` (shuttle has risen 10+ pixels):
        - **Conclusion**: It was a `PLAYER_HIT`
        - **Action**: Set `state = RISING` and clear `ambiguous_event_data`
      - **ELSE** (shuttle stayed at same level, jiggled, or bounced weakly):
        - **Conclusion**: It was a `GROUND_LANDING`
        - **Action**: Set `state = GROUNDED`
        - **Classify In/Out**: Call `classify_in_out(ambiguous_event_data["pos"], court_corners)`
        - **Record** this event using the original frame and position

#### Step 3: Check for New Impact (The "Triage")
- **Check for Impact**: IF `state == FALLING AND vy <= 0`:
  - **Get Context**: Get `impact_pos = shuttle_pos`, `player_data`, and `net_data`
  - **Define Zones**:
    - `player_reach_zone`: Define a generous reachability model around the player(s) (epsx=1.5, epsy=2.0)
    - `clear_ground_zone`: Define a zone that is clearly on the ground (>30px below player's feet)
  - **Run Triage Logic**:
    - **IF** `impact_pos` is near `net_points`:
      - **Event**: `NET_HIT` (Rally over)
      - **Action**: Set `state = GROUNDED` and record the net hit
    - **ELSE IF** `(impact_pos is in player_reach_zone) AND (impact_pos is NOT in clear_ground_zone)`:
      - **Event**: This is an `AMBIGUOUS` impact
      - **Action**:
        - Set `state = AMBIGUOUS`
        - Store `ambiguous_event_data = {"frame": current_frame, "pos": impact_pos}`
    - **ELSE**:
      - **Event**: This is a clear `GROUND_LANDING` (far from players or clearly below them)
      - **Action**:
        - Set `state = GROUNDED`
        - **Classify In/Out**: Call `classify_in_out(impact_pos, court_corners)`
        - Record the ground landing

### Inputs

- **shuttle_pos**: (x, y) from `res/ball/loca_info(denoise)/{video}/...json`
- **player_joints_list**: List of player joint arrays from `res/players/player_kp/{video}.json` (preferred for accurate detection)
- **player_boxes**: List of bounding boxes computed from player joints (fallback if joints unavailable)
- **net_points**: Net keypoints from `res/courts/court_kp/{video}.json` via `NetDetect` (preferred for accurate detection)
- **net_box**: Bounding box from net keypoints (fallback if keypoints unavailable)
- **court_corners**: 4 corners extracted via `CourtDetect.get_homography_corners()` from the 6-point `court_info` (uses indices [0,1,4,5] for top-left, top-right, bottom-left, bottom-right)

### Algorithm Parameters
The algorithm uses several configurable parameters that can be adjusted for different detection sensitivity:

- **smooth_window**: Rolling average window for y-coordinate smoothing (default: 7)
- **vy_window**: Window size for velocity history (default: 10)
- **min_frames_between_landings**: Minimum frames between ground landings to prevent duplicates (default: 60 frames ≈ 2 seconds at 30 fps)
- **look_ahead_window**: Number of frames to wait before resolving ambiguous events (default: 5, configurable 5-10)
- **Player reachability (for ambiguity checking)**: epsx=1.5, epsy=2.0 (generous parameters to catch all potential player hits)
- **Clear ground zone threshold**: 30px below player's feet to classify as clear ground landing
- **Rise detection threshold**: 10 pixels (if shuttle rises 10+ pixels after ambiguous impact, classify as PLAYER_HIT)

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
  - Can be classified immediately (clear impacts) or after look-ahead resolution (ambiguous impacts)
- **NET_HIT**: Shuttle hit the net (fault) - classified immediately when impact is near net
- **PLAYER_HIT**: False positive - impact detected but shuttle was hit by player (filtered out)
  - Only classified after look-ahead resolution confirms shuttle rose after impact

### Helper Functions

- **`classify_in_out(landing_pos, court_corners)`**:
  - Defines a model court and calculates homography matrix from court corners
  - Transforms landing position to top-down view and checks if inside court boundaries
  - Returns "IN" or "OUT"

- **`is_in_player_reach_zone(pos, player_joints_list, epsx=1.5, epsy=2.0)`**:
  - Checks if position is in any player's reach zone using generous parameters
  - Used for ambiguity checking in the triage system

- **`is_in_clear_ground_zone(pos, player_joints_list, threshold_px=30)`**:
  - Checks if position is significantly below player's feet (clear ground zone)
  - Used to identify clear ground landings that don't need look-ahead

### Key Improvements: The "Look-Ahead" Method

The algorithm has been completely redesigned using the "Look-Ahead" method to accurately handle ambiguous situations:

1. **Triage System**: Impacts are first classified as clear or ambiguous:
   - Clear impacts (far from players or clearly below them) are classified immediately
   - Ambiguous impacts (near players) enter a waiting state

2. **Look-Ahead Resolution**: For ambiguous impacts, the algorithm observes post-impact behavior:
   - Waits 5-10 frames (configurable) before making a decision
   - Checks if shuttle rose (PLAYER_HIT) or stayed down (GROUND_LANDING)
   - Uses original impact frame and position for accurate event recording

3. **Handles Edge Cases**:
   - **Fast player hits**: Correctly identifies player hits that initially look like ground landings
   - **Slow ground landings**: Correctly identifies ground landings that initially look like player hits
   - **Net hits**: Immediately classifies net hits without ambiguity

4. **Generous Ambiguity Detection**: Uses generous reachability parameters (epsx=1.5, epsy=2.0) to catch all potential ambiguous cases, then resolves them accurately

5. **Duplicate filtering**: Prevents multiple detections of the same landing event

See `USAGE_LANDING_ANNOTATION.md` for how to run and visualize on video with a timeline.
