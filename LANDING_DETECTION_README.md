## Landing Detection Algorithm

This module detects badminton rally-ending events by identifying impact moments and classifying true ground landings as IN/OUT using a perspective transform.

### States
- RISING: shuttle vertical velocity < 0
- FALLING: shuttle vertical velocity > 0
- GROUNDED: rally ended (ground or net)

### Pipeline (per frame)
1. Smooth shuttle y with a rolling average (window 5–10).
2. Compute smoothed vertical velocity vy.
3. Detect impact when previously FALLING and now vy <= 0.
4. False-positive filtering:
   - If impact inside any player bbox → PLAYER_HIT (false positive), continue.
   - Else if inside net bbox → NET_HIT (fault), end rally.
   - Else → GROUND_LANDING (true positive), end rally.
5. IN/OUT via homography: warp impact point to a top-down court model and check bounds.

### Inputs
- shuttle_pos: (x, y) from `res/ball/loca_info(denoise)/{video}/...json`
- player_boxes: computed from `res/players/player_kp/{video}.json` joints
- net_box: bbox from `res/courts/court_kp/{video}.json` via `NetDetect`
- court_corners: 4 corners extracted via `CourtDetect.get_homography_corners()` from the 6-point `court_info` (uses indices [0,1,4,5] for top-left, top-right, bottom-left, bottom-right)

### Outputs
`res/landings/{video}_landings.json`
```
{
  "video_name": "test7",
  "total_rallies": 1,
  "total_landings": 3,
  "landings": [ { ... events ... } ]
}
```

See `USAGE_LANDING_ANNOTATION.md` for how to run and visualize on video with a timeline.


