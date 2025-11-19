import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.append("src/tools")
sys.path.append("src/models")

from utils import read_json, write_json, find_reference
from LandingDetect import LandingDetector, joints_to_bbox
from CourtDetect import CourtDetect
from NetDetect import NetDetect


def load_ball_dict(result_path, video_name):
    ball_dict = {}
    root_dir = f"{result_path}/ball/loca_info(denoise)/{video_name}"
    if not os.path.exists(root_dir):
        return ball_dict
    for res_root, _, res_files in os.walk(root_dir):
        for res_file in res_files:
            if res_file.lower().endswith('.json'):
                res_json_path = os.path.join(res_root, res_file)
                ball_dict.update(read_json(res_json_path))
    return ball_dict


def compute_net_bbox(net_points, padding=10):
    if net_points is None:
        return None
    xs = [p[0] for p in net_points]
    ys = [p[1] for p in net_points]
    return [int(min(xs) - padding), int(min(ys) - padding), int(max(xs) + padding), int(max(ys) + padding)]


def draw_landing_marker(frame, landing_event):
    """
    Draw landing marker and IN/OUT call on the frame.
    
    Args:
        frame: OpenCV frame to draw on
        landing_event: Event dict with 'pos', 'event_type', and 'in_out' keys
    """
    if landing_event is None:
        return frame
    
    pos = landing_event.get("pos")
    if pos is None:
        return frame
    
    x, y = int(pos[0]), int(pos[1])
    event_type = landing_event.get("event_type", "")
    in_out = landing_event.get("in_out", "")
    
    # Determine colors based on IN/OUT
    if event_type == "GROUND_LANDING":
        if in_out == "IN":
            circle_color = (0, 255, 0)  # Green for IN
            text_color = (0, 255, 0)
            call_text = "IN"
        else:
            circle_color = (0, 0, 255)  # Red for OUT
            text_color = (0, 0, 255)
            call_text = "OUT"
    elif event_type == "NET_HIT":
        circle_color = (0, 165, 255)  # Orange for net hit
        text_color = (0, 165, 255)
        call_text = "NET"
    else:
        return frame
    
    # Draw text label at landing position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(call_text, font, font_scale, thickness)
    
    # Position text above the marker
    text_x = x - text_width // 2
    text_y = y - 30
    
    # Draw text background for better visibility
    cv2.rectangle(frame, 
                  (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + baseline + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, call_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return frame


def draw_timeline(frame, current_frame, total_frames, events, height=16):
    h, w, _ = frame.shape
    bar_y0 = h - height
    overlay = frame.copy()
    # background bar
    cv2.rectangle(overlay, (0, bar_y0), (w, h), (32, 32, 32), -1)
    # draw event markers
    for ev in events:
        f = ev["frame"]
        x = int((f / max(1, total_frames - 1)) * (w - 1))
        color = (128, 128, 128)
        if ev["event_type"] == "GROUND_LANDING":
            color = (0, 200, 0) if ev.get("in_out") == "IN" else (0, 0, 200)
        elif ev["event_type"] == "NET_HIT":
            color = (0, 165, 255)
        # vertical tick
        cv2.line(overlay, (x, bar_y0), (x, h), color, 2)
    # draw current cursor
    cx = int((current_frame / max(1, total_frames - 1)) * (w - 1))
    cv2.line(overlay, (cx, bar_y0), (cx, h), (255, 255, 255), 1)
    # blend
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Landing detection and annotation')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file_path', type=str, help='Single video path')
    input_group.add_argument('--folder_path', type=str, help='Folder with videos')
    parser.add_argument('--result_path', type=str, default='res', help='Results root path')
    parser.add_argument('--force', action='store_true', default=False, help='Overwrite outputs')
    args = parser.parse_args()

    videos = []
    if args.file_path:
        if os.path.isfile(args.file_path) and args.file_path.lower().endswith('.mp4'):
            videos.append(args.file_path)
        else:
            print('Invalid --file_path provided')
            sys.exit(1)
    else:
        for root, _, files in os.walk(args.folder_path):
            for f in files:
                if f.lower().endswith('.mp4'):
                    videos.append(os.path.join(root, f))

    for video_path in videos:
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]

        # Outputs
        out_dir = os.path.join(args.result_path, 'videos', video_name)
        os.makedirs(out_dir, exist_ok=True)
        out_video_path = os.path.join(out_dir, f"{video_name}_landing.mp4")
        out_json_path = os.path.join(args.result_path, 'landings', f"{video_name}_landings.json")
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

        if os.path.exists(out_video_path) and os.path.exists(out_json_path) and not args.force:
            print(f"Skipping {video_name}: outputs exist. Use --force to overwrite.")
            continue

        # Inputs
        ball_dict = load_ball_dict(args.result_path, video_name)
        players_ref = find_reference(video_name, os.path.join(args.result_path, 'players', 'player_kp'))
        courts_ref = find_reference(video_name, os.path.join(args.result_path, 'courts', 'court_kp'))

        if players_ref is None or courts_ref is None:
            print(f"Missing references for {video_name}. Skipping.")
            continue

        players_dict = read_json(players_ref)
        court_ref = read_json(courts_ref)
        court_info = court_ref.get('court_info')

        # Net points via NetDetect (pre_process reads reference and primes data)
        court_detect = CourtDetect()
        net_detect = NetDetect()
        _ = court_detect.pre_process(video_path, courts_ref)
        _ = net_detect.pre_process(video_path, courts_ref)
        net_points = net_detect.normal_net_info
        net_box = compute_net_bbox(net_points)

        # Extract homography corners using CourtDetect helper
        court_corners = court_detect.get_homography_corners(court_info)
        if court_corners is None:
            print(f"Failed to extract court corners for {video_name}. Skipping.")
            continue

        # Video IO
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        detector = LandingDetector()
        detector.set_homography_from_court(court_corners)

        events = []

        # Step 1: Collect all potential landing events by processing all frames
        with tqdm(total=total_frames, desc="Collecting potential landings") as pbar:
            while True:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break

                # Gather per-frame inputs
                shuttle_pos = None
                if str(current_frame) in ball_dict:
                    b = ball_dict[str(current_frame)]
                    if b.get('visible', 0) == 1:
                        shuttle_pos = (int(b['x']), int(b['y']))

                joints = players_dict.get(str(current_frame))
                player_joints_list = []
                player_boxes = []
                if joints is not None:
                    top_joints = joints.get('top')
                    bottom_joints = joints.get('bottom')
                    # Prefer joints for accurate Pose.can_reach() detection
                    if top_joints is not None:
                        player_joints_list.append(top_joints)
                    if bottom_joints is not None:
                        player_joints_list.append(bottom_joints)
                    # Also keep boxes as fallback
                    player_boxes.append(joints_to_bbox(top_joints))
                    player_boxes.append(joints_to_bbox(bottom_joints))

                # Get current net points if available (for more accurate net detection)
                current_net_points = None
                # Try to get net info from current frame if NetDetect supports it
                # For now, use the reference net_points as an approximation
                if net_points is not None:
                    current_net_points = net_points

                # Process frame and collect potential landings (returns None during collection)
                detector.update(
                    current_frame, 
                    shuttle_pos, 
                    player_joints_list=player_joints_list if player_joints_list else None,
                    player_boxes=player_boxes if player_boxes else None,
                    net_box=net_box,
                    net_points=current_net_points,
                    court_corners=court_corners
                )

                pbar.update(1)

        # Step 2: Find the true landing from collected potential landings
        true_landing = detector.get_true_landing(court_corners)
        if true_landing is not None:
            events.append(true_landing)

        # Step 3: Re-process video to annotate the true landing
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        with tqdm(total=total_frames, desc="Annotating video") as pbar:
            while True:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw landing marker if this is near the landing frame (show for 10 frames around landing)
                if true_landing is not None:
                    landing_frame = true_landing["frame"]
                    if abs(current_frame - landing_frame) <= 10:
                        frame = draw_landing_marker(frame, true_landing)

                # Draw timeline
                frame = draw_timeline(frame, current_frame, total_frames, events, height=16)
                writer.write(frame)
                pbar.update(1)

        cap.release()
        writer.release()

        # Persist JSON
        payload = {
            "video_name": video_name,
            "total_rallies": 1,  # placeholder if not splitting rallies yet
            "total_landings": sum(1 for e in events if e["event_type"] == "GROUND_LANDING"),
            "landings": events,
        }
        with open(out_json_path, 'w') as f:
            import json
            json.dump(payload, f, indent=4)


if __name__ == '__main__':
    main()


