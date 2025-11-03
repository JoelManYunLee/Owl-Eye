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

        with tqdm(total=total_frames) as pbar:
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

                # Pass player joints (preferred) and net points (preferred) for accurate detection
                ev = detector.update(
                    current_frame, 
                    shuttle_pos, 
                    player_joints_list=player_joints_list if player_joints_list else None,
                    player_boxes=player_boxes if player_boxes else None,
                    net_box=net_box,
                    net_points=current_net_points,
                    court_corners=court_corners
                )
                if ev is not None:
                    events.append(ev)
                    # Visual mark on frame where event occurs
                    cv2.circle(frame, (ev['pos'][0], ev['pos'][1]), 10,
                               (0, 200, 0) if ev['event_type'] == 'GROUND_LANDING' and ev.get('in_out') == 'IN'
                               else (0, 0, 200) if ev['event_type'] == 'GROUND_LANDING'
                               else (0, 165, 255) if ev['event_type'] == 'NET_HIT'
                               else (200, 200, 0), -1)

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


