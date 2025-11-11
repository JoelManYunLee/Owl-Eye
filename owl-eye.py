import cv2
import numpy as np
from collections import deque
from datetime import datetime
import threading
import os
import sys
import subprocess
import logging
from pathlib import Path

# Import your existing detection modules
try:
    from src.tools.utils import write_json, clear_file, is_video_detect, find_next, find_reference
    from src.tools.VideoClip import VideoClip
    from src.models.PoseDetect import PoseDetect
    from src.models.CourtDetect import CourtDetect
    from src.models.NetDetect import NetDetect
    from src.tools.BallDetect import ball_detect
    from tqdm import tqdm
    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import processing modules: {e}")
    print("Video capture will work, but automatic processing will be disabled.")
    PROCESSING_AVAILABLE = False


class BadmintonChallengeSystem:
    def __init__(self, camera_index=0, buffer_seconds=10, fps=30, result_path="res"):
        """
        Initialize badminton challenge system with video capture and processing.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            buffer_seconds: Number of seconds to keep in buffer
            fps: Target frames per second
            result_path: Path for processing results
        """
        self.camera_index = camera_index
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.result_path = result_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.actual_fps} FPS")
        
        # Rolling buffer to store frames
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # Control flags
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        
        # Create output directories
        self.output_dir = "videos"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)
        
        # Challenge counter
        self.challenge_count = 0
        
    def start(self):
        """Start capturing frames in a separate thread."""
        if self.running:
            print("Capture already running")
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("Video capture started")
        
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now()
                self.frame_buffer.append((frame.copy(), timestamp))
            else:
                print("Failed to read frame")
                break
                
    def save_buffer(self, filename=None):
        """
        Save current buffer to video file.
        
        Args:
            filename: Optional custom filename. If None, uses timestamp.
            
        Returns:
            filepath: Path to saved video file
        """
        if len(self.frame_buffer) == 0:
            print("Buffer is empty, nothing to save")
            return None
            
        # Create a snapshot of the buffer to avoid race condition
        buffer_snapshot = list(self.frame_buffer)
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"challenge_{timestamp}.mp4"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.actual_fps, 
                             (self.frame_width, self.frame_height))
        
        # Write all frames from buffer snapshot
        frame_count = 0
        for frame, timestamp in buffer_snapshot:
            out.write(frame)
            frame_count += 1
            
        out.release()
        
        duration = frame_count / self.actual_fps
        print(f"Saved {frame_count} frames ({duration:.2f} seconds) to {filepath}")
        return filepath
    
    def process_video(self, video_path):
        """
        Process saved video through the badminton detection pipeline.
        This is adapted from your main.py logic.
        
        Args:
            video_path: Path to the video file to process
        """
        if not PROCESSING_AVAILABLE:
            print("Processing modules not available. Skipping processing.")
            return
            
        print(f"\n{'='*50}")
        print(f"Processing challenge video: {video_path}")
        print(f"{'='*50}\n")
        
        try:
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Clear previous results if they exist
            if is_video_detect(video_name):
                clear_file(video_name)
            
            full_video_path = os.path.join(f"{self.result_path}/videos", video_name)
            if not os.path.exists(full_video_path):
                os.makedirs(full_video_path)
            
            # Open the video file
            video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            fps = video.get(cv2.CAP_PROP_FPS)
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Write video information
            video_dict = {
                "video_name": video_name,
                "fps": fps,
                "height": height,
                "width": width,
                "total_frames": total_frames
            }
            write_json(video_dict, video_name, full_video_path)
            
            # Initialize detection models
            pose_detect = PoseDetect()
            court_detect = CourtDetect()
            net_detect = NetDetect()
            video_clip = VideoClip(video_name, fps, total_frames, width,
                                   height, full_video_path)
            
            reference_path = find_reference(video_name)
            if reference_path is None:
                print("No reference frame found. Searching automatically...")
            else:
                print(f"Reference frame: {reference_path}")
            
            # Pre-process court and net detection
            begin_frame = court_detect.pre_process(video_path, reference_path)
            _ = net_detect.pre_process(video_path, reference_path)
            
            next_frame = find_next(video_path, court_detect, begin_frame)
            first_frame = next_frame
            
            normal_court_info = court_detect.normal_court_info
            normal_net_info = net_detect.normal_net_info
            
            # Correct net position
            if normal_net_info is not None and normal_court_info is not None:
                normal_net_info[1][1], normal_net_info[2][1] = \
                    normal_court_info[2][1], normal_court_info[3][1]
            
            court_dict = {
                "first_rally_frame": first_frame,
                "next_rally_frame": next_frame,
                "court_info": normal_court_info,
                "net_info": normal_net_info,
            }
            
            write_json(court_dict, video_name,
                      f"{self.result_path}/courts/court_kp", "w")
            
            # Process each frame
            print("Processing frames...")
            with tqdm(total=total_frames) as pbar:
                while True:
                    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = video.read()
                    
                    if not ret:
                        break
                    
                    have_court = False
                    players_dict = {
                        str(current_frame): {
                            "top": None,
                            "bottom": None
                        }
                    }
                    have_court_dict = {str(current_frame): have_court}
                    
                    if current_frame < next_frame:
                        write_json(have_court_dict, video_name,
                                  f"{self.result_path}/courts/have_court")
                        write_json(players_dict, video_name,
                                  f"{self.result_path}/players/player_kp")
                        court_mse_dict = {str(current_frame): court_detect.mse}
                        write_json(court_mse_dict, video_name,
                                  f"{self.result_path}/courts/court_mse")
                        video_made = video_clip.add_frame(have_court, frame, current_frame)
                        pbar.update(1)
                        continue
                    
                    # Court and player detection
                    court_info, have_court = court_detect.get_court_info(frame)
                    if have_court:
                        original_outputs, human_joints = pose_detect.get_human_joints(frame)
                        have_player, players_joints = court_detect.player_detection(original_outputs)
                        
                        if have_player:
                            players_dict = {
                                str(current_frame): {
                                    "top": players_joints[0],
                                    "bottom": players_joints[1]
                                }
                            }
                    
                    video_made = video_clip.add_frame(have_court, frame, current_frame)
                    if video_made:
                        next_frame = find_next(video_path, court_detect, current_frame)
                        court_dict = {
                            "first_rally_frame": first_frame,
                            "next_rally_frame": next_frame,
                            "court_info": normal_court_info,
                            "net_info": normal_net_info,
                        }
                        write_json(court_dict, video_name,
                                  f"{self.result_path}/courts/court_kp", "w")
                    
                    have_court_dict = {str(current_frame): True}
                    court_mse_dict = {str(current_frame): court_detect.mse}
                    write_json(court_mse_dict, video_name,
                              f"{self.result_path}/courts/court_mse")
                    write_json(have_court_dict, video_name,
                              f"{self.result_path}/courts/have_court")
                    write_json(players_dict, video_name,
                              f"{self.result_path}/players/player_kp")
                    
                    pbar.update(1)
            
            video.release()
            
            # Ball detection using TrackNet
            print("-" * 10 + "Starting Ball Detection" + "-" * 10)
            for res_root, res_dirs, res_files in os.walk(
                    f"{self.result_path}/videos/{video_name}"):
                for res_file in res_files:
                    _, ext = os.path.splitext(res_file)
                    if ext.lower() in ['.mp4']:
                        res_video_path = os.path.join(res_root, res_file)
                        print(res_video_path)
                        ball_detect(res_video_path, self.result_path)
            
            print("-" * 10 + "Challenge Processing Complete" + "-" * 10)
            print(f"Results saved to: {self.result_path}")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            logging.basicConfig(filename='logs/error.log', level=logging.ERROR,
                              format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(f"Error processing {video_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    def initiate_challenge(self):
        """Save buffer and process the video in a separate thread."""
        self.challenge_count += 1
        print(f"\n{'*'*50}")
        print(f"CHALLENGE #{self.challenge_count} INITIATED!")
        print(f"{'*'*50}\n")
        
        # Save the buffer
        filepath = self.save_buffer()
        
        if filepath and PROCESSING_AVAILABLE:
            # Process in separate thread to avoid blocking capture
            self.processing_thread = threading.Thread(
                target=self.process_video,
                args=(filepath,),
                daemon=True
            )
            self.processing_thread.start()
            print("Processing started in background...")
        elif filepath:
            print(f"Video saved but processing unavailable. Process manually with:")
            print(f"  python main.py --file_path {filepath} --result_path {self.result_path}")
        
    def display_live_feed(self):
        """
        Display live video feed with instructions.
        Press 'c' to save current buffer and process (challenge)
        Press 'q' to quit
        """
        print("\n" + "="*60)
        print("BADMINTON CHALLENGE SYSTEM")
        print("="*60)
        print("Controls:")
        print("  'c' - Initiate challenge (save & process last 10 seconds)")
        print("  'q' - Quit")
        print(f"\nBuffer size: {self.buffer_seconds} seconds")
        print(f"Result path: {self.result_path}")
        print(f"Processing available: {PROCESSING_AVAILABLE}")
        print("="*60 + "\n")
        
        while self.running:
            if len(self.frame_buffer) > 0:
                # Get most recent frame
                frame, _ = self.frame_buffer[-1]
                
                # Add status overlay
                display_frame = frame.copy()
                buffer_fill = len(self.frame_buffer) / self.buffer_size
                
                # Status background
                cv2.rectangle(display_frame, (5, 5), (500, 120), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (5, 5), (500, 120), (0, 255, 0), 2)
                
                # Status text
                cv2.putText(display_frame, "BADMINTON CHALLENGE SYSTEM", (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, 
                           f"Buffer: {len(self.frame_buffer)}/{self.buffer_size} ({buffer_fill*100:.0f}%)", 
                           (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Challenges: {self.challenge_count}", (15, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "Press 'C' to challenge", (15, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('Badminton Challenge Camera', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                self.initiate_challenge()
                
            elif key == ord('q') or key == ord('Q'):
                print("\nShutting down...")
                self.stop()
                break
                
    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Video capture stopped")
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


def main():
    """Main entry point for the challenge system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Badminton Challenge System')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--buffer_seconds', type=int, default=10,
                       help='Buffer duration in seconds (default: 10)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--result_path', type=str, default='res',
                       help='Path for processing results (default: res)')
    
    args = parser.parse_args()
    
    # Initialize the challenge system
    system = BadmintonChallengeSystem(
        camera_index=args.camera,
        buffer_seconds=args.buffer_seconds,
        fps=args.fps,
        result_path=args.result_path
    )
    
    try:
        # Start capturing
        system.start()
        
        # Display live feed and handle input
        system.display_live_feed()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.stop()


if __name__ == "__main__":
    main()