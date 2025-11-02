import cv2
import numpy as np
from collections import deque
from datetime import datetime
import threading
import os

class VideoBufferCapture:
    def __init__(self, camera_index=0, buffer_seconds=10, fps=30):
        """
        Initialize video capture with rolling buffer.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            buffer_seconds: Number of seconds to keep in buffer
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        
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
        
        # Create output directory
        self.output_dir = "challenge_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
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
                # Add timestamp to frame
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
        
    def display_live_feed(self):
        """
        Display live video feed with instructions.
        Press 'c' to save current buffer (challenge)
        Press 'q' to quit
        """
        print("\nControls:")
        print("  'c' - Save current buffer (initiate challenge)")
        print("  'q' - Quit")
        print("\nBuffer size: {} seconds".format(self.buffer_seconds))
        
        while self.running:
            if len(self.frame_buffer) > 0:
                # Get most recent frame
                frame, _ = self.frame_buffer[-1]
                
                # Add status overlay
                display_frame = frame.copy()
                buffer_fill = len(self.frame_buffer) / self.buffer_size
                status_text = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size} frames ({buffer_fill*100:.0f}%)"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to save challenge", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Badminton Camera', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print("\n>>> Challenge initiated! Saving buffer...")
                self.save_buffer()
                
            elif key == ord('q'):
                print("\nQuitting...")
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
    # Initialize capture with 10-second buffer at 30 FPS
    capture = VideoBufferCapture(
        camera_index=0,      # Change this if using different camera
        buffer_seconds=10,   # 10-second rolling buffer
        fps=30               # Target 30 FPS
    )
    
    try:
        # Start capturing
        capture.start()
        
        # Display live feed and handle keyboard input
        capture.display_live_feed()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()