# physics based model
import numpy as np
import json
import os
from pathlib import Path
import cv2
import sys


class DragModel:
    @staticmethod
    def load_data(filepath):
        """ Load shuttlecock data from a json file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {filepath}")
            return None
    
    @staticmethod
    def homographyMatrix(file_path_court):
        """ Compute homography matrix from court keypoints"""
        court_data = DragModel.load_data(file_path_court)
        if court_data is None:
            return None, None

        court_info = court_data["court_info"] # access court info
        net_info = court_data["net_info"] # access net info


        # 2D image points from the court keypoints (pixels)
        img_points = np.array([
            court_info[0], # bottom-left
            court_info[1], # bottom-right
            court_info[2], # middle-left
            court_info[3], # middle-right
            court_info[4], # top-left
            court_info[5], # top-right
            net_info[1],   # netpole bottom-left
            net_info[2]   # netpole bottom-right
        ], dtype=np.float32)

        # Corresponding 2D world points (in metres with top left as origin)
        world_points = np.array([
            [0.46, 13.4],   # bottom-left
            [5.64, 13.4],   # bottom-right
            [0.46, 6.7],    # middle-left
            [5.64, 6.7],    # middle-right
            [0.46, 0],      # top-left
            [5.64, 0],      # top-right
            [0, 6.7],       # netpole bottom-left
            [6.1, 6.7]      # netpole bottom-right
        ], dtype=np.float32)

        # Compute homography matrix
        H, status = cv2.findHomography(img_points, world_points)
        return H, status
    
    @staticmethod
    def pixel_to_world(H, x, y):
        """ Convert pixel coordinates to world coordinates using homography matrix"""
        if H is None:
            raise ValueError("Homography H is None")
        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)   # shape (1,1,2)
        world = cv2.perspectiveTransform(pts, H)                    # shape (1,1,2)
        X, Y = float(world[0,0,0]), float(world[0,0,1])
        return X, Y
    
    def get_velocity_world(file_path_ball, H, fps):
        shuttlecock_data = DragModel.load_data(file_path_ball)
        frame_numbers = sorted([int(k) for k in shuttlecock_data.keys()])

        velocities = []
        for i in range(len(frame_numbers)-1):
            frame1_num = frame_numbers[i] # Frame number 1
            frame2_num = frame_numbers[i+1] # Frame number 2
            f1 = shuttlecock_data[str(frame1_num)] # Frame data 1
            f2 = shuttlecock_data[str(frame2_num)] # Frame data 2

            is_visible1 = f1['visible'] == 1
            is_visible2 = f2['visible'] == 1
            is_consecutive = frame1_num == frame2_num - 1
            
            if is_visible1 and is_visible2 and is_consecutive: # confirm both frames are visible and consecutive
                dt = 1/fps  # time difference between frames using fps
                x1, y1 = f1['x'], f1['y'] # pixels
                x2, y2 = f2['x'], f2['y'] # pixels
                X1, Y1 = DragModel.pixel_to_world(H, x1, y1) # world coords
                X2, Y2 = DragModel.pixel_to_world(H, x2, y2) # world coords
                
                vx = (X2 - X1)/dt # velocity in x direction
                vy = (Y2 - Y1)/dt # velocity in y direction
                velocities.append({ #data structure to hold velocity info in m/s
                "frame1": frame1_num,
                "frame2": frame2_num,
                "vx_mps": vx,
                "vy_mps": vy,
                "speed_mps": np.sqrt(vx**2 + vy**2)
                })
        return velocities
    
    @staticmethod
    def estimate_z_position(file_path_court, x_pixel, y_pixel):
        """ 
        Estimate the Z-position (height) of the shuttlecock using pole references.
        """
        court_data = DragModel.load_data(file_path_court)
        if court_data is None:
            return None

        net_info = court_data["net_info"]
        
        left_top_pixel = net_info[0]    
        left_bottom_pixel = net_info[1] 
        right_bottom_pixel = net_info[2]   
        right_top_pixel = net_info[3] 
        
        Z_net = 1.55  # Net height in meters (standard badminton net height)
        
        # Calculate pixel distance in video frame
        dy_left = abs(left_bottom_pixel[1] - left_top_pixel[1])
        dy_right = abs(right_bottom_pixel[1] - right_top_pixel[1])
        
        if dy_left == 0 or dy_right == 0:
            return 0.0
        
        # Calculate scale factors (meters per pixel in vertical direction)
        scale_Z_left = Z_net / dy_left
        scale_Z_right = Z_net / dy_right
        
        # Interpolate scale based on horizontal position
        # Use x-pixel to determine which pole is closer
        x_left = left_bottom_pixel[0]
        x_right = right_bottom_pixel[0]
        
        if x_right != x_left:
            interp_factor = np.clip((x_pixel - x_left) / (x_right - x_left), 0.0, 1.0)
        else:
            interp_factor = 0.5
        
        scale_Z = (1 - interp_factor) * scale_Z_left + interp_factor * scale_Z_right
        
        # Interpolate the ground-level y-pixel at the shuttlecock's x-position
        y_ground_at_x = (1 - interp_factor) * left_bottom_pixel[1] + interp_factor * right_bottom_pixel[1]
        
        # Calculate height based on vertical pixel difference
        # If y_pixel < y_ground_at_x, the shuttlecock is above ground (since y increases downward)
        dy_shuttle = y_ground_at_x - y_pixel
        
        if dy_shuttle < 0:
            return 0.0
        
        Z_est = dy_shuttle * scale_Z
        
        return Z_est
    
    @staticmethod
    def get_velocity_worlds(file_path_ball, H, file_path_court, fps):
        """ Calculate velocity in 3D (X, Y, Z) in m/s """
        shuttlecock_data = DragModel.load_data(file_path_ball)
        frame_numbers = sorted([int(k) for k in shuttlecock_data.keys()])

        velocities = []
        
        positions_3D = {}
        for frame_num in frame_numbers:
            f = shuttlecock_data[str(frame_num)]
            if f['visible'] == 1:
                x, y = f['x'], f['y']
                X, Y = DragModel.pixel_to_world(H, x, y)
                Z = DragModel.estimate_z_position(file_path_court, X, y) # get estimated Z position
                positions_3D[frame_num] = {'X': X, 'Y': Y, 'Z': Z}
                
        # Calculate velocities
        for i in range(len(frame_numbers)-1):
            frame1_num = frame_numbers[i]
            frame2_num = frame_numbers[i+1]

            # Only calculate velocity if both frames were visible and consecutive
            if frame1_num in positions_3D and frame2_num in positions_3D and frame2_num == frame1_num + 1:
                dt = 1/fps
                p1 = positions_3D[frame1_num]
                p2 = positions_3D[frame2_num]
                
                # Component velocities
                vx = (p2['X'] - p1['X']) / dt
                vy = (p2['Y'] - p1['Y']) / dt
                vz = (p2['Z'] - p1['Z']) / dt
                
                # Total 3D speed (Euclidean distance in 3D space)
                speed_mps = np.sqrt(vx**2 + vy**2 + vz**2)
                
                velocities.append({
                "frame1": frame1_num,
                "frame2": frame2_num,
                "vx_mps": vx,
                "vy_mps": vy,
                "vz_mps": vz,
                "speed_mps": speed_mps,
                "Z_frame1": p1['Z'],
                "Z_frame2": p2['Z']  
                })
        return velocities

### Testing
if __name__ == "__main__":
    file_path_ball = 'res/ball/loca_info/test1/test1_273-547.json'
    file_path_court = 'res/courts/court_kp/test1.json'

    H, status = DragModel.homographyMatrix(file_path_court)
    print("Homography matrix:\n", H)
    velocities2D = DragModel.get_velocity_world(file_path_ball, H, fps=30.0)
    velocities3D = DragModel.get_velocity_worlds(file_path_ball, H, file_path_court, fps=30.0)

    print("\nShuttlecock Speed (m/s) and Time (s):")
    print("-" * 35)
    print(f"{'Time (s)':>8} | {'Height (m)':>10} | {'Speed (m/s)':>11}")
    print("-" * 35)
    
    if velocities3D:
        time_per_frame = 1.0 / 30 # Time interval between frames
        
        for v in velocities3D:
            # The time of the measurement is the frame number multiplied by the time per frame.
            # We use 'frame1' as the starting point for the interval.
            start_frame = v['frame1']
            time_s = start_frame * time_per_frame
            speed = v['speed_mps']
            height = v['Z_frame1']
            
            # Print formatted output
            print(f"{time_s:8.3f} | {height:10.3f} | {speed:11.2f}")
            
    else:
        print("No velocities were calculated.")
