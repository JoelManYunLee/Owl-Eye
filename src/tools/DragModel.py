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
            net_info[0],   # netpole bottom-left
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
    def estimate_z_position(file_path_court, X, y_pixel):
        """ 
        Estimate the Z-position (height) of the shuttlecock based on its horizontal position (X)
        and vertical pixel position (y_pixel), using the net poles as scale references.
        """
        court_data = DragModel.load_data(file_path_court)
        if court_data is None:
            return None, None

        net_info = court_data["net_info"] # access net info
        net_pole_data = {
            "Z_net": 1.55, # Height of the net in meters
            "y_BL": net_info[0][1], # y-pixel of Bottom Left Pole (Z=0)
            "y_TL": net_info[1][1], # y-pixel of Top Left Pole (Z=1.55m)
            "y_BR": net_info[2][1], # y-pixel of Bottom Right Pole (Z=0)
            "y_TR": net_info[3][1], # y-pixel of Top Right Pole (Z=1.55m)
            "X_Left": 0.0, # World X-coord of Left Pole
            "X_Right": 6.1 # World X-coord of Right Pole
        }

        Z_net = net_pole_data["Z_net"]
        
        # 1. Calculate vertical pixel distance for each pole in pixels
        dy_left = abs(net_pole_data["y_BL"] - net_pole_data["y_TL"])
        dy_right = abs(net_pole_data["y_BR"] - net_pole_data["y_TR"])

        # 2. Calculate Z-scale (meters/pixel) at each pole's location
        scale_Z_left = Z_net / dy_left
        scale_Z_right = Z_net / dy_right

        # 3. Choose the closest pole's vertical scale based on the shuttlecock's X-coordinate
        X_Left = net_pole_data["X_Left"]
        X_Right = net_pole_data["X_Right"]
        
        # Simple interpolation factor based on X-position
        # This assumes a linear change in vertical perspective between the two poles.
        interp_factor = (X - X_Left) / (X_Right - X_Left)
        
        # Ensure factor is clamped between 0 and 1
        interp_factor = np.clip(interp_factor, 0.0, 1.0) 
        
        # Interpolate the Z-scale factor
        scale_Z = (1 - interp_factor) * scale_Z_left + interp_factor * scale_Z_right
        
        # 4. Estimate Z-position
        # The height is proportional to the difference between the shuttlecock's y-pixel
        # and the y-pixel of the net line (Z=0 at Y=6.7m) at that same X-position.
        
        # CRITICAL ASSUMPTION: We approximate the y-pixel of the net line (Z=0, Y=6.7m) 
        # for the shuttlecock's current X position by interpolating the mid-court Y-pixels.
        
        # For simplicity, we'll use the Y-pixel of the net pole bottom (Y=6.7m, Z=0)
        # as the reference y_pixel_ground (Y_net is the world Y-coord of the net).
        # Since the shuttlecock is generally tracked near the net line (Y=6.7m), this works.
        y_BL = net_pole_data["y_BL"]
        y_BR = net_pole_data["y_BR"]
        
        y_pixel_ground = (1 - interp_factor) * y_BL + interp_factor * y_BR
        
        # The vertical distance (in pixels) from the ground plane at that X-location
        # Since the pixel system has Y increasing DOWN, Z is proportional to (y_pixel_ground - y_pixel)
        dy_shuttle_pixel = y_pixel_ground - y_pixel
        
        # If the shuttlecock is below the ground reference line (impossible), Z is 0
        if dy_shuttle_pixel < 0:
            return 0.0
            
        Z_est = dy_shuttle_pixel * scale_Z
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
                "speed_mps": speed_mps
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
    print("Time (s) | Speed (m/s)")
    print("-" * 35)
    
    if velocities3D:
        time_per_frame = 1.0 / 30 # Time interval between frames
        
        for v in velocities3D:
            # The time of the measurement is the frame number multiplied by the time per frame.
            # We use 'frame1' as the starting point for the interval.
            start_frame = v['frame1']
            time_s = start_frame * time_per_frame
            speed = v['speed_mps']
            
            # Print formatted output
            print(f"{time_s:8.3f} | {speed:9.2f}")
            
    else:
        print("No velocities were calculated.")
