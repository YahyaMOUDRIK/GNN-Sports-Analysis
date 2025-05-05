import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def detect_shooter(video_path, output_dir, ball_conf=0.3, player_conf=0.3):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize YOLOv8 models
    player_model = YOLO("yolov8n.pt")  # For players (class 0)
    ball_model = YOLO("yolov8n.pt")     # For basketballs (train or use custom model)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    ball_trajectory = []  # Stores (x, y) of the ball in each frame
    
    # Step 1: Track the ball's trajectory
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect the ball (use a custom model or filter by class if available)
        ball_results = ball_model(frame, conf=ball_conf, classes=[0])  # Adjust class index if needed
        ball_boxes = ball_results[0].boxes.xyxy.cpu().numpy()
        
        if len(ball_boxes) > 0:
            # Assume the first detected ball is the one being shot
            x1, y1, x2, y2 = map(int, ball_boxes[0])
            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            ball_trajectory.append(ball_center)
        else:
            ball_trajectory.append(None)
        
        frame_count += 1
    
    cap.release()
    
    # Step 2: Find the frame where the ball is released (sudden upward motion)
    release_frame = None
    for i in range(1, len(ball_trajectory)):
        if ball_trajectory[i-1] and ball_trajectory[i]:
            dy = ball_trajectory[i][1] - ball_trajectory[i-1][1]
            if dy < -5:  # Ball moves upward sharply
                release_frame = i
                break
    
    if release_frame is None:
        print("Ball release not detected. Using first frame.")
        release_frame = 0
    
    # Step 3: Detect the shooter in the release frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
    ret, frame = cap.read()
    
    player_results = player_model(frame, conf=player_conf, classes=0)
    player_boxes = player_results[0].boxes.xyxy.cpu().numpy()
    
    if len(player_boxes) == 0:
        print("No players detected in release frame.")
        return
    
    # Find player closest to the ball at release
    ball_x, ball_y = ball_trajectory[release_frame]
    min_distance = float("inf")
    shooter_box = None
    
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box)
        player_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = np.sqrt((player_center[0] - ball_x)**2 + (player_center[1] - ball_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            shooter_box = box
    
    # Step 4: Crop the shooter from all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if shooter_box is not None:
            x1, y1, x2, y2 = map(int, shooter_box)
            cropped = frame[y1:y2, x1:x2]
            
            # Enhance low-quality crops (optional)
            cropped = cv2.resize(cropped, (224, 224))  # Standard size
            cropped = cv2.medianBlur(cropped, 3)       # Reduce noise
            
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.jpg"), cropped)
        
        frame_count += 1
    
    cap.release()
    print(f"Shooter cropped and saved to {output_dir}")

# Example usage
detect_shooter(
    video_path="data/raw_videos/Clip29Miss.mp4",
    output_dir="data/processed/Clip29Miss",
    ball_conf=0.2,    # Lower confidence for ball detection
    player_conf=0.2
)