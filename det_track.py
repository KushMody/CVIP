import cv2
import numpy as np
from collections import deque

class ObjectTracker:
    """Tracks objects across frames using color-based detection and centroid tracking"""
    
    def __init__(self, max_distance=50, history_length=30):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.history_length = history_length
    
    def distance(self, pt1, pt2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def update(self, centroids):
        """Update tracks with new centroids"""
        if len(self.tracks) == 0:
            # First frame - create new tracks for all objects
            for i, centroid in enumerate(centroids):
                self.tracks[self.next_id] = {
                    'centroid': centroid,
                    'history': deque([centroid], maxlen=self.history_length),
                    'frames_missing': 0
                }
                self.next_id += 1
            return self.tracks
        
        # Match new centroids to existing tracks
        existing_ids = list(self.tracks.keys())
        used_centroids = set()
        
        for track_id in existing_ids:
            track = self.tracks[track_id]
            best_match = None
            best_distance = float('inf')
            
            for i, centroid in enumerate(centroids):
                if i in used_centroids:
                    continue
                dist = self.distance(track['centroid'], centroid)
                if dist < best_distance and dist < self.max_distance:
                    best_distance = dist
                    best_match = i
            
            if best_match is not None:
                self.tracks[track_id]['centroid'] = centroids[best_match]
                self.tracks[track_id]['history'].append(centroids[best_match])
                self.tracks[track_id]['frames_missing'] = 0
                used_centroids.add(best_match)
            else:
                self.tracks[track_id]['frames_missing'] += 1
        
        # Create new tracks for unmatched centroids
        for i, centroid in enumerate(centroids):
            if i not in used_centroids:
                self.tracks[self.next_id] = {
                    'centroid': centroid,
                    'history': deque([centroid], maxlen=self.history_length),
                    'frames_missing': 0
                }
                self.next_id += 1
        
        # Remove lost tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v['frames_missing'] < 5}
        
        return self.tracks


def detect_objects_by_color(frame, lower_color, upper_color):
    """
    Detect objects within specific HSV color range
    Returns contours and centroids
    """
    # Convert BGR to HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    valid_contours = []
    
    # Filter contours by area and calculate centroids
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                valid_contours.append(contour)
    
    return valid_contours, centroids, mask


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    tracker = ObjectTracker(max_distance=50, history_length=30)
    
    # Color ranges in HSV (Hue, Saturation, Value)
    # Red objects
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Blue objects
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Green objects
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    fps = 0
    frame_count = 0
    prev_time = cv2.getTickCount()
    
    mode = 1  # 1=Red, 2=Blue, 3=Green
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Select color range based on mode
        if mode == 1:
            contours1, centroids1, _ = detect_objects_by_color(frame, lower_red1, upper_red1)
            contours2, centroids2, _ = detect_objects_by_color(frame, lower_red2, upper_red2)
            contours = contours1 + contours2
            centroids = centroids1 + centroids2
            color_name = "RED"
            display_color = (0, 0, 255)
        elif mode == 2:
            contours, centroids, _ = detect_objects_by_color(frame, lower_blue, upper_blue)
            color_name = "BLUE"
            display_color = (255, 0, 0)
        else:
            contours, centroids, _ = detect_objects_by_color(frame, lower_green, upper_green)
            color_name = "GREEN"
            display_color = (0, 255, 0)
        
        # Update tracker with new centroids
        tracks = tracker.update(centroids)
        
        # Draw results
        for contour in contours:
            cv2.drawContours(frame, [contour], 0, display_color, 2)
        
        for track_id, track_data in tracks.items():
            cx, cy = track_data['centroid']
            
            # Draw circle at centroid
            cv2.circle(frame, (cx, cy), 5, display_color, -1)
            
            # Draw trajectory history
            history = list(track_data['history'])
            if len(history) > 1:
                for i in range(1, len(history)):
                    cv2.line(frame, history[i-1], history[i], display_color, 1)
            
            # Draw ID label
            cv2.putText(frame, f"ID:{track_id}", (cx - 20, cy - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
        
        # Calculate and display FPS
        current_time = cv2.getTickCount()
        elapsed = (current_time - prev_time) / cv2.getTickFrequency()
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
            prev_time = current_time
        
        # Display statistics
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {color_name} | Objects: {len(tracks)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press: 1=Red, 2=Blue, 3=Green, Q=Quit", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Object Detection & Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()