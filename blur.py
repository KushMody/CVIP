# Alternative version with more optimizations
import cv2
import time

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

# Performance settings
frame_skip = 3
frame_counter = 0
prev_faces = []
use_prev_faces = True  # Flag to use previous detection

# Region of interest for face detection (reduce search area)
roi_enabled = False
roi_coords = (0, 0, 640, 480)  # x, y, w, h

# Timing for performance monitoring
last_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (current_time - last_time)) if current_time - last_time > 0 else fps
    last_time = current_time
    
    frame_counter += 1
    
    # Face detection on selected frames
    if frame_counter % frame_skip == 0 or not use_prev_faces:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI if enabled
        if roi_enabled:
            x, y, w, h = roi_coords
            roi_gray = gray[y:y+h, x:x+w]
            faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            # Adjust coordinates to full frame
            faces = [(x+fx, y+fy, fw, fh) for (fx, fy, fw, fh) in faces]
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
        prev_faces = faces
        use_prev_faces = True
    else:
        faces = prev_faces

    # Process faces
    for (x, y, w, h) in faces:
        # Extract and blur face
        face_roi = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (45, 45), 30)
        frame[y:y+h, x:x+w] = blurred_face
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Update ROI for next detection (face tracking optimization)
        if roi_enabled:
            roi_coords = (max(0, x-50), max(0, y-50), min(640, w+100), min(480, h+100))

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Optimized Face Blur", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):  # Reset ROI
        roi_enabled = not roi_enabled
        roi_coords = (0, 0, 640, 480)
        use_prev_faces = False

cap.release()
cv2.destroyAllWindows()