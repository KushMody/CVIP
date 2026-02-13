import cv2
import numpy as np

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Available Cascades:")
for cascade in cv2.data.haarcascades.split():
    print(f"  - {cascade}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    
    # Detect faces with relaxed parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=4,
        minSize=(40, 40)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract lower portion of face (mouth area)
        roi_y_start = y + int(h * 0.4)  # Start from 40% down
        roi_y_end = y + h
        
        face_roi_gray = gray[roi_y_start:roi_y_end, x:x+w]
        face_roi_color = frame[roi_y_start:roi_y_end, x:x+w]
        
        # Detect smile with RELAXED parameters
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.8,
            minNeighbors=15,      # Reduced from 20
            minSize=(15, 15),
            maxSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Show detection info
        cv2.putText(frame, f"Smile detections: {len(smiles)}", (x, y + h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw detected smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        
        # Determine if smiling (need at least 1 detection)
        if len(smiles) >= 1:
            cv2.putText(frame, "SMILING :)", (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NEUTRAL", (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Instructions
    cv2.putText(frame, "SMILE DETECTION - Press Q to quit", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Face & Smile Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended")