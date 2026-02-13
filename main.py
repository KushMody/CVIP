import cv2
import torch

# Load YOLOv5 pretrained model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small model, fast

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 detection
    results = model(frame)

    # Results contains bounding boxes, labels, confidence
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    for i, label in enumerate(labels):
        x1, y1, x2, y2, conf = cords[i]
        x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        class_name = model.names[int(label)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Print detected object in CMD
        print(f"Detected: {class_name}, Confidence: {conf:.2f}")

    cv2.imshow("YOLO Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
