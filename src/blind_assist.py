import cv2
from ultralytics import YOLO
import pyttsx3
import pytesseract
from PIL import Image

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Important navigation objects
IMPORTANT_OBJECTS = [
    "person", "car", "bus", "truck", "bicycle",
    "motorcycle", "chair", "bench", "door"
]

cap = cv2.VideoCapture(0)
spoken_objects = set()

print("Press 't' for OCR | 'r' reset memory | 'q' quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    h, w, _ = frame.shape
    frame_center = w // 2

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Filter only important objects
            if label not in IMPORTANT_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center = (x1 + x2) // 2

            # Direction detection
            if obj_center < frame_center - 100:
                direction = "on left"
            elif obj_center > frame_center + 100:
                direction = "on right"
            else:
                direction = "ahead"

            # Distance estimation
            area = (x2 - x1) * (y2 - y1)

            if area > 80000:
                distance_msg = "very close"
                danger = True
            elif area > 30000:
                distance_msg = "near"
                danger = False
            else:
                distance_msg = "far"
                danger = False

            # Priority Alert Message
            if danger:
                message = "Danger! " + label + " " + direction + ", very close stop"
            else:
                message = label + " " + direction + ", " + distance_msg

            # Speak only once
            if label not in spoken_objects:
                print("Speaking:", message)
                engine.say(message)
                engine.runAndWait()
                spoken_objects.add(label)

            # Draw bounding box
            color = (0,0,255) if danger else (0,255,0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, message, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Blind Assistive System", frame)

    key = cv2.waitKey(1) & 0xFF

    # OCR Text Reader
    if key == ord('t'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        if text.strip() != "":
            print("Detected Text:", text)
            engine.say(text)
            engine.runAndWait()
        else:
            engine.say("No readable text found")
            engine.runAndWait()

    # Reset spoken memory
    if key == ord('r'):
        spoken_objects.clear()
        print("Memory Reset")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
