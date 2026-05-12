import cv2
from ultralytics import YOLO
import pyttsx3
import pytesseract
import time

# -----------------------------
# Initialize Voice Engine
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("../models/yolov8n.pt")

# -----------------------------
# Important Objects
# -----------------------------
IMPORTANT_OBJECTS = [

    # Humans
    "person",

    # Vehicles
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "train",

    # Road Safety
    "traffic light",
    "stop sign",
    "parking meter",

    # Indoor Navigation
    "chair",
    "bench",
    "couch",
    "bed",
    "dining table",
    "potted plant",
    "tv",
    "laptop",
    "cell phone",

    # Doors & Access
    "door",

    # Animals
    "dog",
    "cat",
    "horse",
    "cow",

    # Public Environment
    "fire hydrant",
    "backpack",
    "handbag",
    "suitcase",

    # Kitchen / Dangerous Objects
    "knife",
    "bottle",
    "cup",

    # Navigation Obstacles
    "stairs",
    "wall",

]

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# -----------------------------
# Voice Cooldown Memory
# -----------------------------
spoken_objects = {}

# -----------------------------
# FPS Variables
# -----------------------------
prev_time = 0

# -----------------------------
# Voice Cooldown Seconds
# -----------------------------
VOICE_COOLDOWN = 5

print("Press 't' for OCR | 'r' reset memory | 'q' quit")

# =============================
# MAIN LOOP
# =============================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    # -----------------------------
    # YOLO Detection
    # -----------------------------
    results = model(frame, verbose=False)

    h, w, _ = frame.shape
    frame_center = w // 2

    # =============================
    # PROCESS DETECTIONS
    # =============================
    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            conf = float(box.conf[0])

            # Confidence Filter
            if conf < 0.5:
                continue

            # Important Objects Only
            if label not in IMPORTANT_OBJECTS:
                continue

            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            obj_center = (x1 + x2) // 2

            # -----------------------------
            # Direction Detection
            # -----------------------------
            if obj_center < frame_center - 200:
                direction = "far left"

            elif obj_center < frame_center - 100:
                direction = "left"

            elif obj_center > frame_center + 200:
                direction = "far right"

            elif obj_center > frame_center + 100:
                direction = "right"

            else:
                direction = "ahead"

            # -----------------------------
            # Distance Estimation
            # -----------------------------
            area = (x2 - x1) * (y2 - y1)

            if area > 120000:
                distance_msg = "emergency"
                danger = True

            elif area > 80000:
                distance_msg = "very close"
                danger = True

            elif area > 30000:
                distance_msg = "near"
                danger = False

            else:
                distance_msg = "far"
                danger = False

            # -----------------------------
            # Alert Message
            # -----------------------------
            if distance_msg == "emergency":

                message = f"Emergency stop! {label} extremely close"

            elif danger:

                message = f"Danger! {label} {direction}, very close"

            else:

                message = f"{label} {direction}, {distance_msg}"

            # -----------------------------
            # Unique Object Key
            # -----------------------------
            object_key = f"{label}_{direction}"

            # -----------------------------
            # Smart Voice Cooldown
            # -----------------------------
            current = time.time()

            should_speak = (
                object_key not in spoken_objects
                or current - spoken_objects[object_key] > VOICE_COOLDOWN
            )

            if should_speak:

                print("Speaking:", message)

                try:
                    engine.say(message)
                    engine.runAndWait()

                except Exception as e:
                    print("Voice Error:", e)

                spoken_objects[object_key] = current

            # -----------------------------
            # Bounding Box Color
            # -----------------------------
            color = (0, 0, 255) if danger else (0, 255, 0)

            # Draw Rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display Text
            cv2.putText(
                frame,
                message,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # -----------------------------
    # FPS Display
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # -----------------------------
    # Show Window
    # -----------------------------
    cv2.imshow("Blind Assistive System", frame)

    key = cv2.waitKey(1) & 0xFF

    # =============================
    # OCR TEXT READER
    # =============================
    if key == ord('t'):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.threshold(
            gray,
            150,
            255,
            cv2.THRESH_BINARY
        )[1]

        text = pytesseract.image_to_string(gray)

        if text.strip() != "":

            print("Detected Text:", text)

            try:
                engine.say(text)
                engine.runAndWait()

            except Exception as e:
                print("OCR Voice Error:", e)

        else:

            engine.say("No readable text found")
            engine.runAndWait()

    # =============================
    # RESET MEMORY
    # =============================
    if key == ord('r'):

        spoken_objects.clear()

        print("Memory Reset")

    # =============================
    # QUIT
    # =============================
    if key == ord('q'):
        break

# -----------------------------
# Release Resources
# -----------------------------
cap.release()

cv2.destroyAllWindows()