# AI Blind Assistive Navigation System

An AI-powered real-time assistive navigation system designed for visually impaired users using YOLOv8, OpenCV, OCR, and voice feedback.

---

# Features

- Real-time object detection
- Voice navigation assistance
- Emergency obstacle alerts
- OCR text reading
- Direction detection
- Distance estimation
- FPS monitoring
- Smart voice cooldown system
- Accessibility-focused AI solution

---

# Screenshots

## Real-Time Detection

![Detection](assets/screenshots/detection.png)

## OCR Reading

![OCR](assets/screenshots/ocr.png)

## Navigation Assistance

![Navigation](assets/screenshots/navigation.png)

---

# System Architecture

![Architecture](assets/architecture.png)

---

# Technologies Used

- Python
- YOLOv8
- OpenCV
- PyTTSX3
- Pytesseract
- NumPy

---

# Project Structure

```bash
Blind_Assistive_System/
│
├── assets/
│   ├── screenshots/
│   └── architecture.png
│
├── models/
│   └── yolov8n.pt
│
├── src/
│   └── blind_assist.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
```

## Navigate to Project

```bash
cd Blind_Assistive_System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Install OCR Engine (Mac)

```bash
brew install tesseract
```

---

# Run Project

```bash
cd src
python blind_assist.py
```

---

# Controls

| Key | Action |
|-----|--------|
| t | OCR Text Reading |
| r | Reset Voice Memory |
| q | Quit Application |

---

# Future Improvements

- Depth estimation
- Smart glasses integration
- Mobile application
- Cloud connectivity
- GPS emergency alerts
- Raspberry Pi deployment

---

# Resume Description

Developed an AI-powered Blind Assistive Navigation System using YOLOv8, OpenCV, OCR, and speech synthesis to provide real-time obstacle detection, directional voice alerts, text reading assistance, and accessibility support for visually impaired users.

---

# License

MIT License
