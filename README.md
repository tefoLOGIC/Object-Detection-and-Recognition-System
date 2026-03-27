# Object Detection and Recognition System 🎯

A real-time object detection and recognition system built with Python and YOLOv3, capable of detecting 80+ object categories in both static images and live webcam streams.

---

## 📌 Project Overview

This project implements two modes of object detection:
- **Image Detection** — Detects and labels objects in a static image and saves the result
- **Real-time Detection** — Uses your webcam to detect objects live and saves the output as a video

The system uses the **YOLOv3 (You Only Look Once)** deep learning model with the **COCO dataset** (80 object classes), powered by OpenCV's DNN module.

---

## 🗂️ Project Structure

```
Object-Detection-and-Recognition-System/
│
├── src/
│   ├── image_detection.py          # Detect objects in a static image
│   └── realtime_detection.py       # Real-time detection via webcam
│
├── model/
│   ├── yolov3.cfg                  # YOLOv3 network configuration
│   ├── coco.names                  # 80 COCO class labels
│   └── yolov3.weights              # ⚠️ Not included — download separately (see below)
│
├── samples/
│   ├── input/                      # Place your sample input images here
│   └── output/                     # Output results saved here
│
├── OUTPUT VIDEOS/                  # Auto-created folder for webcam output videos
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚠️ Download YOLOv3 Weights

The `yolov3.weights` file (~236MB) is **not included** in this repository due to file size limits.

Download it from the official YOLO website:

🔗 [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

After downloading, place it inside the `model/` folder:
```
model/
└── yolov3.weights   ✅
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/Object-Detection-and-Recognition-System.git
cd Object-Detection-and-Recognition-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv3 weights (see above)

---

## 🚀 How to Run

### 🖼️ Image Detection
Detects objects in a static image and displays + saves the result.

1. Place your input image in `samples/input/`
2. Update the `image_path` in `src/image_detection.py` to point to your image
3. Run:
```bash
python src/image_detection.py
```
- Output saved to `samples/output/result.jpg`

---

### 📷 Real-time Webcam Detection
Detects objects live via your webcam and saves the video.

```bash
python src/realtime_detection.py
```
- Press `ESC` to stop
- Output videos auto-saved to `OUTPUT VIDEOS/` with timestamped filenames

---

## 🧠 How It Works

1. **Load YOLOv3 model** using OpenCV's `cv2.dnn.readNet()`
2. **Preprocess** the image/frame into a blob (416×416, normalized)
3. **Forward pass** through YOLOv3 to get detections
4. **Apply NMS** (Non-Maximum Suppression) to remove duplicate boxes
5. **Draw bounding boxes** with class labels and confidence scores
6. **Save** result image or video

---

## 🛠️ Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.10 |
| OpenCV (cv2) | 4.9.0.80 |
| NumPy | 1.26.4 |
| YOLOv3 Model | COCO (80 classes) |
| Platform | PyCharm |

---

## 📦 Detectable Object Classes (Sample)

Person, Car, Bicycle, Dog, Cat, Chair, Bottle, Laptop, Phone, Bus, Truck, Bird, and 68 more from the COCO dataset.

---

## 👩‍💻 Author

**Tanuja Devi. M**
---

## 📄 License

This project is for educational purposes.
YOLOv3 model by Joseph Redmon — [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
