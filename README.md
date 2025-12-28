# 🚀 YOLO-SAM Toolkit: Modular Computer Vision Pipeline

This toolkit is a professional and modular solution for video and image processing, combining the speed of **YOLOv8** with the segmentation precision of **SAM 2.1 (Segment Anything Model)**. 

The system uses a **layer architecture** that allows sequential data processing, facilitating the creation of customized and automated workflows.

---

## ✨ Key Features

* **Intelligent Hardware Detection:** Automatic installer that configures PyTorch for NVIDIA (CUDA), AMD (ROCm), or CPU.
* **Precision Segmentation:** Native integration with SAM 2.1 for high-quality masks.
* **Modular Architecture:** Add or remove layers (filters, counters, visualizers) without touching the core code.
* **Data Management:** Automatic export of results to JSON and processed videos.
* **Optimized for Python 3.12.10:** Leveraging the latest performance and stability improvements.

---

## 🛠️ Prerequisites

The use of **Miniconda** is strongly recommended to avoid library conflicts.

1. **Install Miniconda:** [Download here](https://docs.conda.io/en/latest/miniconda.html)
2. **Verify Version:**
```bash
   python --version  # Should be 3.12.10
   conda --version
```

---

## 📥 Installation

### 1. Clone the repository
```bash
git clone https://github.com/FernandoVaras/yolov8_sam2_toolkit.git
cd yolov8_sam2_toolkit
```

### 2.1. Create the environment (Miniconda)

This step installs the base dependencies (OpenCV, NumPy < 2.0, etc.).
```bash
conda env create -f environment.yml
conda activate yolo_sam_toolkit
pip install -r requirements.txt
```

### 2.2. Create the environment (pip + venv)

This step installs the base dependencies (OpenCV, NumPy < 2.0, etc.).
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Complete Setup (Hardware + SAM 2)

Run the master script that detects your GPU, installs the correct PyTorch, and configures SAM 2:
```bash
python setup/setup_all.py
```

This script will clone SAM 2, compile it for your hardware, and download the weights (`.pt`) automatically.

---

## 🏗️ Layer Architecture

The heart of the toolkit is the `ProcessMedia` class. You can stack processors in a list and they will execute in order. Each layer receives a `frame_data` dictionary containing the results from previous layers.

### Usage example:
```python
from core.process_media import ProcessMedia
from core.yolo_layer import YOLOProcessor
from core.sam2_layer import SAM2Processor

pipeline = ProcessMedia(
    source="data/input/video.mp4",
    processors=[
        YOLOProcessor(model="yolov8n.pt", confidence=0.5), # Layer 1: Detect
        SAM2Processor(model_type="large"),                # Layer 2: Segment
    ],
    output=["video", "json"]
)
pipeline.run()
```

### How to create a Custom Layer?

It's very simple. Just create a class with a `process` method. 
```python
class MyCustomFilter:
    def process(self, frame_data):
        # frame_data['boxes'] contains YOLO boxes
        # frame_data['masks'] contains SAM 2 masks
        # frame_data['frame'] is the current image (OpenCV)
        
        print(f"Detected objects: {len(frame_data['boxes'])}")
        return frame_data
```

---

## 🧪 Testing and Verification

Before starting a project, verify that your hardware is properly configured with these test files:

* **YOLO Test:** `python tests/test_yolo.py` (Verifies detection and GPU usage).
* **SAM 2 Test:** `python tests/sam2_test.py` (Verifies segmentation and CUDA kernels).
* **Complete Pipeline:** `python main_example.py` (Runs a real example).

---

## 💻 Hardware Support

| Hardware | Operating System | Status | Notes |
|----------|------------------|--------|-------|
| NVIDIA GPU | Windows / Linux | ✅ Supported | CUDA 12.1 (Recommended) |
| AMD GPU | Linux (ROCm) | ✅ Supported | Requires ROCm drivers |
| AMD GPU | Windows | ⚠️ Partial | Runs in CPU mode for stability |
| Apple Silicon | macOS | ✅ Supported | Uses MPS acceleration |

---

## 📂 Project Structure
```
yolo-sam-toolkit/
├── segment-anything-2/  # Automatically cloned during setup
├── setup/
│   └── setup_all.py     # Intelligent installer
├── core/
│   ├── process_media.py # Main pipeline engine
│   ├── yolo_layer.py    # YOLOv8 processor
│   └── sam2_layer.py    # SAM 2 processor
├── tests/               # Validation scripts (yolo_test.py, etc.)
├── data/
│   ├── input/           # Folder for your videos/images
│   └── output/          # Results (Video + JSON + Frames)
├── environment.yml      # Configuration for Miniconda
└── README.md
```

---

## 📄 License

This project is under the MIT license. The YOLOv8 (Ultralytics) and SAM 2 (Meta) libraries have their own copyright licenses.