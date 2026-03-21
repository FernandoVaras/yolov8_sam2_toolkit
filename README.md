# YOLO-SAM Toolkit: Modular Computer Vision Pipeline

A modular toolkit for video and image processing that combines **YOLOv8** (detection) with **SAM 2.1** (segmentation). Built on a stackable layer architecture where each processor receives and returns a shared dictionary (`frame_data`).

---

## 1. Installation

### 1.1 Clone the repository

```bash
git clone https://github.com/FernandoVaras/yolov8_sam2_toolkit.git
cd yolov8_sam2_toolkit
```

### 1.2 Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 1.3 Install base dependencies

```bash
pip install -r requirements.txt
```

This installs all base dependencies without PyTorch. Required versions:

| Package | Version |
|---------|---------|
| ultralytics | >= 8.0.0 |
| roboflow | >= 1.1.0 |
| opencv-python | >= 4.8.0 |
| pillow | >= 9.5.0 |
| numpy | >= 1.24.0, < 2.0.0 |
| scipy | >= 1.10.0 |
| matplotlib | >= 3.7.0 |
| tqdm | >= 4.65.0 |
| pyyaml | >= 6.0 |
| requests | >= 2.31.0 |

> **Note:** NumPy must be < 2.0.0. SAM2 and ultralytics have compatibility issues with NumPy 2.x.

### 1.4 Complete setup (Hardware + SAM 2)

```bash
python setup/setup_all.py
```

This interactive script handles the full environment setup:

1. **Hardware detection** — Detects your GPU (NVIDIA / AMD / CPU) and asks whether to install PyTorch optimized for your hardware.
2. **SAM 2** — Offers 3 options:
   - **Full installation:** Clones the SAM 2 repository, installs it, and downloads the model weights you choose.
   - **Models only:** Downloads weights only (if SAM 2 is already installed).
   - **Skip:** Skips SAM 2 entirely.
3. **SAM 2 model selection:**
   - `tiny` — Fastest, lowest accuracy
   - `small` — Fast, good accuracy
   - `base_plus` — Balanced
   - `large` — Slowest, highest accuracy
   - You can select multiple models separated by commas (e.g., `tiny,large`).
4. **OpenH264 DLL (Windows only)** — OpenCV uses the `mp4v` codec by default, which some Windows media players cannot open. The setup offers to download the OpenH264 DLL (~1.5 MB) from Cisco, which enables H.264 video output that plays natively on any Windows player. If the DLL is present, the pipeline automatically uses H.264; otherwise, it falls back to `mp4v`.

---

## 2. Basic usage

```python
from core import ProcessMedia, YOLOProcessor, SAM2Processor, VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/video.mp4",
    processors=[
        YOLOProcessor(model="models/my_model.pt", confidence=0.5, entities=2),
        SAM2Processor(input_source="yolo:boxes", model_type="large", max_entities=2),
        VisualizationProcessor(
            input_keys={"yolo": ["boxes", "confidences", "labels"], "sam2": ["masks", "scores", "centroids"]},
            show_masks=True,
            show_boxes=True,
            show_centroids=True,
        )
    ],
    output=["video", "json"]
)
pipeline.run()
```

Results are saved to `data/output/<timestamp>/` with the processed video and a `.jsonl` file containing per-frame data.

---

## 3. ProcessMedia (Pipeline)

Main engine that runs processors sequentially on each frame.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | *(required)* | Path to input video or image |
| `processors` | `[]` | List of processor objects to run in order |
| `output` | `"video"` | Output types: `"video"`, `"json"`, `"frames"`, or a list with multiple |
| `output_dir` | `"data/output"` | Base folder for results |
| `use_vis_frame` | `True` | If `True`, uses the visualized frame for the output video |
| `start_frame` | `0` | First frame to process (for partial video processing) |
| `end_frame` | `None` | Last frame to process. `None` = process until the end |

**About outputs:**
- `"video"` — Processed MP4 video with visualizations applied. Uses H.264 codec if OpenH264 DLL is available, otherwise falls back to mp4v.
- `"json"` — JSONL file (one JSON line per frame) with metadata and scalar results
- `"frames"` — Individual JPEG frames, organized in subdirectories of 1,000

---

## 4. YOLOProcessor

Detection layer using YOLOv8. Runs inference and applies configurable filters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"yolov8n.pt"` | Path to YOLO model (.pt) |
| `confidence` | `0.5` | Minimum confidence to accept a detection |
| `entities` | `None` | Exact number of detections to keep (takes top N by confidence) |
| `max_entities` | `None` | Maximum number of detections allowed |
| `min_entities` | `None` | Minimum detections. If not reached, lowers confidence automatically (max 2 retries) |
| `area` | `None` | Enables adaptive area filter: uses the first detection's area as reference |
| `area_error` | `0.2` | Adaptive area tolerance (0.2 = +/-20% from reference) |
| `area_max` | `None` | Maximum area in pixels to accept a detection |
| `area_min` | `None` | Minimum area in pixels |
| `classes` | `None` | List of class IDs to include (e.g., `[0, 1]`). Others are discarded |
| `exclude_classes` | `None` | List of class IDs to exclude |
| `max_overlap` | `None` | IoU threshold for additional NMS between detections (e.g., `0.9`) |
| `edge_margin` | `0` | Margin in pixels from frame edge. Detections inside the margin are discarded |
| `roi` | `None` | Region of interest `[x1, y1, x2, y2]`. Only detections with center inside are kept |
| `output_key` | `"yolo"` | Namespace on the data bus where results are written |
| `use_tracking` | `False` | Enable persistent identity tracking across frames |
| `tracking_proximity` | `80` | Max distance in pixels between centroids for identity matching |
| `tracking_area_tolerance` | `0.5` | Allowed area variation between frames for identity matching (0.5 = +/-50%) |

**Cannot combine:**
- `entities` with `max_entities` or `min_entities`
- `area` with `area_max` or `area_min`

**Tracking:** When `use_tracking=True`, detections are reordered by persistent identity slots using centroid proximity and area consistency. This ensures that slot 0 always corresponds to the same object across frames, keeping colors and trajectories stable in the visualization. Requires `max_entities` or `entities` to be set.

**Data written to the bus** (`frame_data["yolo"]`):

| Key | Type | Description |
|-----|------|-------------|
| `boxes` | numpy array | Bounding boxes `[x1, y1, x2, y2]` |
| `classes` | numpy array | Class IDs per detection |
| `labels` | list[str] | Class names (e.g., `["mouse", "mouse"]`) |
| `confidences` | numpy array | Confidence scores |
| `masks` | list | Segmentation masks (if the YOLO model supports them) |
| `keypoints` | list | Keypoints (if the YOLO model supports them) |

---

## 5. SAM2Processor

Segmentation layer using SAM 2.1. Generates precise masks and maintains identity across frames with fixed slots.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"large"` | SAM2 model: `"large"` or `"tiny"` |
| `input_source` | *(required)* | Bus mode: `"yolo:boxes"` (reads YOLO boxes). Manual mode: list of points `[{"points": [[x, y]], "labels": [1]}]` |
| `output_key` | `"sam2"` | Namespace on the data bus |
| `max_entities` | `2` | Maximum number of objects to track (fixed identity slots) |
| `proximity_threshold` | `50` | Max distance in pixels between centroids to consider it the same object across frames |
| `area_tolerance` | `0.10` | Allowed area variation between frames (0.10 = +/-10%) |
| `iou_threshold` | `0.5` | IoU threshold to remove duplicate masks |
| `skip_threshold` | `5.0` | If centroids move less than N pixels, reuses previous output without running SAM2. `0` = disabled |

**Input modes:**
- **Bus mode** (`input_source="yolo:boxes"`): Reads bounding boxes from YOLO's bus output and uses them as prompts for SAM2.
- **Manual mode** (`input_source=[{"points": [[300, 500]], "labels": [1]}, ...]`): Uses manually defined points for the first frame. From the second frame onward, tracks automatically using previous centroids.

**Data written to the bus** (`frame_data["sam2"]`):

| Key | Type | Description |
|-----|------|-------------|
| `masks` | list[numpy array] | Binary mask per slot |
| `centroids` | list[tuple] | Center `(x, y)` of each mask |
| `areas` | list[float] | Area in pixels of each mask |
| `scores` | list[float] | SAM2 confidence score for each mask |

---

## 6. VisualizationProcessor

Visualization layer that draws on the frame. Reads data from any namespace on the bus.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_keys` | `{}` | Dict mapping namespaces to visualization elements. E.g., `{"yolo": ["boxes"], "sam2": ["masks", "centroids"]}` |
| `show_masks` | `False` | Draw masks with semi-transparent color and border |
| `show_boxes` | `False` | Draw bounding boxes |
| `show_trajectories` | `False` | Draw movement trails from centroids |
| `show_keypoints` | `False` | Draw keypoints |
| `show_centroids` | `True` | Draw centroid markers |
| `trail_length` | `30` | Number of frames to keep in trajectory history |
| `box_thickness` | `2` | Bounding box line thickness |
| `font_scale` | `0.6` | Text label size |
| `mask_alpha` | `0.35` | Mask transparency (0 = invisible, 1 = solid) |
| `mask_border_thickness` | `2` | Mask outline thickness |
| `trail_thickness` | `2` | Trajectory line thickness |

Each identity slot is assigned a fixed color automatically. Colors remain consistent across frames.

---

## 7. Creating a custom processor

Any class with a `process(self, frame_data) -> dict` method works as a pipeline layer. No base class inheritance required.

### Minimal example

```python
class ObjectCounterProcessor:

    def __init__(self, input_key="yolo"):
        self.input_key = input_key

    def process(self, frame_data):
        count = 0
        ns = frame_data.get(self.input_key)
        if ns is not None:
            boxes = ns.get('boxes')
            if boxes is not None:
                count = len(boxes)

        frame_data['metadata']['object_count'] = count
        return frame_data
```

Usage:

```python
from core import ProcessMedia, YOLOProcessor
from core.example_layer import ObjectCounterProcessor

pipeline = ProcessMedia(
    source="data/input/video_2s.mp4",
    processors=[
        YOLOProcessor(model="models/my_model.pt", confidence=0.5),
        ObjectCounterProcessor(input_key="yolo"),
    ],
    output=["json"]
)
pipeline.run()
```

### What `frame_data` contains

The dictionary your processor receives has this structure:

```python
{
    'frame': numpy_array,        # Current BGR image (OpenCV)
    'frame_index': int,          # Frame number
    'metadata': {},              # Dict for data that goes into the JSONL
    'previous_data': dict,       # Previous frame's data (without frame/vis_frame)
    'yolo': { ... },             # YOLOProcessor data (if in the pipeline)
    'sam2': { ... },             # SAM2Processor data (if in the pipeline)
    'vis_frame': numpy_array,    # Visualized frame (if VisualizationProcessor ran before)
}
```

Namespaces (`yolo`, `sam2`, etc.) only exist if the corresponding processor is in the pipeline and has already executed.

### Things to keep in mind

1. **Always return `frame_data`.** If your `process()` method doesn't return a dict, the pipeline raises an error.

2. **Read from the bus using namespaces.** YOLO data is not at `frame_data['boxes']` but at `frame_data['yolo']['boxes']`. Always use the namespace of the processor that produced the data.

3. **Don't delete other processors' data.** Add your results without removing existing ones. Downstream processors may need them.

4. **Write to `metadata` for export.** Only data in `frame_data['metadata']` is saved to the JSONL output. Put counts, metrics, flags, etc. there.

5. **Avoid storing large arrays in `metadata`.** The pipeline automatically excludes large numpy arrays from JSONL, but it's better not to put them there. Use only scalars, tuples, and short lists.

6. **The `validate()` method is optional.** If implemented, it receives the list of previous processors. Useful for warning if a required detector is missing upstream.

7. **`previous_data` holds the previous frame's data** but without `frame` or `vis_frame` (cleaned to save RAM). You can use it for inter-frame logic.

8. **Processor order matters.** A processor can only read data from processors that appear before it in the list.

---

## 8. Project structure

```
yolov8_sam2_toolkit/
├── core/
│   ├── __init__.py              # Exports: ProcessMedia, YOLOProcessor, SAM2Processor, VisualizationProcessor
│   ├── pipeline.py              # Main pipeline engine (ProcessMedia)
│   ├── yolo_processor.py        # YOLOv8 detection layer
│   ├── sam_processor.py         # SAM 2.1 segmentation layer
│   ├── visualization.py         # Visualization layer
│   └── example_layer.py         # Example custom layer
├── tracking/
│   ├── identity_matcher.py      # Identity matching between frames by centroids
│   ├── mask_utils.py            # Mask IoU and duplicate filtering
│   └── trajectory_tracker.py    # Position history for trajectories
├── setup/
│   └── setup_all.py             # Interactive installer (PyTorch + SAM2 + OpenH264)
├── examples/
│   ├── test_yolo.py             # YOLO-only test with tracking
│   ├── test_yolo_tracking.py    # Dedicated YOLO tracking test
│   ├── test_sam2.py             # SAM2-only test (manual points)
│   ├── test_united.py           # Full pipeline test (YOLO + SAM2 + Vis)
│   └── test_empty.py            # Pipeline test with no processors
├── data/
│   ├── input/                   # Input videos and images
│   └── output/                  # Results (video + JSONL + frames)
├── models/                      # YOLO model weights (.pt)
├── segment-anything-2/          # SAM2 repository (cloned by setup_all.py)
├── utils/
│   ├── tracking_utils.py        # Box geometry helpers for tracking
│   └── model_classes.json       # YOLO model classes (auto-generated)
├── requirements.txt             # Python dependencies (without PyTorch)
├── environment.yml              # Conda environment (alternative)
└── README.md
```

---

## 9. Supported hardware

| Hardware | OS | Status | Notes |
|----------|-----|--------|-------|
| NVIDIA GPU | Windows / Linux | Supported | CUDA 12.1 (recommended) |
| AMD GPU | Linux | Supported | Requires ROCm drivers |
| AMD GPU | Windows | Partial | Runs in CPU mode |
| Apple Silicon | macOS | Supported | Uses MPS acceleration |
| CPU | Any | Supported | Functional but slow for SAM2 |

---

## 10. Windows video playback

Output videos use the **mp4v** codec by default. Some Windows media players may not open these files. To fix this, run `python setup/setup_all.py` and select **yes** when asked to download the OpenH264 DLL. This enables H.264 output that plays natively on any Windows player. Alternatively, install [VLC](https://www.videolan.org/) which plays all formats.

---

## License

MIT. YOLOv8 (Ultralytics) and SAM 2 (Meta) libraries have their own licenses.
