import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia, YOLOProcessor, SAM2Processor, VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/video_10s.mp4",
    processors=[YOLOProcessor(
            model="models/yolo_8l_rat.pt",
            confidence=0.6,
            entities=2
        ), 
        SAM2Processor(
            input_source="yolo:boxes",
            model_type="large",
            max_entities=2
        ),
        VisualizationProcessor(
            input_keys={"sam2": ["masks", "scores", "centroids"]},
            show_masks=True,
            show_boxes=True,
            show_trajectories=True,
            show_keypoints=True,
            show_centroids=True,
        )
],
    output=["video", "json", "frames"]
)

pipeline.run()