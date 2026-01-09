import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia, YOLOProcessor, VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/frame_00001.jpg",
    processors=[YOLOProcessor(
            model="models/yolo_8l_rat.pt",
            confidence=0.7,
            entities=2
        ), 
        VisualizationProcessor(
            input_keys={"yolo": ["boxes", "keypoints", "confidences", "labels"]},
            show_masks=False,
            show_boxes=True,
            show_trajectories=False,
            show_keypoints=True,
            show_centroids=False,
        )
],
    output=["video", "json", "frames"]
)

pipeline.run()