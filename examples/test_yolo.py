import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia, YOLOProcessor, VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/video_2s.mp4",
    processors=[YOLOProcessor(
            model="models/yolo_8l_rat.pt",
            confidence=0.5,
            max_entities=2,
            max_overlap=0.9,
            min_entities=2,
            use_tracking=True,
        ),
        VisualizationProcessor(
            input_keys={"yolo": ["boxes", "keypoints", "confidences", "labels"]},
            show_masks=False,
            show_boxes=True,
            show_keypoints=True,
        )
],
    output=["video", "json", "frames"]
)

pipeline.run()