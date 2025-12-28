import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia, SAM2Processor
from utils import VisualizationProcessor

# Configuración de los puntos
mis_ratas = [
    {
        "points": [[320, 810]],             # Rata 1: 
        "labels": [1]
    },
    {
        "points": [[500, 770]],             # Rata 2: 
        "labels": [1]
    }
]

pipeline = ProcessMedia(
    source="data/input/video_2s.mp4",
    processors=[
        SAM2Processor(
            input_source=mis_ratas,
            model_type="tiny",
            max_entities=2
        ),
        VisualizationProcessor(
            input_keys={"sam2": ["masks", "scores"]},
            show_masks=True,
            show_boxes=True,
            show_trajectories=True,
            show_keypoints=True,
            show_centroids=True,
        )
    ],
    output=["video", "frames"]
)

pipeline.run()


