import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia, SAM2Processor, VisualizationProcessor

# Configuración de los puntos
mis_ratas = [
    {
        "points": [[300, 500]],             # Rata 1: 
        "labels": [1]
    },
    {
        "points": [[800, 70]],             # Rata 2: 
        "labels": [1]
    }
]

pipeline = ProcessMedia(
    source="data/input/clip_1m17s-1m27s (1).mp4",
    processors=[
        SAM2Processor(
            input_source=mis_ratas,
            model_type="tiny",
            max_entities=2
        ),
        VisualizationProcessor(
            input_keys={"sam2": ["masks", "scores"]},
            show_masks=True,
            show_boxes=False,
            show_trajectories=False,
            show_keypoints=False,
            show_centroids=False,
        )
    ],
    output=["video", "frames", "json"]
)

pipeline.run()


