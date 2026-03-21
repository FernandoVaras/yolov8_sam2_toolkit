import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia

# Way to use
pipeline = ProcessMedia(
    source="data/input/clip_1m17s-1m27s (1).mp4",
    processors=[],
    output=["video", "json", "frames"]
)

pipeline.run()