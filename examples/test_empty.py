import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ProcessMedia

# Way to use
pipeline = ProcessMedia(
    source="data/input/frame_00001.jpg",
    processors=[],
    output=["video", "json", "frames"]
)

pipeline.run()