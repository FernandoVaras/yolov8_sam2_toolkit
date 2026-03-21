import cv2
import gc
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Keys to exclude from JSONL serialization (heavy arrays)
_EXCLUDE_KEYS = {'frame', 'vis_frame', 'previous_data'}
_EXCLUDE_NUMPY = True  # Strip all numpy arrays from JSONL output


class ProcessMedia:
    def __init__(
        self,
        source,
        processors=None,
        output="video",
        output_dir="data/output",
        use_vis_frame=True,
        start_frame=0,
        end_frame=None
    ):
        self.source_path = Path(source)
        self.processors = processors if processors is not None else []
        self.output_dir = Path(output_dir)
        self.output_types = [output] if isinstance(output, str) else output
        self.use_vis_frame = use_vis_frame
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.is_image = self.source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = self.output_dir / self.timestamp
        self.output_base = self.source_path.stem

        self.fps = 1
        self.width = None
        self.height = None
        self.total_frames = 1

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._validate_pipeline()

    def _validate_pipeline(self):
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source not found: {self.source_path}")

        if len(self.processors) == 0:
            print("[WARNING] No processors provided. Will copy media only.")

        for i, proc in enumerate(self.processors):
            if hasattr(proc, 'validate'):
                proc.validate(self.processors[:i])

    def _get_output_frame(self, frame_data):
        """Get the appropriate frame for output (vis_frame or original frame)"""
        if self.use_vis_frame and 'vis_frame' in frame_data:
            return frame_data['vis_frame']
        return frame_data['frame']

    # ------------------------------------------------------------------
    # JSONL streaming helpers
    # ------------------------------------------------------------------
    def _make_jsonl_entry(self, frame_data):
        """Extract only scalar/serializable data from frame_data for JSONL."""
        entry = {}
        for k, v in frame_data.items():
            if k in _EXCLUDE_KEYS:
                continue
            entry[k] = self._sanitize_value(v)
        return entry

    def _sanitize_value(self, obj):
        """Recursively convert to JSON-safe types, stripping large numpy arrays."""
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._sanitize_value(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            sanitized = [self._sanitize_value(i) for i in obj]
            return sanitized
        if isinstance(obj, np.ndarray):
            if _EXCLUDE_NUMPY and obj.size > 100:
                return None
            return obj.tolist()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return str(obj)

    def _write_jsonl_line(self, f, entry):
        """Write a single JSONL line."""
        f.write(json.dumps(entry, separators=(',', ':')) + '\n')

    def _write_jsonl_header(self, f):
        """Write info header as the first JSONL line."""
        header = {
            '_type': 'header',
            'source': str(self.source_path),
            'media_type': 'image' if self.is_image else 'video',
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps if not self.is_image else None,
            'total_frames': self.total_frames,
            'timestamp': self.timestamp,
            'layers': [p.__class__.__name__ for p in self.processors]
        }
        self._write_jsonl_line(f, header)

    # ------------------------------------------------------------------
    # Frame sharding for output frames
    # ------------------------------------------------------------------
    def _get_frame_shard_path(self, frames_dir, frame_count):
        """Get sharded subdirectory path: frames/001000/frame_00001.jpg"""
        shard = ((frame_count // 1000) + 1) * 1000
        shard_dir = frames_dir / f"{shard:06d}"
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"frame_{frame_count:05d}.jpg"

    # ------------------------------------------------------------------
    # Resource monitor and heartbeat
    # ------------------------------------------------------------------
    def _heartbeat(self, frame_count, loop_start_time):
        """Every 100 frames: clear VRAM, log resources, show ETA."""
        # VRAM cleanup
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - loop_start_time
        fps_avg = frame_count / elapsed if elapsed > 0 else 0
        remaining = self.total_frames - frame_count
        eta_seconds = remaining / fps_avg if fps_avg > 0 else 0
        eta_min = eta_seconds / 60

        parts = [f"[MONITOR] Frame {frame_count}/{self.total_frames}"]
        parts.append(f"FPS: {fps_avg:.1f}")
        parts.append(f"ETA: {eta_min:.1f}min")

        if HAS_PSUTIL:
            ram = psutil.virtual_memory()
            parts.append(f"RAM: {ram.percent:.0f}%")

        if HAS_TORCH and torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            parts.append(f"VRAM: {vram_gb:.2f}GB")

        print(" | ".join(parts))

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def _process_frame_logic(self, frame, frame_count, previous_data=None):
        frame_data = {
            'frame': frame,
            'frame_index': frame_count,
            'metadata': {},
            'previous_data': previous_data
        }

        for processor in self.processors:
            result = processor.process(frame_data)

            if not isinstance(result, dict):
                raise TypeError(
                    f"Processor '{processor.__class__.__name__}' must return a dict, "
                    f"but returned {type(result).__name__}. "
                    f"Ensure your processor's process() method returns the frame_data dictionary."
                )

            frame_data = result

        return frame_data

    def _prune_previous_data(self, current_data):
        """Remove heavy data before storing as previous_data."""
        current_data['frame'] = None
        current_data['vis_frame'] = None
        return current_data

    def run(self):
        print("=" * 60)
        print(f"ProcessMedia - Starting")
        print("=" * 60)
        print(f"Input: {self.source_path}")
        print(f"Type: {'Image' if self.is_image else 'Video'}")
        print(f"Processors: {[p.__class__.__name__ for p in self.processors]}")
        print(f"Output types: {self.output_types}")
        print(f"Use vis_frame: {self.use_vis_frame}")
        print(f"Output folder: {self.output_folder}")
        if not self.is_image:
            print(f"Frame range: {self.start_frame} -> {self.end_frame or 'end'}")
        print("=" * 60)

        # Open JSONL file for streaming if json output requested
        jsonl_file = None
        jsonl_path = None
        if "json" in self.output_types:
            jsonl_path = self.output_folder / f"{self.output_base}.jsonl"

        if self.is_image:
            frame = cv2.imread(str(self.source_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {self.source_path}")

            self.height, self.width = frame.shape[:2]
            print(f"[INFO] Image: {self.width}x{self.height}")

            print(f"[INFO] Processing image...")
            result = self._process_frame_logic(frame, 0, None)

            output_frame = self._get_output_frame(result)

            output_image_path = self.output_folder / f"{self.output_base}_processed.jpg"
            cv2.imwrite(str(output_image_path), output_frame)
            print(f"[SAVED] Image: {output_image_path}")

            if "frames" in self.output_types:
                f_dir = self.output_folder / "frames"
                f_dir.mkdir(exist_ok=True)
                frame_path = f_dir / "frame_00000.jpg"
                cv2.imwrite(str(frame_path), output_frame)
                print(f"[SAVED] Frame: {frame_path}")

            if jsonl_path:
                with open(jsonl_path, 'w') as f:
                    self._write_jsonl_header(f)
                    entry = self._make_jsonl_entry(result)
                    self._write_jsonl_line(f, entry)
                print(f"[SAVED] JSONL: {jsonl_path}")

        else:
            cap = cv2.VideoCapture(str(self.source_path))

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.source_path}")

            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Apply end_frame limit
            effective_end = self.end_frame if self.end_frame is not None else self.total_frames
            effective_end = min(effective_end, self.total_frames)
            effective_total = effective_end - self.start_frame

            if self.total_frames > 50000:
                print(f"[WARNING] Large video detected: {self.total_frames} frames. Ensure sufficient RAM/VRAM.")

            print(f"[INFO] Video: {self.width}x{self.height} @ {self.fps} FPS, {self.total_frames} frames")
            if self.start_frame > 0 or self.end_frame is not None:
                print(f"[INFO] Processing range: {self.start_frame}-{effective_end} ({effective_total} frames)")

            video_writer = None
            video_path = None
            if "video" in self.output_types:
                video_path = self.output_folder / f"{self.output_base}.mp4"
                # Use H.264 if OpenH264 DLL is available, otherwise fall back to mp4v
                h264_dll = Path(__file__).resolve().parent.parent / "openh264-1.8.0-win64.dll"
                if h264_dll.exists():
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, self.fps, (self.width, self.height)
                )

            frames_dir = None
            if "frames" in self.output_types:
                frames_dir = self.output_folder / "frames"
                frames_dir.mkdir(exist_ok=True)

            # Skip to start_frame
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            frame_count = self.start_frame
            previous_data = None
            loop_start_time = time.time()

            # Open JSONL for streaming writes
            if jsonl_path:
                jsonl_file = open(jsonl_path, 'w')
                self._write_jsonl_header(jsonl_file)

            try:
                while frame_count < effective_end:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_data = self._process_frame_logic(frame, frame_count, previous_data)

                    output_frame = self._get_output_frame(current_data)

                    if video_writer:
                        video_writer.write(output_frame)

                    if frames_dir:
                        frame_path = self._get_frame_shard_path(frames_dir, frame_count)
                        cv2.imwrite(str(frame_path), output_frame)

                    # Stream JSONL entry to disk immediately
                    if jsonl_file:
                        entry = self._make_jsonl_entry(current_data)
                        self._write_jsonl_line(jsonl_file, entry)

                    frame_count += 1

                    # Prune heavy data before storing as previous_data
                    previous_data = self._prune_previous_data(current_data)

                    # Heartbeat every 100 frames
                    frames_processed = frame_count - self.start_frame
                    if frames_processed % 100 == 0:
                        self._heartbeat(frames_processed, loop_start_time)
                    elif frames_processed % 10 == 0:
                        progress = (frames_processed / effective_total) * 100 if effective_total > 0 else 100
                        print(f"[INFO] Processing frame {frame_count}/{effective_end} ({progress:.1f}%)")

            finally:
                cap.release()
                if video_writer:
                    video_writer.release()
                if jsonl_file:
                    jsonl_file.close()

            print("\n" + "=" * 60)
            print("Saving outputs...")
            print("=" * 60)

            if video_path:
                print(f"[SAVED] Video: {video_path}")

            if frames_dir:
                total_written = frame_count - self.start_frame
                print(f"[SAVED] Frames: {frames_dir} ({total_written} frames)")

            if jsonl_path:
                print(f"[SAVED] JSONL: {jsonl_path}")

        print("=" * 60)
        print(f"[SUCCESS] Processing complete")
        print(f"[SUCCESS] Output folder: {self.output_folder}")
        print("=" * 60)
