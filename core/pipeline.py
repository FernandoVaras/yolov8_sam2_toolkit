import cv2
import json
import os
from datetime import datetime
from pathlib import Path

class ProcessMedia:
    def __init__(self, source, processors=None, output="video", output_dir="data/output", use_vis_frame=True):
        self.source_path = Path(source)
        self.processors = processors if processors is not None else []
        self.output_dir = Path(output_dir)
        self.output_types = [output] if isinstance(output, str) else output
        self.use_vis_frame = use_vis_frame
        
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

    def _export_json(self, log):
        output_path = self.output_folder / f"{self.output_base}.json"
        
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj

        clean_log = make_serializable(log)

        final_data = {
            'info': {
                'source': str(self.source_path),
                'type': 'image' if self.is_image else 'video',
                'resolution': f"{self.width}x{self.height}",
                'fps': self.fps if not self.is_image else None,
                'total_frames': self.total_frames,
                'timestamp': self.timestamp,
                'layers': [p.__class__.__name__ for p in self.processors]
            },
            'results': clean_log
        }

        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"[SAVED] JSON: {output_path}")

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

    def run(self):
        print("="*60)
        print(f"ProcessMedia - Starting")
        print("="*60)
        print(f"Input: {self.source_path}")
        print(f"Type: {'Image' if self.is_image else 'Video'}")
        print(f"Processors: {[p.__class__.__name__ for p in self.processors]}")
        print(f"Output types: {self.output_types}")
        print(f"Use vis_frame: {self.use_vis_frame}")
        print(f"Output folder: {self.output_folder}")
        print("="*60)

        metadata_log = []

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

            json_entry = {k: v for k, v in result.items() if k not in ['frame', 'vis_frame', 'previous_data']}
            metadata_log.append(json_entry)

        else:
            cap = cv2.VideoCapture(str(self.source_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.source_path}")
            
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[INFO] Video: {self.width}x{self.height} @ {self.fps} FPS, {self.total_frames} frames")

            video_writer = None
            video_path = None
            if "video" in self.output_types:
                video_path = self.output_folder / f"{self.output_base}.mp4"
                video_writer = cv2.VideoWriter(
                    str(video_path), 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    self.fps, 
                    (self.width, self.height)
                )

            frames_dir = None
            if "frames" in self.output_types:
                frames_dir = self.output_folder / "frames"
                frames_dir.mkdir(exist_ok=True)

            frame_count = 0
            previous_data = None

            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                
                current_data = self._process_frame_logic(frame, frame_count, previous_data)
                
                output_frame = self._get_output_frame(current_data)
                
                if video_writer:
                    video_writer.write(output_frame)
                
                if frames_dir:
                    cv2.imwrite(str(frames_dir / f"frame_{frame_count:05d}.jpg"), output_frame)

                json_entry = {k: v for k, v in current_data.items() if k not in ['frame', 'vis_frame', 'previous_data']}
                metadata_log.append(json_entry)

                frame_count += 1
                previous_data = current_data
                
                if frame_count % 10 == 0 or frame_count == self.total_frames:
                    progress = (frame_count / self.total_frames) * 100
                    print(f"[INFO] Processing frame {frame_count}/{self.total_frames} ({progress:.1f}%)")

            cap.release()
            if video_writer: 
                video_writer.release()
            
            print("\n" + "="*60)
            print("Saving outputs...")
            print("="*60)
            
            if video_path:
                print(f"[SAVED] Video: {video_path}")
            
            if frames_dir:
                print(f"[SAVED] Frames: {frames_dir} ({frame_count} frames)")

        if "json" in self.output_types:
            self._export_json(metadata_log)

        print("="*60)
        print(f"[SUCCESS] Processing complete")
        print(f"[SUCCESS] Output folder: {self.output_folder}")
        print("="*60)