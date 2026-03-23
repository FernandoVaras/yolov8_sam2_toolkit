import sys
import torch
import numpy as np
from pathlib import Path
from tracking.identity_matcher import IdentityMatcher
from tracking.mask_utils import filter_duplicates

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent
SAM2_DIR = PROJECT_ROOT / "segment-anything-2"

if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"[ERROR] SAM 2 not found at {SAM2_DIR}")
    print(f"[ERROR] Make sure you have run: python setup/setup_all.py")
    raise ImportError(f"SAM 2 import failed: {e}")


class SAM2Processor:
    """SAM 2 Processor with fixed identity slot tracking."""

    def __init__(
        self,
        model_type="large",
        input_source=None,
        output_key="sam2",
        max_entities=2,
        proximity_threshold=50,
        area_tolerance=0.10,
        iou_threshold=0.5
    ):
        """
        Args:
            model_type: "large" or "tiny"
            input_source: "yolo:boxes" (bus mode) or [{"points": [[x, y]], "labels": [1]}] (manual mode)
            output_key: Output namespace in frame_data
            max_entities: Maximum number of tracked objects
            proximity_threshold: Maximum distance (px) for identity matching
            area_tolerance: Allowed area variation (0.4 = +/-40%)
            iou_threshold: Threshold to remove duplicate masks
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_autocast = self.device.type == 'cuda'
        self.output_key = output_key
        self.input_source = input_source
        self.max_entities = max_entities
        self.iou_threshold = iou_threshold

        # Identity tracking
        self.matcher = IdentityMatcher(
            max_entities=max_entities,
            proximity_threshold=proximity_threshold,
            area_tolerance=area_tolerance
        )
        self.first_frame_done = False

        # Input mode detection
        self.is_bus_mode = isinstance(input_source, str)
        self.is_manual_mode = isinstance(input_source, list)

        if input_source is None:
            raise ValueError(
                "SAM2Processor: input_source cannot be None.\n"
                "Provide either:\n"
                "  - Bus mode: input_source='yolo:boxes'\n"
                "  - Manual mode: input_source=[{'points': [[x, y]], 'labels': [1]}]"
            )

        # Model initialization
        ckpt_path = SAM2_DIR / f"checkpoints/sam2.1_hiera_{model_type}.pt"
        config_name = f"configs/sam2.1/sam2.1_hiera_{'l' if model_type == 'large' else 't'}.yaml"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found at: {ckpt_path}\n"
                f"Run 'python setup/setup_all.py' to download weights."
            )

        print(f"[SAM2] Initializing model ({model_type}) on {self.device}")
        print(f"[SAM2] Checkpoint: {ckpt_path}")
        print(f"[SAM2] Input mode: {'Bus' if self.is_bus_mode else 'Manual' if self.is_manual_mode else 'Auto-tracking'}")
        print(f"[SAM2] Fixed identity slots: {max_entities}")

        self.model = build_sam2(config_name, str(ckpt_path), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    # ------------------------------------------------------------------
    # Bus reading
    # ------------------------------------------------------------------

    def _read_from_bus(self, frame_data):
        """Read bounding boxes from the data bus."""
        if not self.is_bus_mode:
            return []

        try:
            parts = self.input_source.split(':')
            if len(parts) != 2:
                print(f"[WARNING] SAM2: Invalid bus path '{self.input_source}'. Expected 'namespace:key'")
                return []

            namespace, key = parts
            if namespace not in frame_data:
                return []

            data = frame_data[namespace].get(key, [])
            if hasattr(data, '__len__') and len(data) == 0:
                return []

            return data if isinstance(data, (list, np.ndarray)) else []
        except Exception as e:
            print(f"[WARNING] SAM2: Error reading from bus '{self.input_source}': {e}")
            return []

    # ------------------------------------------------------------------
    # Segmentation strategies
    # ------------------------------------------------------------------

    def _segment_from_manual_prompts(self):
        """Segment using manual point prompts (first frame initialization)."""
        all_masks = []
        all_scores = []

        print(f"[SAM2] Initializing with {len(self.input_source)} manual prompts")

        for idx, prompt in enumerate(self.input_source):
            if idx >= self.max_entities:
                print(f"[SAM2] Skipping prompt {idx+1}: exceeds max_entities={self.max_entities}")
                break

            pts = np.array(prompt["points"])
            lbls = np.array(prompt.get("labels", [1] * len(pts)))

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast):
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts, point_labels=lbls, multimask_output=False
                )

            if len(masks) > 0:
                all_masks.append(masks[0])
                all_scores.append(float(scores[0]))
                print(f"[SAM2] Slot {idx}: Initialized (score: {scores[0]:.3f})")

        self.first_frame_done = True
        return all_masks, all_scores

    def _segment_from_centroids(self):
        """Segment using previous frame centroids (auto-tracking)."""
        all_masks = []
        all_scores = []

        active_slots = [i for i, c in enumerate(self.matcher.prev_centroids) if c is not None]

        for slot_idx in active_slots:
            pt = self.matcher.prev_centroids[slot_idx]

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast):
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([pt]), point_labels=np.array([1]),
                    multimask_output=False
                )

            if len(masks) > 0:
                all_masks.append(masks[0])
                all_scores.append(float(scores[0]))

        return all_masks, all_scores

    def _segment_from_boxes(self, boxes):
        """Segment using bounding boxes from bus."""
        all_masks = []
        all_scores = []

        print(f"[SAM2] Using {len(boxes)} boxes from bus '{self.input_source}'")

        for box in boxes[:self.max_entities]:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast):
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box), multimask_output=False
                )

            if len(masks) > 0:
                all_masks.append(masks[0])
                all_scores.append(float(scores[0]))

        return all_masks, all_scores

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _format_output(self, matched_data, frame_shape):
        """Format matched data into consistent output structure."""
        matched_masks, matched_centroids, matched_areas, matched_scores = matched_data
        h, w = frame_shape[:2]

        output_masks = []
        output_centroids = []
        output_areas = []
        output_scores = []

        for slot_idx in range(self.max_entities):
            if matched_masks[slot_idx] is not None:
                output_masks.append(matched_masks[slot_idx])
                output_centroids.append(matched_centroids[slot_idx])
                output_areas.append(matched_areas[slot_idx])
                output_scores.append(matched_scores[slot_idx])
            else:
                output_masks.append(np.zeros((h, w), dtype=bool))
                output_centroids.append((0, 0))
                output_areas.append(0)
                output_scores.append(0.0)

        return output_masks, output_centroids, output_areas, output_scores

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, frame_data):
        """Run SAM2 segmentation with identity tracking on a single frame."""
        frame = frame_data['frame']

        # Compute image embeddings
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast):
            self.predictor.set_image(frame)

        # Select segmentation strategy
        all_masks = []
        all_scores = []

        if not self.first_frame_done and self.is_manual_mode:
            all_masks, all_scores = self._segment_from_manual_prompts()

        elif self.first_frame_done and self.is_manual_mode:
            all_masks, all_scores = self._segment_from_centroids()

        elif self.is_bus_mode:
            boxes = self._read_from_bus(frame_data)

            if len(boxes) > 0:
                all_masks, all_scores = self._segment_from_boxes(boxes)
            elif any(c is not None for c in self.matcher.prev_centroids):
                active_slots = [i for i, c in enumerate(self.matcher.prev_centroids) if c is not None]
                print(f"[SAM2] Bus empty, auto-tracking {len(active_slots)} slots")
                all_masks, all_scores = self._segment_from_centroids()
            else:
                print(f"[WARNING] SAM2: No input available on Frame 0. Cannot initialize tracking.")
        else:
            print(f"[WARNING] SAM2: Unexpected state - no valid input mode detected")

        # Filter duplicates and match to identity slots
        unique_masks, unique_scores = filter_duplicates(
            all_masks, all_scores, self.max_entities, self.iou_threshold
        )
        matched_data = self.matcher.match(unique_masks, unique_scores)

        # Format output
        output_masks, output_centroids, output_areas, output_scores = self._format_output(
            matched_data, frame.shape
        )

        frame_data[self.output_key] = {
            'masks': output_masks,
            'centroids': output_centroids,
            'areas': output_areas,
            'scores': output_scores
        }

        # Metadata
        if 'metadata' not in frame_data:
            frame_data['metadata'] = {}

        active_entities = sum(1 for m in matched_data[0] if m is not None)
        frame_data['metadata'][f'{self.output_key}_info'] = {
            'entities_active': active_entities,
            'max_entities': self.max_entities,
            'proximity_threshold': self.matcher.proximity_threshold,
            'area_tolerance': self.matcher.area_tolerance,
            'mode': 'manual_init' if (not self.first_frame_done and self.is_manual_mode)
                    else 'auto_tracking' if self.first_frame_done and self.is_manual_mode
                    else 'bus' if self.is_bus_mode else 'unknown'
        }

        return frame_data

    def reset_tracking(self):
        """Reset tracking state to initial conditions."""
        self.matcher.reset()
        self.first_frame_done = False
        print("[SAM2] Tracking state reset")
