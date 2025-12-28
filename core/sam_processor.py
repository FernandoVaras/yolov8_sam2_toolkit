import sys
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

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
    def __init__(
        self,
        model_type="large",
        input_source=None,
        output_key="sam2",
        max_entities=2,
        proximity_threshold=50,
        area_tolerance=0.4,
        iou_threshold=0.5
    ):
        """
        SAM 2 Processor with fixed identity slot tracking.
        
        Args:
            model_type: "large" or "tiny"
            input_source: "yolo:boxes" (bus mode) or [{"points": [[x, y]], "labels": [1]}] (manual mode)
            output_key: Output namespace in frame_data
            max_entities: Maximum number of tracked objects
            proximity_threshold: Maximum distance (px) for identity matching
            area_tolerance: Allowed area variation (0.4 = ±40%)
            iou_threshold: Threshold to remove duplicate masks
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_key = output_key
        self.input_source = input_source
        self.max_entities = max_entities
        self.proximity_threshold = proximity_threshold
        self.area_tolerance = area_tolerance
        self.iou_threshold = iou_threshold
        
        self.prev_centroids = [None] * max_entities
        self.prev_areas = [None] * max_entities
        self.prev_masks = [None] * max_entities
        self.first_frame_done = False
        
        self.is_bus_mode = isinstance(input_source, str)
        self.is_manual_mode = isinstance(input_source, list)
        
        if input_source is None:
            raise ValueError(
                "SAM2Processor: input_source cannot be None for initialization.\n"
                "You must provide either:\n"
                "  - Bus mode: input_source='yolo:boxes'\n"
                "  - Manual mode: input_source=[{'points': [[x, y]], 'labels': [1]}]"
            )
        
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

    def _get_centroid(self, mask):
        y, x = np.where(mask)
        if len(x) == 0:
            return None
        return (np.mean(x), np.mean(y))

    def _calculate_iou(self, m1, m2):
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return intersection / union if union > 0 else 0

    def _read_from_bus(self, frame_data):
        if not self.is_bus_mode:
            return []
        
        try:
            parts = self.input_source.split(':')
            if len(parts) != 2:
                print(f"[WARNING] SAM2: Invalid bus path '{self.input_source}'. Expected format 'namespace:key'")
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

    def _filter_duplicates(self, masks, scores):
        if len(masks) <= self.max_entities:
            return masks, scores

        areas = [m.sum() for m in masks]
        sorted_idx = np.argsort(areas)[::-1]
        ordered_masks = [masks[i] for i in sorted_idx]
        ordered_scores = [scores[i] for i in sorted_idx]
        
        unique_masks = []
        unique_scores = []
        
        for m_curr, score_curr in zip(ordered_masks, ordered_scores):
            is_dup = False
            for m_uniq in unique_masks:
                if self._calculate_iou(m_curr, m_uniq) > self.iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique_masks.append(m_curr)
                unique_scores.append(score_curr)
                if len(unique_masks) >= self.max_entities:
                    break
        
        return unique_masks, unique_scores

    def _match_tracking(self, current_masks, current_scores):
        if not current_masks:
            return (
                [None] * self.max_entities,
                [None] * self.max_entities,
                [None] * self.max_entities,
                [0.0] * self.max_entities
            )

        curr_centroids = []
        curr_areas = []
        for m in current_masks:
            c = self._get_centroid(m)
            curr_centroids.append(c if c else (0, 0))
            curr_areas.append(m.sum())

        if all(c is None for c in self.prev_centroids):
            matched_masks = [None] * self.max_entities
            matched_centroids = [None] * self.max_entities
            matched_areas = [None] * self.max_entities
            matched_scores = [0.0] * self.max_entities
            
            for i, (mask, centroid, area, score) in enumerate(zip(
                current_masks, curr_centroids, curr_areas, current_scores
            )):
                if i < self.max_entities:
                    matched_masks[i] = mask
                    matched_centroids[i] = centroid
                    matched_areas[i] = area
                    matched_scores[i] = score
            
            return matched_masks, matched_centroids, matched_areas, matched_scores

        valid_prev_indices = [i for i, c in enumerate(self.prev_centroids) if c is not None]
        
        if not valid_prev_indices:
            matched_masks = [None] * self.max_entities
            matched_centroids = [None] * self.max_entities
            matched_areas = [None] * self.max_entities
            matched_scores = [0.0] * self.max_entities
            
            for i, (mask, centroid, area, score) in enumerate(zip(
                current_masks, curr_centroids, curr_areas, current_scores
            )):
                if i < self.max_entities:
                    matched_masks[i] = mask
                    matched_centroids[i] = centroid
                    matched_areas[i] = area
                    matched_scores[i] = score
            
            return matched_masks, matched_centroids, matched_areas, matched_scores
        
        valid_prev_centroids = [self.prev_centroids[i] for i in valid_prev_indices]
        dist_matrix = cdist(curr_centroids, valid_prev_centroids)
        
        matched_masks = [None] * self.max_entities
        matched_centroids = [None] * self.max_entities
        matched_areas = [None] * self.max_entities
        matched_scores = [0.0] * self.max_entities
        used_curr_idx = []

        for matrix_col_idx, slot_idx in enumerate(valid_prev_indices):
            dists = dist_matrix[:, matrix_col_idx]
            area_ref = self.prev_areas[slot_idx]
            
            for i_curr in np.argsort(dists):
                if i_curr in used_curr_idx:
                    continue
                
                dist_ok = dists[i_curr] < self.proximity_threshold
                var_area = abs(curr_areas[i_curr] - area_ref) / area_ref if area_ref and area_ref > 0 else 1.0
                area_ok = var_area <= self.area_tolerance

                if dist_ok and area_ok:
                    matched_masks[slot_idx] = current_masks[i_curr]
                    matched_centroids[slot_idx] = curr_centroids[i_curr]
                    matched_areas[slot_idx] = curr_areas[i_curr]
                    matched_scores[slot_idx] = current_scores[i_curr]
                    used_curr_idx.append(i_curr)
                    break

        for i_curr, mask in enumerate(current_masks):
            if i_curr not in used_curr_idx:
                for slot_idx in range(self.max_entities):
                    if matched_masks[slot_idx] is None:
                        matched_masks[slot_idx] = mask
                        matched_centroids[slot_idx] = curr_centroids[i_curr]
                        matched_areas[slot_idx] = curr_areas[i_curr]
                        matched_scores[slot_idx] = current_scores[i_curr]
                        break

        return matched_masks, matched_centroids, matched_areas, matched_scores

    def process(self, frame_data):
        frame = frame_data['frame']
        h, w = frame.shape[:2]
        self.predictor.set_image(frame)
        all_masks = []
        all_scores = []
        
        if not self.first_frame_done and self.is_manual_mode:
            print(f"[SAM2] Initializing with {len(self.input_source)} manual prompts")
            
            for idx, prompt in enumerate(self.input_source):
                if idx >= self.max_entities:
                    print(f"[SAM2] Skipping prompt {idx+1}: exceeds max_entities={self.max_entities}")
                    break
                    
                pts = np.array(prompt["points"])
                lbls = np.array(prompt.get("labels", [1] * len(pts)))
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    all_masks.append(masks[0])
                    all_scores.append(float(scores[0]))
                    print(f"[SAM2] Slot {idx}: Initialized (score: {scores[0]:.3f})")
            
            self.first_frame_done = True
        
        elif self.first_frame_done and self.is_manual_mode:
            active_slots = [i for i, c in enumerate(self.prev_centroids) if c is not None]
            
            for slot_idx in active_slots:
                pt = self.prev_centroids[slot_idx]
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                if len(masks) > 0:
                    all_masks.append(masks[0])
                    all_scores.append(float(scores[0]))
        
        elif self.is_bus_mode:
            boxes = self._read_from_bus(frame_data)
            
            if len(boxes) > 0:
                print(f"[SAM2] Using {len(boxes)} boxes from bus '{self.input_source}'")
                for box in boxes[:self.max_entities]:
                    masks, scores, _ = self.predictor.predict(
                        box=np.array(box),
                        multimask_output=False
                    )
                    if len(masks) > 0:
                        all_masks.append(masks[0])
                        all_scores.append(float(scores[0]))
            
            elif any(c is not None for c in self.prev_centroids):
                active_slots = [i for i, c in enumerate(self.prev_centroids) if c is not None]
                print(f"[SAM2] Bus empty, auto-tracking {len(active_slots)} slots")
                
                for slot_idx in active_slots:
                    pt = self.prev_centroids[slot_idx]
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([pt]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    if len(masks) > 0:
                        all_masks.append(masks[0])
                        all_scores.append(float(scores[0]))
            else:
                print(f"[WARNING] SAM2: No input available on Frame 0. Cannot initialize tracking.")
        
        else:
            print(f"[WARNING] SAM2: Unexpected state - no valid input mode detected")
        
        unique_masks, unique_scores = self._filter_duplicates(all_masks, all_scores)
        matched_masks, matched_centroids, matched_areas, matched_scores = self._match_tracking(
            unique_masks, unique_scores
        )

        self.prev_centroids = matched_centroids
        self.prev_areas = matched_areas
        self.prev_masks = matched_masks

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

        frame_data[self.output_key] = {
            'masks': output_masks,
            'centroids': output_centroids,
            'areas': output_areas,
            'scores': output_scores
        }
        
        if 'metadata' not in frame_data:
            frame_data['metadata'] = {}
        
        active_entities = sum(1 for m in matched_masks if m is not None)
        frame_data['metadata'][f'{self.output_key}_info'] = {
            'entities_active': active_entities,
            'max_entities': self.max_entities,
            'proximity_threshold': self.proximity_threshold,
            'area_tolerance': self.area_tolerance,
            'mode': 'manual_init' if (not self.first_frame_done and self.is_manual_mode) 
                    else 'auto_tracking' if self.first_frame_done and self.is_manual_mode 
                    else 'bus' if self.is_bus_mode else 'unknown'
        }

        return frame_data

    def reset_tracking(self):
        self.prev_centroids = [None] * self.max_entities
        self.prev_areas = [None] * self.max_entities
        self.prev_masks = [None] * self.max_entities
        self.first_frame_done = False
        print("[SAM2] Tracking state reset")
