import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist


class IdentityMatcher:
    """
    Matches detected entities to fixed identity slots across frames
    using proximity and area-based heuristics.
    """
    
    def __init__(
        self,
        max_entities: int,
        proximity_threshold: float = 50.0,
        area_tolerance: float = 0.4
    ):
        """
        Args:
            max_entities: Maximum number of tracked slots
            proximity_threshold: Maximum centroid distance (px) for matching
            area_tolerance: Allowed area variation (0.4 = ±40%)
        """
        self.max_entities = max_entities
        self.proximity_threshold = proximity_threshold
        self.area_tolerance = area_tolerance
        
        # Tracking state (only scalars — no masks stored)
        self.prev_centroids = [None] * max_entities
        self.prev_areas = [None] * max_entities
    
    def reset(self):
        """Clear all tracking history."""
        self.prev_centroids = [None] * self.max_entities
        self.prev_areas = [None] * self.max_entities
    
    def match(
        self,
        current_masks: List[np.ndarray],
        current_scores: List[float]
    ) -> Tuple[List, List, List, List]:
        """
        Match current detections to fixed identity slots.
        
        Args:
            current_masks: List of binary masks from current frame
            current_scores: Confidence scores for each mask
            
        Returns:
            Tuple of (matched_masks, matched_centroids, matched_areas, matched_scores)
            Each is a list of length max_entities with None for inactive slots
        """
        if not current_masks:
            return self._empty_slots()
        
        curr_centroids = [self._get_centroid(m) for m in current_masks]
        curr_areas = [m.sum() for m in current_masks]
        
        # First frame: assign sequentially
        if all(c is None for c in self.prev_centroids):
            matched = self._initialize_slots(
                current_masks, curr_centroids, curr_areas, current_scores
            )
        else:
            # Match to previous frame
            matched = self._match_to_previous(
                current_masks, curr_centroids, curr_areas, current_scores
            )
        
        # Update state (only scalars)
        self.prev_centroids = matched[1]
        self.prev_areas = matched[2]
        
        return matched
    
    def _empty_slots(self) -> Tuple[List, List, List, List]:
        """Return empty slot structure."""
        return (
            [None] * self.max_entities,
            [None] * self.max_entities,
            [None] * self.max_entities,
            [0.0] * self.max_entities
        )
    
    def _initialize_slots(
        self,
        masks: List[np.ndarray],
        centroids: List[Tuple],
        areas: List[float],
        scores: List[float]
    ) -> Tuple[List, List, List, List]:
        """Initialize slots on first frame."""
        matched_masks = [None] * self.max_entities
        matched_centroids = [None] * self.max_entities
        matched_areas = [None] * self.max_entities
        matched_scores = [0.0] * self.max_entities
        
        for i, (mask, centroid, area, score) in enumerate(
            zip(masks, centroids, areas, scores)
        ):
            if i < self.max_entities:
                matched_masks[i] = mask
                matched_centroids[i] = centroid
                matched_areas[i] = area
                matched_scores[i] = score
        
        return matched_masks, matched_centroids, matched_areas, matched_scores
    
    def _match_to_previous(
        self,
        curr_masks: List[np.ndarray],
        curr_centroids: List[Tuple],
        curr_areas: List[float],
        curr_scores: List[float]
    ) -> Tuple[List, List, List, List]:
        """Match current detections to previous slots."""
        valid_prev_indices = [
            i for i, c in enumerate(self.prev_centroids) if c is not None
        ]
        
        if not valid_prev_indices:
            return self._initialize_slots(
                curr_masks, curr_centroids, curr_areas, curr_scores
            )
        
        # Compute distance matrix
        valid_prev_centroids = [self.prev_centroids[i] for i in valid_prev_indices]
        dist_matrix = cdist(curr_centroids, valid_prev_centroids)
        
        # Initialize output
        matched_masks = [None] * self.max_entities
        matched_centroids = [None] * self.max_entities
        matched_areas = [None] * self.max_entities
        matched_scores = [0.0] * self.max_entities
        used_curr_idx = []
        
        # Match to existing slots
        for matrix_col_idx, slot_idx in enumerate(valid_prev_indices):
            dists = dist_matrix[:, matrix_col_idx]
            area_ref = self.prev_areas[slot_idx]
            
            for i_curr in np.argsort(dists):
                if i_curr in used_curr_idx:
                    continue
                
                # Check proximity
                dist_ok = dists[i_curr] < self.proximity_threshold
                
                # Check area consistency
                if area_ref and area_ref > 0:
                    var_area = abs(curr_areas[i_curr] - area_ref) / area_ref
                else:
                    var_area = 1.0
                area_ok = var_area <= self.area_tolerance
                
                if dist_ok and area_ok:
                    matched_masks[slot_idx] = curr_masks[i_curr]
                    matched_centroids[slot_idx] = curr_centroids[i_curr]
                    matched_areas[slot_idx] = curr_areas[i_curr]
                    matched_scores[slot_idx] = curr_scores[i_curr]
                    used_curr_idx.append(i_curr)
                    break
        
        # Assign unmatched detections to empty slots
        for i_curr, mask in enumerate(curr_masks):
            if i_curr not in used_curr_idx:
                for slot_idx in range(self.max_entities):
                    if matched_masks[slot_idx] is None:
                        matched_masks[slot_idx] = mask
                        matched_centroids[slot_idx] = curr_centroids[i_curr]
                        matched_areas[slot_idx] = curr_areas[i_curr]
                        matched_scores[slot_idx] = curr_scores[i_curr]
                        break
        
        return matched_masks, matched_centroids, matched_areas, matched_scores
    
    @staticmethod
    def _get_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate centroid of binary mask."""
        y, x = np.where(mask)
        if len(x) == 0:
            return None
        return (np.mean(x), np.mean(y))

    # ------------------------------------------------------------------
    # Agnostic matching (for YOLO boxes, no masks needed)
    # ------------------------------------------------------------------

    def match_from_data(
        self,
        centroids: List[Optional[Tuple[float, float]]],
        areas: List[float],
        scores: Optional[List[float]] = None
    ) -> Tuple[List, List, List, List]:
        """
        Agnostic matcher: receives centroids and areas directly (no masks).
        Used by YOLOProcessor. The original match() for SAM2 is untouched.

        Args:
            centroids: List of (cx, cy) tuples, one per detection
            areas:     List of areas, one per detection
            scores:    List of scores/confidences (optional, defaults to 1.0)

        Returns:
            Tuple of 4 lists of length max_entities:
            (slot_indices, matched_centroids, matched_areas, matched_scores)
            slot_indices[i] = original detection index assigned to slot i (or None)
        """
        n = len(centroids)
        if scores is None:
            scores = [1.0] * n

        if n == 0:
            return self._empty_slots_from_data()

        # First frame: initialize slots sequentially
        if all(c is None for c in self.prev_centroids):
            slot_indices = list(range(min(n, self.max_entities)))
            slot_indices += [None] * (self.max_entities - len(slot_indices))

            matched_centroids = [None] * self.max_entities
            matched_areas     = [0.0]  * self.max_entities
            matched_scores    = [0.0]  * self.max_entities

            for det_idx in range(min(n, self.max_entities)):
                matched_centroids[det_idx] = centroids[det_idx]
                matched_areas[det_idx]     = areas[det_idx]
                matched_scores[det_idx]    = scores[det_idx]

            self.prev_centroids = matched_centroids[:]
            self.prev_areas     = matched_areas[:]
            return slot_indices, matched_centroids, matched_areas, matched_scores

        # Subsequent frames: match by proximity + area
        current_centroids_arr = np.array([c for c in centroids if c is not None])
        prev_centroids_arr    = np.array([c if c is not None else (0.0, 0.0)
                                          for c in self.prev_centroids])

        if len(current_centroids_arr) == 0:
            return self._empty_slots_from_data()

        dist_matrix = cdist(prev_centroids_arr, current_centroids_arr)

        used_detections = set()
        slot_indices       = [None] * self.max_entities
        matched_centroids  = [None] * self.max_entities
        matched_areas      = [0.0]  * self.max_entities
        matched_scores     = [0.0]  * self.max_entities

        for slot_idx in range(self.max_entities):
            if self.prev_centroids[slot_idx] is None:
                continue
            prev_area = self.prev_areas[slot_idx]

            sorted_dets = np.argsort(dist_matrix[slot_idx])
            for det_idx in sorted_dets:
                if det_idx in used_detections:
                    continue
                if dist_matrix[slot_idx][det_idx] > self.proximity_threshold:
                    break
                area_ratio = (areas[det_idx] / prev_area) if prev_area > 0 else 1.0
                if 1.0 - self.area_tolerance <= area_ratio <= 1.0 + self.area_tolerance:
                    slot_indices[slot_idx]      = int(det_idx)
                    matched_centroids[slot_idx] = centroids[det_idx]
                    matched_areas[slot_idx]     = areas[det_idx]
                    matched_scores[slot_idx]    = scores[det_idx]
                    used_detections.add(det_idx)
                    break

        # Assign remaining detections to free slots
        free_slots = [i for i in range(self.max_entities) if slot_indices[i] is None]
        remaining  = [i for i in range(n) if i not in used_detections]
        for slot_idx, det_idx in zip(free_slots, remaining):
            slot_indices[slot_idx]      = det_idx
            matched_centroids[slot_idx] = centroids[det_idx]
            matched_areas[slot_idx]     = areas[det_idx]
            matched_scores[slot_idx]    = scores[det_idx]

        self.prev_centroids = matched_centroids[:]
        self.prev_areas     = matched_areas[:]
        return slot_indices, matched_centroids, matched_areas, matched_scores

    def _empty_slots_from_data(self):
        """Empty structure for match_from_data()."""
        empty = [None] * self.max_entities
        zeros = [0.0]  * self.max_entities
        return empty, empty[:], zeros[:], zeros[:]