import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YOLOProcessor:
    """YOLOv8 detection layer with configurable filtering, NMS, and adaptive retries."""

    def __init__(
        self,
        model="yolov8n.pt",
        confidence=0.5,
        max_entities=None,
        min_entities=None,
        entities=None,
        area_max=None,
        area_min=None,
        area=None,
        area_error=0.2,
        classes=None,
        exclude_classes=None,
        max_overlap=None,
        edge_margin=0,
        roi=None,
        output_key="yolo",
        use_tracking=False,
        tracking_proximity=80,
        tracking_area_tolerance=0.5
    ):
        # Model
        self.model_path = model
        self.model = YOLO(model)
        self._device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._use_autocast = self._device_type == 'cuda'

        # Namespace
        self.output_key = output_key

        # Filtering parameters
        self.confidence = confidence
        self.max_entities = max_entities
        self.min_entities = min_entities
        self.entities = entities
        self.area_max = area_max
        self.area_min = area_min
        self.area = area
        self.area_error = area_error
        self.classes = classes
        self.exclude_classes = exclude_classes
        self.max_overlap = max_overlap
        self.edge_margin = edge_margin
        self.roi = roi

        # Adaptive area (calculated on first frame)
        self.adaptive_area_min = None
        self.adaptive_area_max = None

        # Identity tracking
        self.use_tracking = use_tracking
        if self.use_tracking:
            from tracking import IdentityMatcher
            self.identity_matcher = IdentityMatcher(
                max_entities=self.max_entities or self.entities or 10,
                proximity_threshold=tracking_proximity,
                area_tolerance=tracking_area_tolerance
            )
        else:
            self.identity_matcher = None

        self._save_model_classes()
        self._validate_params()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, previous_processors):
        """Warn if there is already a YOLO layer before this one."""
        yolo_count = sum(1 for p in previous_processors if isinstance(p, YOLOProcessor))
        if yolo_count > 0:
            print(f"[INFO] YOLOProcessor: Already {yolo_count} YOLO layers before this one.")

    def _validate_params(self):
        """Check for conflicting parameter combinations."""
        if self.entities is not None and (self.max_entities is not None or self.min_entities is not None):
            raise ValueError("Cannot use 'entities' with 'max_entities' or 'min_entities'")
        if self.area is not None and (self.area_max is not None or self.area_min is not None):
            raise ValueError("Cannot use 'area' with 'area_max' or 'area_min'")

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, frame_data):
        """Run YOLO inference and apply all configured filters."""
        frame = frame_data['frame']

        # 1. Inference
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self._use_autocast):
            results = self.model(frame, conf=self.confidence, verbose=False)[0]

        if len(results.boxes) == 0:
            frame_data[self.output_key] = {
                'boxes': np.array([]),
                'classes': np.array([]),
                'labels': [],
                'confidences': np.array([]),
                'masks': [],
                'keypoints': []
            }
            return frame_data

        # 2. Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        masks = [m.cpu().numpy() for m in results.masks.data] if results.masks is not None else []
        keypoints = (
            [k.cpu().numpy() for k in results.keypoints.data]
            if hasattr(results, 'keypoints') and results.keypoints is not None
            else []
        )

        # 3. Adaptive area (first frame)
        if self.area is not None and self.adaptive_area_min is None and len(boxes) > 0:
            first_area = self._calculate_area(boxes[0])
            self.adaptive_area_min = first_area * (1 - self.area_error)
            self.adaptive_area_max = first_area * (1 + self.area_error)
            print(f"[YOLO] Adaptive area set: {self.adaptive_area_min:.0f}-{self.adaptive_area_max:.0f} pixels")

        # 4. Filter -> NMS -> Quantity adjustment
        boxes, classes, confidences, masks, keypoints = self._filter_detections(
            boxes, classes, confidences, masks, keypoints, frame.shape
        )
        boxes, classes, confidences, masks, keypoints = self._apply_nms(
            boxes, classes, confidences, masks, keypoints
        )
        boxes, classes, confidences, masks, keypoints = self._adjust_quantity(
            boxes, classes, confidences, masks, keypoints, frame, self.confidence
        )

        # 5. Labels
        labels = [self.model.names[int(cls)] for cls in classes] if len(classes) > 0 else []

        # 6. Identity tracking: reorder all data by persistent slot assignment
        if self.use_tracking and self.identity_matcher is not None and len(boxes) > 0:
            from tracking import boxes_to_centroids, boxes_to_areas, reorder_by_slots

            centroids = boxes_to_centroids(boxes)
            areas = boxes_to_areas(boxes)

            slot_indices, matched_centroids, matched_areas, matched_scores = \
                self.identity_matcher.match_from_data(
                    centroids=centroids,
                    areas=areas,
                    scores=list(confidences)
                )

            max_e = self.identity_matcher.max_entities
            boxes       = reorder_by_slots(list(boxes),       slot_indices, max_e, empty_value=None)
            classes     = reorder_by_slots(list(classes),     slot_indices, max_e, empty_value=None)
            confidences = reorder_by_slots(list(confidences), slot_indices, max_e, empty_value=0.0)
            labels      = reorder_by_slots(labels,            slot_indices, max_e, empty_value="")
            if masks:
                masks = reorder_by_slots(masks, slot_indices, max_e, empty_value=None)
            if keypoints:
                keypoints = reorder_by_slots(keypoints, slot_indices, max_e, empty_value=None)

            # Clean Nones to keep downstream compatibility
            boxes       = [x for x in boxes if x is not None]
            classes     = [x for x in classes if x is not None]
            confidences = [c for c in confidences if c != 0.0]
            labels      = [l for l in labels if l != ""]
            if masks:
                masks = [x for x in masks if x is not None]
            if keypoints:
                keypoints = [x for x in keypoints if x is not None]

        # 7. Write to bus
        frame_data[self.output_key] = {
            'boxes': boxes,
            'classes': classes,
            'labels': labels,
            'confidences': confidences,
            'masks': masks,
            'keypoints': keypoints
        }

        # 7. Metadata
        if 'metadata' not in frame_data:
            frame_data['metadata'] = {}
        frame_data['metadata'][f'{self.output_key}_info'] = {
            'model': self.model_path,
            'detections_count': len(boxes)
        }

        return frame_data

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _calculate_area(self, box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _is_near_edge(self, box, frame_shape):
        if self.edge_margin == 0:
            return False
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        return (x1 < self.edge_margin or y1 < self.edge_margin or
                x2 > w - self.edge_margin or y2 > h - self.edge_margin)

    def _is_in_roi(self, box):
        if self.roi is None:
            return True
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2

    def _save_model_classes(self):
        """Save model class names to JSON for external tools."""
        try:
            utils_dir = Path(__file__).parent.parent / "utils"
            utils_dir.mkdir(exist_ok=True)
            output_path = utils_dir / "model_classes.json"
            with open(output_path, 'w') as f:
                json.dump(self.model.names, f, indent=2)
            print(f"[YOLO] Model classes saved: {output_path}")
        except Exception as e:
            print(f"[WARNING] YOLO: Could not save classes: {e}")

    # ------------------------------------------------------------------
    # Detection filtering
    # ------------------------------------------------------------------

    def _filter_detections(self, boxes, classes, confidences, masks, keypoints, frame_shape):
        """Apply class, area, edge, and ROI filters."""
        if len(boxes) == 0:
            return boxes, classes, confidences, masks, keypoints

        valid_indices = []

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if self.classes is not None and cls not in self.classes:
                continue
            if self.exclude_classes is not None and cls in self.exclude_classes:
                continue

            area = self._calculate_area(box)
            if self.area_max is not None and area > self.area_max:
                continue
            if self.area_min is not None and area < self.area_min:
                continue
            if self.adaptive_area_min is not None:
                if area < self.adaptive_area_min or area > self.adaptive_area_max:
                    continue

            if self._is_near_edge(box, frame_shape):
                continue
            if not self._is_in_roi(box):
                continue

            valid_indices.append(i)

        if len(valid_indices) == 0:
            return np.array([]), np.array([]), np.array([]), [], []

        boxes = boxes[valid_indices]
        classes = classes[valid_indices]
        confidences = confidences[valid_indices]
        filtered_masks = [masks[i] for i in valid_indices] if masks and len(masks) > 0 else []
        filtered_keypoints = [keypoints[i] for i in valid_indices] if keypoints and len(keypoints) > 0 else []

        return boxes, classes, confidences, filtered_masks, filtered_keypoints

    def _apply_nms(self, boxes, classes, confidences, masks, keypoints):
        """Apply additional NMS if max_overlap is configured."""
        if self.max_overlap is None or len(boxes) == 0:
            return boxes, classes, confidences, masks, keypoints

        sorted_indices = np.argsort(confidences)[::-1]
        keep_indices = []

        for i in sorted_indices:
            should_keep = True
            for j in keep_indices:
                if self._calculate_iou(boxes[i], boxes[j]) > self.max_overlap:
                    should_keep = False
                    break
            if should_keep:
                keep_indices.append(i)

        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        confidences = confidences[keep_indices]
        filtered_masks = [masks[i] for i in keep_indices] if masks and len(masks) > 0 else []
        filtered_keypoints = [keypoints[i] for i in keep_indices] if keypoints and len(keypoints) > 0 else []

        return boxes, classes, confidences, filtered_masks, filtered_keypoints

    # ------------------------------------------------------------------
    # Quantity adjustment
    # ------------------------------------------------------------------

    def _select_top_n(self, n, boxes, classes, confidences, masks, keypoints):
        """Select top N detections by confidence."""
        sorted_indices = np.argsort(confidences)[::-1][:n]
        filtered_masks = [masks[i] for i in sorted_indices] if masks and len(masks) > 0 else []
        filtered_keypoints = [keypoints[i] for i in sorted_indices] if keypoints and len(keypoints) > 0 else []
        return (boxes[sorted_indices], classes[sorted_indices], confidences[sorted_indices],
                filtered_masks, filtered_keypoints)

    def _adjust_quantity(self, boxes, classes, confidences, masks, keypoints, frame, current_confidence):
        """Adjust detection count. Max 2 retries for min_entities, no recursion."""
        n_detections = len(boxes)

        # Fixed entities: take top N
        if self.entities is not None:
            if n_detections >= self.entities:
                return self._select_top_n(self.entities, boxes, classes, confidences, masks, keypoints)
            return boxes, classes, confidences, masks, keypoints

        # Max entities: take top N
        if self.max_entities is not None and n_detections > self.max_entities:
            return self._select_top_n(self.max_entities, boxes, classes, confidences, masks, keypoints)

        # Min entities: lower threshold with max 2 retries
        if self.min_entities is not None and n_detections < self.min_entities:
            retry_confidences = [max(0.1, current_confidence - 0.10), 0.15]

            for attempt, new_confidence in enumerate(retry_confidences):
                if new_confidence >= current_confidence:
                    continue

                print(f"[YOLO] Retry {attempt+1}/2: confidence {current_confidence:.2f} -> {new_confidence:.2f}"
                      f" to meet min_entities={self.min_entities}")

                with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self._use_autocast):
                    results = self.model(frame, conf=new_confidence, verbose=False)[0]

                if len(results.boxes) == 0:
                    continue

                new_boxes = results.boxes.xyxy.cpu().numpy()
                new_classes = results.boxes.cls.cpu().numpy().astype(int)
                new_confidences = results.boxes.conf.cpu().numpy()
                new_masks = (
                    [results.masks.data[i].cpu().numpy() for i in range(len(results.boxes))]
                    if results.masks is not None else []
                )
                new_keypoints = (
                    [results.keypoints.data[i].cpu().numpy() for i in range(len(results.boxes))]
                    if hasattr(results, 'keypoints') and results.keypoints is not None else []
                )

                new_boxes, new_classes, new_confidences, new_masks, new_keypoints = self._filter_detections(
                    new_boxes, new_classes, new_confidences, new_masks, new_keypoints, frame.shape
                )
                new_boxes, new_classes, new_confidences, new_masks, new_keypoints = self._apply_nms(
                    new_boxes, new_classes, new_confidences, new_masks, new_keypoints
                )

                if len(new_boxes) >= self.min_entities:
                    return new_boxes, new_classes, new_confidences, new_masks, new_keypoints

                if len(new_boxes) > n_detections:
                    boxes, classes, confidences = new_boxes, new_classes, new_confidences
                    masks, keypoints = new_masks, new_keypoints
                    n_detections = len(new_boxes)

        return boxes, classes, confidences, masks, keypoints

    # ------------------------------------------------------------------
    # Tracking reset
    # ------------------------------------------------------------------

    def reset_tracking(self):
        """
        Reset the IdentityMatcher state between videos.
        Call this when processing a new video/sequence to avoid
        matching frame 0 of the new video against the last frame
        of the previous one.
        """
        if self.identity_matcher is not None:
            self.identity_matcher.reset()
