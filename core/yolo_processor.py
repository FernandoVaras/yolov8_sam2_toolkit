import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

class YOLOProcessor:
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
        output_key="yolo"
    ):
        # MODEL LOADING
        self.model_path = model
        self.model = YOLO(model)
        
        # NAMESPACE
        self.output_key = output_key
        
        # FILTERING PARAMETERS
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
        
        # ADAPTIVE AREA (calculated on first frame)
        self.adaptive_area_min = None
        self.adaptive_area_max = None
        
        # INITIALIZATION
        self._save_model_classes()
        self._validate_params()

    # ==========================================================
    # PIPELINE VALIDATION
    # ==========================================================
    def validate(self, previous_processors):
        """
        Checks for conflicts in the pipeline. 
        Since YOLO is usually the first layer, we only warn if there's another YOLO before.
        """
        yolo_count = sum(1 for p in previous_processors if isinstance(p, YOLOProcessor))
        if yolo_count > 0:
            print(f"[INFO] YOLOProcessor: Already {yolo_count} YOLO layers before this one.")

    def _validate_params(self):
        """Internal validation of parameter logic"""
        if self.entities is not None and (self.max_entities is not None or self.min_entities is not None):
            raise ValueError("Cannot use 'entities' with 'max_entities' or 'min_entities'")
        if self.area is not None and (self.area_max is not None or self.area_min is not None):
            raise ValueError("Cannot use 'area' with 'area_max' or 'area_min'")

    # ==========================================================
    # MAIN PROCESSING
    # ==========================================================
    def process(self, frame_data):
        """Runs YOLO and applies all configured filters"""
        frame = frame_data['frame']
        
        # 1. RUN INFERENCE
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        if len(results.boxes) == 0:
            # If nothing detected, initialize with empty structures
            frame_data[self.output_key] = {
                'boxes': np.array([]),
                'classes': np.array([]),
                'labels': [],
                'confidences': np.array([]),
                'masks': [],
                'keypoints': []
            }
            return frame_data

        # 2. INITIAL EXTRACTION
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        # Masks and Keypoints (if model supports them)
        masks = [m.cpu().numpy() for m in results.masks.data] if results.masks is not None else []
        keypoints = [k.cpu().numpy() for k in results.keypoints.data] if hasattr(results, 'keypoints') and results.keypoints is not None else []

        # 3. ADAPTIVE AREA LOGIC
        if self.area is not None and self.adaptive_area_min is None:
            if len(boxes) > 0:
                first_area = self._calculate_area(boxes[0])
                self.adaptive_area_min = first_area * (1 - self.area_error)
                self.adaptive_area_max = first_area * (1 + self.area_error)
                print(f"[YOLO] Adaptive area set: {self.adaptive_area_min:.0f}-{self.adaptive_area_max:.0f} pixels")

        # 4. APPLY ALL FILTERS
        boxes, classes, confidences, masks, keypoints = self._filter_detections(
            boxes, classes, confidences, masks, keypoints, frame.shape
        )
        boxes, classes, confidences, masks, keypoints = self._apply_nms(
            boxes, classes, confidences, masks, keypoints
        )
        boxes, classes, confidences, masks, keypoints = self._adjust_quantity(
            boxes, classes, confidences, masks, keypoints, frame, self.confidence
        )

        # 5. GENERATE LABELS FROM CLASS IDs
        labels = [self.model.names[int(cls)] for cls in classes] if len(classes) > 0 else []

        # 6. STORE RESULTS IN NAMESPACED KEY
        frame_data[self.output_key] = {
            'boxes': boxes,
            'classes': classes,
            'labels': labels,
            'confidences': confidences,
            'masks': masks,
            'keypoints': keypoints
        }
        
        # 7. ADD METADATA (using dynamic key name)
        if 'metadata' not in frame_data:
            frame_data['metadata'] = {}
        
        frame_data['metadata'][f'{self.output_key}_info'] = {
            'model': self.model_path,
            'detections_count': len(boxes)
        }
        
        return frame_data

    # ==========================================================
    # CALCULATION AND VALIDATION METHODS
    # ==========================================================
    def _calculate_area(self, box):
        """Calculate box area"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_near_edge(self, box, frame_shape):
        """Check if box is near frame edge"""
        if self.edge_margin == 0:
            return False
        
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        return (x1 < self.edge_margin or 
                y1 < self.edge_margin or 
                x2 > w - self.edge_margin or 
                y2 > h - self.edge_margin)
    
    def _is_in_roi(self, box):
        """Check if box center is in ROI"""
        if self.roi is None:
            return True
        
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        box_x1, box_y1, box_x2, box_y2 = box
        
        # Box center must be in ROI
        center_x = (box_x1 + box_x2) / 2
        center_y = (box_y1 + box_y2) / 2
        
        return (roi_x1 <= center_x <= roi_x2 and 
                roi_y1 <= center_y <= roi_y2)

    def _save_model_classes(self):
        """Save model classes to JSON for visualization"""
        try:
            class_names = self.model.names
            utils_dir = Path(__file__).parent.parent / "utils"
            utils_dir.mkdir(exist_ok=True)
            
            output_path = utils_dir / "model_classes.json"
            with open(output_path, 'w') as f:
                json.dump(class_names, f, indent=2)
            
            print(f"[YOLO] Model classes saved: {output_path}")
        except Exception as e:
            print(f"[WARNING] YOLO: Could not save classes: {e}")

    # ==========================================================
    # DETECTION FILTERING
    # ==========================================================
    def _filter_detections(self, boxes, classes, confidences, masks, keypoints, frame_shape):
        """Apply all filters to detections and keep corresponding masks/keypoints"""
        if len(boxes) == 0:
            return boxes, classes, confidences, masks, keypoints
        
        valid_indices = []
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            # Filter by class
            if self.classes is not None and cls not in self.classes:
                continue
            
            if self.exclude_classes is not None and cls in self.exclude_classes:
                continue
            
            # Filter by area
            area = self._calculate_area(box)
            
            if self.area_max is not None and area > self.area_max:
                continue
            
            if self.area_min is not None and area < self.area_min:
                continue
            
            # Adaptive area (from frame 0)
            if self.adaptive_area_min is not None:
                if area < self.adaptive_area_min or area > self.adaptive_area_max:
                    continue
            
            # Filter by edge margin
            if self._is_near_edge(box, frame_shape):
                continue
            
            # Filter by ROI
            if not self._is_in_roi(box):
                continue
            
            valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([]), np.array([]), [], []
        
        # Filter boxes, classes, confidences
        boxes = boxes[valid_indices]
        classes = classes[valid_indices]
        confidences = confidences[valid_indices]
        
        # Filter masks if they exist
        filtered_masks = []
        if masks is not None and len(masks) > 0:
            filtered_masks = [masks[i] for i in valid_indices]
        
        # Filter keypoints if they exist
        filtered_keypoints = []
        if keypoints is not None and len(keypoints) > 0:
            filtered_keypoints = [keypoints[i] for i in valid_indices]
        
        return boxes, classes, confidences, filtered_masks, filtered_keypoints
    
    def _apply_nms(self, boxes, classes, confidences, masks, keypoints):
        """Apply additional NMS if max_overlap is configured"""
        if self.max_overlap is None or len(boxes) == 0:
            return boxes, classes, confidences, masks, keypoints
        
        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        
        keep_indices = []
        
        for i in sorted_indices:
            # Check overlap with kept boxes
            should_keep = True
            for j in keep_indices:
                iou = self._calculate_iou(boxes[i], boxes[j])
                if iou > self.max_overlap:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
        
        # Filter all data
        boxes = boxes[keep_indices]
        classes = classes[keep_indices]
        confidences = confidences[keep_indices]
        
        filtered_masks = []
        if masks is not None and len(masks) > 0:
            filtered_masks = [masks[i] for i in keep_indices]
        
        filtered_keypoints = []
        if keypoints is not None and len(keypoints) > 0:
            filtered_keypoints = [keypoints[i] for i in keep_indices]
        
        return boxes, classes, confidences, filtered_masks, filtered_keypoints
    
    def _adjust_quantity(self, boxes, classes, confidences, masks, keypoints, frame, current_confidence):
        """Adjust number of detections based on max/min/entities"""
        n_detections = len(boxes)
        
        # Fixed entities: take top N by confidence
        if self.entities is not None:
            if n_detections >= self.entities:
                sorted_indices = np.argsort(confidences)[::-1][:self.entities]
                
                filtered_masks = []
                if masks is not None and len(masks) > 0:
                    filtered_masks = [masks[i] for i in sorted_indices]
                
                filtered_keypoints = []
                if keypoints is not None and len(keypoints) > 0:
                    filtered_keypoints = [keypoints[i] for i in sorted_indices]
                
                return boxes[sorted_indices], classes[sorted_indices], confidences[sorted_indices], filtered_masks, filtered_keypoints
            else:
                return boxes, classes, confidences, masks, keypoints
        
        # Max entities: take top N
        if self.max_entities is not None and n_detections > self.max_entities:
            sorted_indices = np.argsort(confidences)[::-1][:self.max_entities]
            
            filtered_masks = []
            if masks is not None and len(masks) > 0:
                filtered_masks = [masks[i] for i in sorted_indices]
            
            filtered_keypoints = []
            if keypoints is not None and len(keypoints) > 0:
                filtered_keypoints = [keypoints[i] for i in sorted_indices]
            
            return boxes[sorted_indices], classes[sorted_indices], confidences[sorted_indices], filtered_masks, filtered_keypoints
        
        # Min entities: lower threshold if needed
        if self.min_entities is not None and n_detections < self.min_entities:
            new_confidence = max(0.1, current_confidence - 0.05)
            
            if new_confidence < current_confidence:
                print(f"[YOLO] Lowering confidence {current_confidence:.2f} -> {new_confidence:.2f} to meet min_entities={self.min_entities}")
                
                # Re-run detection with lower confidence
                results = self.model(frame, conf=new_confidence, verbose=False)[0]
                
                if len(results.boxes) > 0:
                    new_boxes = results.boxes.xyxy.cpu().numpy()
                    new_classes = results.boxes.cls.cpu().numpy().astype(int)
                    new_confidences = results.boxes.conf.cpu().numpy()
                    
                    # Extract masks and keypoints
                    new_masks = []
                    if results.masks is not None:
                        new_masks = [results.masks.data[i].cpu().numpy() for i in range(len(results.boxes))]
                    
                    new_keypoints = []
                    if hasattr(results, 'keypoints') and results.keypoints is not None:
                        new_keypoints = [results.keypoints.data[i].cpu().numpy() for i in range(len(results.boxes))]
                    
                    # Re-apply filters
                    new_boxes, new_classes, new_confidences, new_masks, new_keypoints = self._filter_detections(
                        new_boxes, new_classes, new_confidences, new_masks, new_keypoints, frame.shape
                    )
                    new_boxes, new_classes, new_confidences, new_masks, new_keypoints = self._apply_nms(
                        new_boxes, new_classes, new_confidences, new_masks, new_keypoints
                    )
                    
                    # Check if we now have enough
                    if len(new_boxes) >= self.min_entities:
                        return new_boxes, new_classes, new_confidences, new_masks, new_keypoints
                    elif len(new_boxes) > n_detections:
                        return self._adjust_quantity(new_boxes, new_classes, new_confidences, new_masks, new_keypoints, frame, new_confidence)
        
        return boxes, classes, confidences, masks, keypoints
        