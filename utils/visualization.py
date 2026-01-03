import cv2
import numpy as np
from collections import deque


class VisualizationProcessor:
    
    def __init__(
        self,
        input_keys=None,
        show_masks=False,
        show_boxes=False,
        show_trajectories=False,
        show_keypoints=False,
        show_centroids=True,
        trail_length=30,
        box_thickness=2,
        font_scale=0.6,
        mask_alpha=0.35,
        mask_border_thickness=2,
        trail_thickness=2
    ):
        """
        Visualization Processor with Bus Architecture and Fixed Identity Slots.
        
        Args:
            input_keys: Dict mapping namespaces to visualization elements
                       Example: {'yolo': ['boxes'], 'sam2': ['masks', 'centroids', 'scores']}
            show_masks: Enable/disable mask visualization
            show_boxes: Enable/disable bounding box visualization
            show_trajectories: Enable/disable trajectory trails
            show_keypoints: Enable/disable keypoint visualization
            show_centroids: Enable/disable centroid markers
            trail_length: Number of frames to keep in trajectory history
            box_thickness: Thickness of bounding boxes
            font_scale: Scale factor for text labels
            mask_alpha: Transparency of mask overlay (0-1)
            mask_border_thickness: Thickness of mask outline
            trail_thickness: Thickness of trajectory lines
        """
        self.input_keys = input_keys or {}
        self.show_masks = show_masks
        self.show_boxes = show_boxes
        self.show_trajectories = show_trajectories
        self.show_keypoints = show_keypoints
        self.show_centroids = show_centroids
        self.trail_length = trail_length
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.mask_alpha = mask_alpha
        self.mask_border_thickness = mask_border_thickness
        self.trail_thickness = trail_thickness
        
        self.trajectory_history = {}
        
        self.colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
            (200, 150, 100),
            (255, 180, 100),
            (150, 200, 255),
            (180, 255, 150),
        ]
    
    def _get_color(self, slot_idx):
        return self.colors[slot_idx % len(self.colors)]
    
    def _get_from_bus(self, frame_data, namespace, key):
        if namespace not in frame_data:
            return None
        data = frame_data[namespace].get(key)
        return data if data is not None else None
    
    def _draw_masks(self, frame, masks, slot_scores=None):
        if not self.show_masks or masks is None or len(masks) == 0:
            return frame
        
        for slot_idx, mask in enumerate(masks):
            if mask is None:
                continue
            
            if isinstance(mask, np.ndarray):
                is_active = mask.sum() > 0
            else:
                continue
            
            if not is_active:
                continue
            
            score = slot_scores[slot_idx] if slot_scores is not None and slot_idx < len(slot_scores) else 0.0
            if score == 0.0:
                continue
            
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (frame.shape[1], frame.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            mask_bool = mask > 0.5 if mask.dtype == bool else mask > 0
            
            color = self._get_color(slot_idx)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask_bool] = color
            
            frame = cv2.addWeighted(frame, 1, colored_mask, self.mask_alpha, 0)
            
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color, self.mask_border_thickness)
        
        return frame
    
    def _draw_boxes(self, frame, boxes, slot_scores=None, labels=None):
        if not self.show_boxes or boxes is None:
            return frame
        
        if isinstance(boxes, np.ndarray) and len(boxes) == 0:
            return frame
        
        for slot_idx, box in enumerate(boxes):
            if box is None:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            
            score = slot_scores[slot_idx] if slot_scores is not None and slot_idx < len(slot_scores) else 1.0
            if score == 0.0:
                continue
            
            # Obtener el label del objeto
            label = labels[slot_idx] if labels is not None and slot_idx < len(labels) else None
            
            color = self._get_color(slot_idx)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            frame = self._draw_label(frame, (x1, y1), slot_idx, score, color, label)
        
        return frame
    
    def _draw_keypoints(self, frame, keypoints_list, slot_scores=None):
        if not self.show_keypoints or keypoints_list is None or len(keypoints_list) == 0:
            return frame
        
        for slot_idx, keypoints in enumerate(keypoints_list):
            if keypoints is None or len(keypoints) == 0:
                continue
            
            # Verificar si el slot tiene score válido
            if slot_scores is not None and slot_idx < len(slot_scores):
                score = slot_scores[slot_idx]
                if score == 0.0:
                    continue
            
            color = self._get_color(slot_idx)
            
            for kp in keypoints:
                if len(kp) >= 2:
                    x, y = int(kp[0]), int(kp[1])
                    confidence = kp[2] if len(kp) >= 3 else 1.0
                    
                    if confidence > 0.5:
                        cv2.circle(frame, (x, y), 5, color, -1)
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_centroids(self, frame, centroids, slot_scores=None):
        if not self.show_centroids or centroids is None:
            return frame
        
        for slot_idx, centroid in enumerate(centroids):
            if centroid is None or centroid == (0, 0):
                continue
            
            score = slot_scores[slot_idx] if slot_scores is not None and slot_idx < len(slot_scores) else 0.0
            if score == 0.0:
                continue
            
            color = self._get_color(slot_idx)
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_label(self, frame, position, slot_id, score, color, label=None):
        x, y = position
        
        # Si hay label, usarlo, sino usar ID
        if label:
            text = f"{label}"
        else:
            text = f"ID:{slot_id}"
        
        # Agregar el score/confidence
        if score > 0:
            text += f" {score:.2f}"
        
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        
        padding = 8
        bg_x1 = x
        bg_y1 = y - th - baseline - padding * 2
        bg_x2 = x + tw + padding * 2
        bg_y2 = y
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
        
        text_x = x + padding
        text_y = y - baseline - padding
        cv2.putText(
            frame, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        
        return frame
    
    def _draw_trajectories(self, frame, max_slots):
        if not self.show_trajectories:
            return frame
        
        for slot_idx in range(max_slots):
            if slot_idx not in self.trajectory_history:
                continue
            
            history = list(self.trajectory_history[slot_idx])
            if len(history) < 2:
                continue
            
            color = self._get_color(slot_idx)
            num_points = len(history)
            
            for i in range(1, num_points):
                if history[i-1] is None or history[i] is None:
                    continue
                
                alpha = i / num_points
                thickness = max(1, int(self.trail_thickness * alpha))
                trail_color = tuple(int(c * alpha) for c in color)
                
                cv2.line(
                    frame, 
                    (int(history[i-1][0]), int(history[i-1][1])),
                    (int(history[i][0]), int(history[i][1])),
                    trail_color, 
                    thickness,
                    cv2.LINE_AA
                )
        
        return frame
    
    def _update_trajectories(self, centroids):
        for slot_idx, centroid in enumerate(centroids):
            if slot_idx not in self.trajectory_history:
                self.trajectory_history[slot_idx] = deque(maxlen=self.trail_length)
            
            if centroid is not None and centroid != (0, 0):
                self.trajectory_history[slot_idx].append(centroid)
            else:
                self.trajectory_history[slot_idx].append(None)
    
    def process(self, frame_data):
        frame = frame_data['frame'].copy()
        
        max_slots = 0
        centroids_data = None
        scores_data = None
        labels_data = None
        
        for namespace, keys in self.input_keys.items():
            if 'centroids' in keys:
                centroids_data = self._get_from_bus(frame_data, namespace, 'centroids')
                if centroids_data:
                    max_slots = max(max_slots, len(centroids_data))
            
            # Buscar scores o confidences
            if 'scores' in keys:
                scores_data = self._get_from_bus(frame_data, namespace, 'scores')
            if 'confidences' in keys:
                scores_data = self._get_from_bus(frame_data, namespace, 'confidences')
            
            # Buscar labels
            if 'labels' in keys:
                labels_data = self._get_from_bus(frame_data, namespace, 'labels')
        
        if centroids_data:
            self._update_trajectories(centroids_data)
        
        for namespace, keys in self.input_keys.items():
            
            if 'masks' in keys:
                masks = self._get_from_bus(frame_data, namespace, 'masks')
                frame = self._draw_masks(frame, masks, scores_data)
            
            if 'boxes' in keys:
                boxes = self._get_from_bus(frame_data, namespace, 'boxes')
                frame = self._draw_boxes(frame, boxes, scores_data, labels_data)
            
            if 'keypoints' in keys:
                keypoints = self._get_from_bus(frame_data, namespace, 'keypoints')
                frame = self._draw_keypoints(frame, keypoints, scores_data)
        
        frame = self._draw_trajectories(frame, max_slots)
        
        if centroids_data:
            frame = self._draw_centroids(frame, centroids_data, scores_data)
        
        frame_data['vis_frame'] = frame
        
        if 'metadata' not in frame_data:
            frame_data['metadata'] = {}
        
        frame_data['metadata']['visualization'] = {
            'show_masks': self.show_masks,
            'show_boxes': self.show_boxes,
            'show_trajectories': self.show_trajectories,
            'show_keypoints': self.show_keypoints,
            'show_centroids': self.show_centroids
        }
        
        return frame_data
    
    def reset_trajectories(self):
        self.trajectory_history.clear()
        print("[VIS] Trajectory history reset")