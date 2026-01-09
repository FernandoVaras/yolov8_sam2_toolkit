import numpy as np
from typing import List, Tuple


def calculate_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two binary masks.
    
    Args:
        m1: First binary mask
        m2: Second binary mask
        
    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0


def filter_duplicates(
    masks: List[np.ndarray],
    scores: List[float],
    max_entities: int,
    iou_threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Remove duplicate masks based on IoU, keeping largest ones.
    
    Args:
        masks: List of binary masks
        scores: Confidence scores for each mask
        max_entities: Maximum number of masks to keep
        iou_threshold: IoU threshold for considering masks as duplicates
        
    Returns:
        Tuple of (unique_masks, unique_scores)
    """
    if len(masks) <= max_entities:
        return masks, scores
    
    # Sort by area (largest first)
    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]
    ordered_masks = [masks[i] for i in sorted_idx]
    ordered_scores = [scores[i] for i in sorted_idx]
    
    unique_masks = []
    unique_scores = []
    
    for m_curr, score_curr in zip(ordered_masks, ordered_scores):
        is_duplicate = False
        for m_uniq in unique_masks:
            if calculate_iou(m_curr, m_uniq) > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_masks.append(m_curr)
            unique_scores.append(score_curr)
            if len(unique_masks) >= max_entities:
                break
    
    return unique_masks, unique_scores