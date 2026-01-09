from collections import deque
from typing import Dict, List, Optional, Tuple


class TrajectoryTracker:
    """
    Maintains trajectory history for tracked entities using fixed identity slots.
    Stores position history and provides trajectory data for visualization.
    """
    
    def __init__(self, max_length: int = 30):
        """
        Args:
            max_length: Maximum number of points to keep in trajectory history
        """
        self.history: Dict[int, deque] = {}
        self.max_length = max_length
    
    def update(self, slot_idx: int, centroid: Optional[Tuple[float, float]]):
        """
        Update trajectory history for a specific slot.
        
        Args:
            slot_idx: Fixed identity slot index
            centroid: (x, y) position or None if entity not detected
        """
        if slot_idx not in self.history:
            self.history[slot_idx] = deque(maxlen=self.max_length)
        
        if centroid is not None and centroid != (0, 0):
            self.history[slot_idx].append(centroid)
        else:
            self.history[slot_idx].append(None)
    
    def get_trajectory(self, slot_idx: int) -> List[Optional[Tuple[float, float]]]:
        """
        Get trajectory history for a specific slot.
        
        Args:
            slot_idx: Fixed identity slot index
            
        Returns:
            List of (x, y) positions, with None for missing frames
        """
        return list(self.history.get(slot_idx, []))
    
    def get_all_trajectories(self) -> Dict[int, List[Optional[Tuple[float, float]]]]:
        """
        Get all trajectory histories.
        
        Returns:
            Dict mapping slot_idx to trajectory list
        """
        return {slot_idx: list(history) for slot_idx, history in self.history.items()}
    
    def reset(self):
        """Clear all trajectory history."""
        self.history.clear()
    
    def __len__(self):
        """Return number of tracked slots."""
        return len(self.history)