"""
tracking_utils.py
Funciones utilitarias de geometria para tracking basado en bounding boxes.
Usadas por YOLOProcessor para calcular centroides y areas antes de pasar
los datos al IdentityMatcher.
"""

from typing import List, Tuple, Optional
import numpy as np


def box_to_centroid(box: List[float]) -> Tuple[float, float]:
    """
    Calcula el centroide (cx, cy) de una bounding box [x1, y1, x2, y2].

    Args:
        box: Lista o array con [x1, y1, x2, y2]

    Returns:
        Tupla (cx, cy) con el centro geometrico de la box
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def box_to_area(box: List[float]) -> float:
    """
    Calcula el area de una bounding box [x1, y1, x2, y2].

    Args:
        box: Lista o array con [x1, y1, x2, y2]

    Returns:
        Area en pixeles cuadrados
    """
    x1, y1, x2, y2 = box
    return float((x2 - x1) * (y2 - y1))


def boxes_to_centroids(boxes: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Convierte una lista de boxes a una lista de centroides.

    Args:
        boxes: Lista de boxes, cada una [x1, y1, x2, y2]

    Returns:
        Lista de tuplas (cx, cy)
    """
    return [box_to_centroid(b) for b in boxes]


def boxes_to_areas(boxes: List[List[float]]) -> List[float]:
    """
    Convierte una lista de boxes a una lista de areas.

    Args:
        boxes: Lista de boxes, cada una [x1, y1, x2, y2]

    Returns:
        Lista de areas en pixeles cuadrados
    """
    return [box_to_area(b) for b in boxes]


def reorder_by_slots(
    data_list: List,
    slot_assignments: List[Optional[int]],
    max_entities: int,
    empty_value=None
) -> List:
    """
    Reordena una lista de datos segun los slots asignados por el IdentityMatcher.

    Args:
        data_list:        Lista de datos en orden original de YOLO
        slot_assignments: Lista de slot_idx para cada deteccion (puede tener None)
        max_entities:     Numero total de slots posibles
        empty_value:      Valor a usar para slots vacios (default: None)

    Returns:
        Lista de longitud max_entities con datos en orden de slot

    Ejemplo:
        YOLO detecto 2 ratas: [rata_B, rata_A] (orden por confianza)
        slot_assignments = [1, 0]  (el matcher dice: primera va al slot 1, segunda al slot 0)
        Resultado: [rata_A, rata_B]  (slot 0 = rata_A, slot 1 = rata_B)
    """
    result = [empty_value] * max_entities
    for detection_idx, slot_idx in enumerate(slot_assignments):
        if slot_idx is not None and 0 <= slot_idx < max_entities:
            if detection_idx < len(data_list):
                result[slot_idx] = data_list[detection_idx]
    return result
