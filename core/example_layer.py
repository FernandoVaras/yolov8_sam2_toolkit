class ObjectCounterProcessor:
    """
    Example custom layer: counts detected objects and writes the total to metadata.
    Reads from a configurable namespace on the data bus.
    """

    def __init__(self, input_key="yolo", target_label="Objects"):
        self.input_key = input_key
        self.target_label = target_label

    def validate(self, previous_processors):
        """Warn if no detector is found before this layer."""
        has_detector = any(hasattr(p, 'process') for p in previous_processors)
        if not has_detector:
            print(f"[WARNING] {self.__class__.__name__}: No detectors found. Count will be 0.")

    def process(self, frame_data):
        """Count boxes from the configured namespace and store in metadata."""
        count = 0
        ns = frame_data.get(self.input_key)
        if ns is not None:
            boxes = ns.get('boxes')
            if boxes is not None:
                count = len(boxes)

        frame_data['metadata']['object_count'] = count
        return frame_data
