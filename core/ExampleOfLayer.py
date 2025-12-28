import cv2

class ObjectCounterProcessor:
    """
    Custom layer that counts how many bounding boxes were detected 
    by previous layers and draws the total on the frame.
    """
    def __init__(self, target_label="Objects"):
        # Initial configuration
        self.target_label = target_label

    # ==========================================================
    # STEP 1: VALIDATION (CHECK FOR REQUIRED DATA) - OPTIONAL
    # ==========================================================
    
    def validate(self, previous_processors):
        """
        Check if there is a detector (like YOLO) before this layer.
        """
        has_detector = any(
            hasattr(p, 'process') for p in previous_processors
        )
        if not has_detector:
            print(f"[WARNING] {self.__class__.__name__}: No detectors found in previous layers. Count will be 0.")

    # ==========================================================
    # STEP 2: PROCESSING (CORE LOGIC) - YOUR CODE HERE
    # ==========================================================

    def process(self, frame_data):
        """
        Reads the 'boxes' key and performs the counting logic.
        """
        # 1. Retrieve boxes from the context dictionary
        boxes = frame_data.get('boxes')
        
        # 2. Perform counting
        count = 0
        if boxes is not None:
            # We assume boxes is a list or a NumPy array
            count = len(boxes)
            
        # ==========================================================
        # STEP 3: UPDATE (SAVE RESULTS WITHOUT DELETING DATA) - MANDATORY
        # ==========================================================

        # 1. Add the count to the metadata (this will go into your JSON)
        frame_data['metadata']['object_count'] = count
        
        # 2. RETURN the updated dictionary to the pipeline
        return frame_data