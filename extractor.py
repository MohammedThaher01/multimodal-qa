from ultralytics import YOLO
import numpy as np

class SpatialExtractor:
    def __init__(self, model_path="yolov8m.pt"):
        # Load the YOLOv8 model into memory once when the class is initialized
        self.model = YOLO(model_path)
        
    def analyze_image(self, image_pil):
        """
        Takes a PIL image, runs YOLO inference, and returns a structured string
        detailing object counts and their spatial locations.
        """
        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image_pil)
        
        # Run inference (verbose=False keeps the terminal clean)
        results = self.model(img_array, conf=0.25, imgsz=1024, verbose=False)[0]
        
        # We need the image dimensions to calculate relative positions
        img_height, img_width = img_array.shape[:2]
        
        detected_objects = []
        counts = {}
        
        # Iterate through every bounding box YOLO found
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            # Update counts
            counts[class_name] = counts.get(class_name, 0) + 1
            
            # Get bounding box coordinates [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords
            
            # Calculate the center point of the object
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Determine spatial position (e.g., Top-Left, Bottom-Right)
            vertical_pos = "Top" if center_y < (img_height / 2) else "Bottom"
            horizontal_pos = "Left" if center_x < (img_width / 2) else "Right"
            
            detected_objects.append(f"- {class_name} ({vertical_pos}-{horizontal_pos})")
            
        # Build the final metadata string for the LLM
        if not counts:
            return "YOLOv8 Analysis: No objects detected in the scene."
            
        summary = "YOLOv8 Analysis Results:\n"
        summary += "Total Object Counts:\n"
        for name, count in counts.items():
            summary += f"- {count} {name}(s)\n"
            
        summary += "\nSpatial Layout (Relative Positions):\n"
        summary += "\n".join(detected_objects)
        
        return summary
