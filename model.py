from ultralytics import YOLO
import cv2

class AIModel:
    def __init__(self, model_path):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def annotate_image(self, image):
        print("Running inference on the image...")
        results = self.model.predict(image)

        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            conf = result.boxes.conf  # Confidence scores
            cls = result.boxes.cls  # Class IDs

            for box, c, cls_id in zip(boxes, conf, cls):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(cls_id)]} {c:.2f}"
                color = (0, 255, 0)  # Green color for bounding boxes

                # Draw the bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
