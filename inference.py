import cv2
from ultralytics import YOLO

# Load the YOLO model (make sure to replace with your specific model if needed)
model = YOLO('yolov8n.pt')  # Path to your YOLO model (yolov8n.pt or other trained models)

def run_inference(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {image_path}")

        # Run inference
        results = model(image)

        # Extract and display results (class names, confidence, and bounding boxes)
        output = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = result.names[int(box.cls)]  # Get class name
                conf = box.conf.item()  # Confidence score
                coords = box.xyxy[0].tolist()  # Bounding box coordinates
                # Draw bounding box and label on the image
                x1, y1, x2, y2 = map(int, coords)
                label = f"{cls} ({conf:.2f})"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label

                output.append(f"{cls} ({conf:.2f}) at {coords}")

        if not output:
            print("No objects detected.")
        else:
            print("\n".join(output))

        # Save and display the annotated image
        output_image_path = "annotated_image.jpg"
        cv2.imwrite(output_image_path, image)  # Save the annotated image to a file
        print(f"Annotated image saved as {output_image_path}")
        
        # Display the annotated image in a window
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Change this path to any image you want to test
    image_path = r'E:\Assignment\data\coco2017\coco\images\val2017\000000002532.jpg'  # Update the image path here
    run_inference(image_path)
