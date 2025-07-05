import os
from pathlib import Path
from ultralytics import YOLO

def create_validation_labels(model_path, val_images_dir, val_labels_dir):
    """
    Use trained model to generate initial validation labels
    
    Args:
        model_path: Path to trained model (.pt file)
        val_images_dir: Directory with validation images
        val_labels_dir: Directory for validation labels
    """
    # Create output directory if needed
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # Get all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(Path(val_images_dir).glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} validation images")
    
    # Process each image
    for img_path in image_files:
        results = model(img_path)
        
        # Create label file path
        label_path = os.path.join(val_labels_dir, f"{img_path.stem}.txt")
        
        # Write detections to label file
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get normalized xywh coordinates
                    x, y, w, h = box.xywhn[0].tolist()
                    cls = int(box.cls.item())
                    
                    # Write YOLO format: class x_center y_center width height
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    print(f"Created {len(image_files)} validation label files")

# Example usage
model_path = "/Users/nawa/ultralytics/runs/detect/train7/weights/best.pt"
val_images_dir = "/Users/nawa/Desktop/for-dataset/final-database/final-step/4th-batch/retraining/images/val"
val_labels_dir = "/Users/nawa/Desktop/for-dataset/final-database/final-step/4th-batch/retraining/labels/val"

create_validation_labels(model_path, val_images_dir, val_labels_dir)