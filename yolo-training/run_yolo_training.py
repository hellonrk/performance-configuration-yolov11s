import os
import subprocess
import sys

def run_yolo_training():
    """
    Run YOLO training with user-specified parameters including augmentation level
    """
    # Ask for model path
    model_path = input("Enter the path to your best.pt model: ").strip()
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    
    # Ask for dataset.yaml location
    dataset_yaml = input("Enter the path to your dataset.yaml file: ").strip()
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset YAML file '{dataset_yaml}' not found")
        return
    
    # Ask for training parameters
    epochs = input("Enter number of epochs (default: 25): ").strip()
    epochs = epochs if epochs else "25"
    
    batch_size = input("Enter batch size (default: 16): ").strip()
    batch_size = batch_size if batch_size else "16"
    
    learning_rate = input("Enter learning rate (default: 0.001 for fine-tuning): ").strip()
    learning_rate = learning_rate if learning_rate else "0.001"
    
    # Ask for augmentation option
    print("\nAugmentation Options:")
    print("1 - Disable augmentation (augment=False)")
    print("2 - Enable default augmentation (augment=True)")
    
    aug_choice = input("Enter augmentation option (1-2, default: 2): ").strip()
    
    # Set augmentation parameter
    if aug_choice == "1":
        aug_param = "augment=False"
    else:
        aug_param = "augment=True"
    
    # Construct the command
    command = f"yolo train model={model_path} data={dataset_yaml} epochs={epochs} imgsz=640 batch={batch_size} lr0={learning_rate} {aug_param}"
    
    # Display the command
    print("\nRunning command:")
    print(command)
    print("\nTraining started. This may take a while...\n")
    
    try:
        # Run the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print("\nTraining completed successfully!")
        else:
            print(f"\nTraining failed with return code {return_code}")
            
    except Exception as e:
        print(f"\nError running training command: {str(e)}")

if __name__ == "__main__":
    print("=== YOLOv11 Training Script ===")
    run_yolo_training()