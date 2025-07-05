import os
import csv
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2
from collections import Counter
import sys

def get_valid_path(prompt, check_exists=True, is_file=False):
    """Get a valid file/directory path from user input."""
    while True:
        path = input(prompt)
        
        # Check if path exists when required
        if check_exists and not os.path.exists(path):
            print(f"Error: The path '{path}' does not exist. Please enter a valid path.")
            continue
            
        # Check if path is a file when required
        if check_exists and is_file and not os.path.isfile(path):
            print(f"Error: '{path}' is not a file. Please enter a valid file path.")
            continue
            
        # If we get here, path is valid
        return path

def count_performers(image_path, model):
    """Count performers in an image using the YOLO model."""
    if not os.path.exists(image_path):
        return 0, 0  # Return count 0 and confidence 0 if image doesn't exist
    
    # Run detection
    results = model(image_path, conf=0.25, iou=0.7)
    
    # Extract boxes and confidences
    boxes = results[0].boxes
    count = len(boxes)
    
    # Calculate average confidence
    avg_conf = 0
    if count > 0:
        avg_conf = float(boxes.conf.mean())
    
    return count, avg_conf

def classify_performance(count):
    """Classify performance based on performer count."""
    if count == 1:
        return "Solo"
    elif count == 2:
        return "Duo"
    elif 3 <= count <= 5:
        return "Small Group"
    else:  # count > 5
        return "Large Group"

def analyze_consistency(counts):
    """Analyze if manual review is needed based on count consistency."""
    # Filter out zeros (missing images)
    valid_counts = [c for c in counts if c > 0]
    
    if not valid_counts:
        return "No valid frames found", 0
    
    # Check if all counts are the same
    if len(set(valid_counts)) == 1:
        return "Consistent (100%)", 100
    
    # Count frequency of each performer count
    count_freq = Counter(valid_counts)
    most_common = count_freq.most_common(1)[0]
    
    # Calculate percentage of agreement
    agreement_pct = (most_common[1] / len(valid_counts)) * 100
    
    if agreement_pct >= 60:
        return f"Mostly consistent ({agreement_pct:.0f}%)", agreement_pct
    else:
        return f"Inconsistent - Manual review required ({agreement_pct:.0f}%)", agreement_pct

def get_final_verdict(consistency_pct, avg_confidence):
    """Generate a final verdict based on consistency and confidence."""
    if consistency_pct == 0:
        return "INVALID: No valid frames found for analysis"
    
    if consistency_pct == 100:
        confidence_level = ""
        if avg_confidence >= 0.8:
            confidence_level = "HIGH"
        elif avg_confidence >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        return f"RELIABLE: Consistent detection with {confidence_level} confidence"
    
    elif consistency_pct >= 80:
        return "MOSTLY RELIABLE: Good consistency but some variation between frames"
    
    elif consistency_pct >= 60:
        return "ACCEPTABLE: Majority agreement but should be verified"
    
    else:
        return "UNRELIABLE: Manual review required due to inconsistent detections"

def main():
    print("\n===== YOLOv11 Performer Detection Analysis =====\n")
    
    # Get path to model
    model_path = get_valid_path("Enter path to trained YOLO model (.pt file): ", check_exists=True, is_file=True)
    
    # Get path to frames directory
    frames_dir = get_valid_path("Enter path to folder containing video frames: ", check_exists=True)
    
    # Get CSV input path
    csv_input = get_valid_path("Enter path to input CSV with video numbers and links: ", check_exists=True, is_file=True)
    
    # Get CSV output path
    csv_output = get_valid_path("Enter path for output CSV results (will be created): ", check_exists=False)
    
    # Load the YOLOv11 model
    print(f"\nLoading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load the original CSV to get video numbers and Vimeo links
    try:
        df_input = pd.read_csv(csv_input)
        video_info = dict(zip(df_input['video_number'], df_input['vimeo_link']))
        print(f"Loaded {len(video_info)} videos from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Process all videos
    results = []
    
    print(f"\nAnalyzing frames with YOLOv11 model...")
    for video_number, vimeo_link in tqdm(video_info.items(), desc="Processing videos"):
        # Check if frames exist for this video
        sample_counts = []
        sample_confidences = []
        sample_details = []
        
        # Process each of the 5 frames
        for i in range(1, 6):
            image_path = os.path.join(frames_dir, f"{video_number}_{i}.jpg")
            count, confidence = count_performers(image_path, model)
            
            sample_counts.append(count)
            sample_confidences.append(confidence)
            
            # Format sample details
            if count > 0:
                classification = classify_performance(count)
                sample_details.append(f"{count} performers ({classification}, conf: {confidence:.2f})")
            else:
                sample_details.append("No frame found")
        
        # Analyze consistency
        remarks, consistency_pct = analyze_consistency(sample_counts)
        
        # Calculate average confidence (only for frames with detections)
        valid_confidences = [conf for count, conf in zip(sample_counts, sample_confidences) if count > 0]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
        
        # Get final verdict
        verdict = get_final_verdict(consistency_pct, avg_confidence)
        
        # Find most common count (mode)
        valid_counts = [c for c in sample_counts if c > 0]
        most_common_count = Counter(valid_counts).most_common(1)[0][0] if valid_counts else 0
        
        # Determine final classification
        final_classification = classify_performance(most_common_count) if most_common_count > 0 else "Unknown"
        
        # Add to results
        results.append({
            'video_number': video_number,
            'vimeo_link': vimeo_link,
            'sample_1': sample_details[0],
            'sample_2': sample_details[1],
            'sample_3': sample_details[2],
            'sample_4': sample_details[3],
            'sample_5': sample_details[4],
            'remarks': remarks,
            'consistency': consistency_pct,
            'avg_confidence': avg_confidence if valid_confidences else 0,
            'most_common_count': most_common_count,
            'classification': final_classification,
            'verdict': verdict
        })
    
    # Save results to CSV
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_number', 'vimeo_link', 'sample_1', 'sample_2', 
                     'sample_3', 'sample_4', 'sample_5', 'remarks', 
                     'consistency', 'avg_confidence', 'most_common_count', 
                     'classification', 'verdict']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Generate summary statistics
    total_videos = len(results)
    consistent_videos = sum(1 for r in results if r['consistency'] == 100)
    mostly_consistent = sum(1 for r in results if 60 <= r['consistency'] < 100)
    needs_review = sum(1 for r in results if r['consistency'] < 60)
    no_frames = sum(1 for r in results if r['remarks'] == "No valid frames found")
    
    # Classification statistics
    classification_counts = Counter([r['classification'] for r in results if r['classification'] != "Unknown"])
    
    print("\n===== Analysis Complete! =====")
    print(f"Results saved to {csv_output}")
    
    print("\nSummary Statistics:")
    print(f"Total videos processed: {total_videos}")
    print(f"Fully consistent results (100%): {consistent_videos} ({consistent_videos/total_videos*100:.1f}%)")
    print(f"Mostly consistent results (60-99%): {mostly_consistent} ({mostly_consistent/total_videos*100:.1f}%)")
    print(f"Needs manual review (<60% consistency): {needs_review} ({needs_review/total_videos*100:.1f}%)")
    print(f"No frames found: {no_frames} ({no_frames/total_videos*100:.1f}%)")
    
    print("\nPerformance Classification Distribution:")
    for classification, count in classification_counts.most_common():
        print(f"{classification}: {count} videos ({count/total_videos*100:.1f}%)")
    
    # Create a filtered CSV for videos that need manual review
    if needs_review > 0:
        manual_review_df = pd.DataFrame([r for r in results if r['consistency'] < 60])
        manual_review_file = os.path.splitext(csv_output)[0] + "_needs_review.csv"
        manual_review_df.to_csv(manual_review_file, index=False)
        print(f"\nVideos needing manual review saved to {manual_review_file}")
    
    print("\nThank you for using the YOLOv11 Performer Detection Analysis tool!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")