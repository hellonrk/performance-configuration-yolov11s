import os
import csv
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2

# Path configurations
MODEL_PATH = "/Users/nawa/Desktop/for-dataset/yolo-results-improved/yolov11s_improved/weights/best.pt"
FRAMES_DIR = "for-final"
CSV_INPUT = "final-database.csv"
CSV_OUTPUT = "performer_analysis_results.csv"

# Load the YOLOv11s model
model = YOLO(MODEL_PATH)

# Load the original CSV to get video numbers and Vimeo links
df_input = pd.read_csv(CSV_INPUT)
video_info = dict(zip(df_input['video_number'], df_input['vimeo_link']))

# Function to count performers in an image
def count_performers(image_path):
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

# Function to classify performance based on performer count
def classify_performance(count):
    if count == 1:
        return "Solo"
    elif count == 2:
        return "Duo"
    elif 3 <= count <= 5:
        return "Small Group"
    else:  # count > 5
        return "Large Group"

# Function to analyze if manual review is needed
def analyze_consistency(counts):
    # Filter out zeros (missing images)
    valid_counts = [c for c in counts if c > 0]
    
    if not valid_counts:
        return "No valid frames found", 0
    
    # Check if all counts are the same
    if len(set(valid_counts)) == 1:
        return "Consistent (100%)", 100
    
    # Count frequency of each performer count
    from collections import Counter
    count_freq = Counter(valid_counts)
    most_common = count_freq.most_common(1)[0]
    
    # Calculate percentage of agreement
    agreement_pct = (most_common[1] / len(valid_counts)) * 100
    
    if agreement_pct >= 60:
        return f"Mostly consistent ({agreement_pct:.0f}%)", agreement_pct
    else:
        return f"Inconsistent - Manual review required ({agreement_pct:.0f}%)", agreement_pct

# Process all videos
results = []

print(f"Analyzing frames with YOLOv11s model...")
for video_number, vimeo_link in tqdm(video_info.items(), desc="Processing videos"):
    # Check if frames exist for this video
    sample_counts = []
    sample_confidences = []
    sample_details = []
    
    # Process each of the 5 frames
    for i in range(1, 6):
        image_path = os.path.join(FRAMES_DIR, f"{video_number}_{i}.jpg")
        count, confidence = count_performers(image_path)
        
        sample_counts.append(count)
        sample_confidences.append(confidence)
        
        # Format sample details: "3 performers (Duo)" with confidence
        if count > 0:
            classification = classify_performance(count)
            sample_details.append(f"{count} performers ({classification}, conf: {confidence:.2f})")
        else:
            sample_details.append("No frame found")
    
    # Analyze consistency
    remarks, consistency_pct = analyze_consistency(sample_counts)
    
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
        'consistency': consistency_pct
    })

# Save results to CSV
with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['video_number', 'vimeo_link', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5', 'remarks', 'consistency']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# Generate summary statistics
total_videos = len(results)
consistent_videos = sum(1 for r in results if r['consistency'] == 100)
mostly_consistent = sum(1 for r in results if 60 <= r['consistency'] < 100)
needs_review = sum(1 for r in results if r['consistency'] < 60)
no_frames = sum(1 for r in results if r['remarks'] == "No valid frames found")

print("\nAnalysis complete!")
print(f"Results saved to {CSV_OUTPUT}")
print("\nSummary:")
print(f"Total videos processed: {total_videos}")
print(f"Fully consistent results (100%): {consistent_videos} ({consistent_videos/total_videos*100:.1f}%)")
print(f"Mostly consistent results (60-99%): {mostly_consistent} ({mostly_consistent/total_videos*100:.1f}%)")
print(f"Needs manual review (<60% consistency): {needs_review} ({needs_review/total_videos*100:.1f}%)")
print(f"No frames found: {no_frames} ({no_frames/total_videos*100:.1f}%)")

# Create a filtered CSV for videos that need manual review
if needs_review > 0:
    manual_review_df = pd.DataFrame([r for r in results if r['consistency'] < 60])
    manual_review_file = "needs_manual_review.csv"
    manual_review_df.to_csv(manual_review_file, index=False)
    print(f"\nVideos needing manual review saved to {manual_review_file}")