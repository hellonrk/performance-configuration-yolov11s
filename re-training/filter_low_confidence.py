import pandas as pd
import re
import numpy as np

def extract_confidence(text):
    """Extract the confidence value from text like '1 performers (Solo, conf: 0.81)'"""
    if pd.isna(text):
        return None
        
    # Use regular expression to find the confidence value
    match = re.search(r'conf:\s*(0\.\d+)', str(text))
    if match:
        return float(match.group(1))
    return None

def calculate_avg_confidence(row):
    """Calculate the average confidence across all sample columns"""
    confidences = []
    
    # Extract confidence from each sample column
    for i in range(1, 6):
        sample_key = f'sample_{i}'
        if sample_key in row and not pd.isna(row[sample_key]):
            conf = extract_confidence(row[sample_key])
            if conf is not None:
                confidences.append(conf)
    
    # Return average if we have values, otherwise None
    if confidences:
        return sum(confidences) / len(confidences)
    return None

def main():
    try:
        # Read the CSV file
        print("Reading the CSV file...")
        df = pd.read_csv('performer_analysis_results_with_verdict.csv')
        print(f"Successfully read file with {len(df)} rows")
        
        # Calculate average confidence for each row
        print("Calculating average confidence scores...")
        df['avg_confidence'] = df.apply(calculate_avg_confidence, axis=1)
        
        # Convert to percentage for easier reading
        df['avg_confidence_pct'] = df['avg_confidence'] * 100
        
        # Filter rows with average confidence below 70%
        low_conf_df = df[df['avg_confidence'] < 0.7].copy()
        
        # If no rows found with low confidence
        if len(low_conf_df) == 0:
            print("No videos found with average confidence below 70%")
            return
            
        # Sort by average confidence (ascending)
        low_conf_df = low_conf_df.sort_values(by='avg_confidence')
        
        # Save filtered rows to a new CSV
        output_file = 'for_manual_review.csv'
        low_conf_df.to_csv(output_file, index=False)
        
        print(f"\nFound {len(low_conf_df)} videos with average confidence below 70%")
        print(f"Saved to '{output_file}' for manual review")
        
        # Show some statistics
        print(f"\nConfidence distribution of flagged videos:")
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]
        labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-65%', '65-70%']
        
        # Create histogram of confidence values
        hist, _ = np.histogram(low_conf_df['avg_confidence'], bins=bins)
        
        # Print distribution
        for i, count in enumerate(hist):
            print(f"  {labels[i]}: {count} videos")
            
    except FileNotFoundError:
        print("Error: The file 'performer_analysis_results_with_verdict.csv' was not found.")
        print("Make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()