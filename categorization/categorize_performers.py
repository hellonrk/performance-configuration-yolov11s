import pandas as pd
import re

# Function to extract performer count from text like "1 performers (Solo, conf: 0.81)"
def extract_performer_count(text):
    if pd.isna(text):
        return None
    
    # Use regular expression to extract the number at the beginning
    match = re.match(r'^(\d+)', str(text))
    if match:
        return int(match.group(1))
    
    # Check if it contains "5+"
    if "5+" in str(text):
        return "5+"
        
    return None

# Function to categorize performer count
def categorize_performers(count):
    if count is None:
        return "unknown"
        
    if count == "5+" or (isinstance(count, str) and "5+" in count):
        return "large group (5+)"
        
    try:
        count = int(count)
        if count == 1:
            return "solo (1)"
        elif count == 2:
            return "duo (2)"
        elif 3 <= count <= 5:
            return "small group (3-5)"
        elif count > 5:
            return "large group (5+)"
        else:
            return "unknown"
    except (ValueError, TypeError):
        return "unknown"

# Function to determine the final verdict
def determine_verdict(row):
    # Extract counts from each sample
    counts = []
    for i in range(1, 6):
        sample_key = f'sample_{i}'
        if sample_key in row and not pd.isna(row[sample_key]):
            count = extract_performer_count(row[sample_key])
            if count is not None:
                counts.append(count)
    
    if not counts:
        return "unknown"
    
    # Convert counts to categories
    categories = [categorize_performers(count) for count in counts]
    
    # Count occurrences of each category
    category_counts = {}
    for category in categories:
        if category != "unknown":
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
    
    # If no valid categories were found
    if not category_counts:
        return "unknown"
    
    # Find the most common category
    max_count = 0
    most_common = "unknown"
    
    for category, count in category_counts.items():
        if count > max_count:
            max_count = count
            most_common = category
    
    # Special case: If any sample suggests "large group (5+)" and consistency is low
    consistency = 0
    if 'consistency' in row and not pd.isna(row['consistency']):
        try:
            consistency = float(row['consistency'])
        except (ValueError, TypeError):
            consistency = 0
    
    # Check if any category is "large group (5+)"        
    if "large group (5+)" in categories and consistency < 0.8:
        return "large group (5+)"
    
    return most_common

# Main execution
if __name__ == "__main__":
    try:
        # Read the CSV file
        print("Reading the CSV file...")
        df = pd.read_csv('performer_analysis_results.csv')
        print(f"Successfully read file with {len(df)} rows")
        
        # Add the verdict column
        print("Analyzing and adding verdicts...")
        df['new_verdict'] = df.apply(determine_verdict, axis=1)
        
        # Save to a new file
        output_file = 'performer_analysis_results_with_verdict.csv'
        df.to_csv(output_file, index=False)
        print(f"Successfully saved results to '{output_file}'")
        
        # Show statistics
        verdict_counts = df['new_verdict'].value_counts()
        print("\nVerdict distribution:")
        for verdict, count in verdict_counts.items():
            percentage = count / len(df) * 100
            print(f"{verdict}: {count} videos ({percentage:.1f}%)")
            
    except FileNotFoundError:
        print("Error: The file 'performer_analysis_results_with_verdict.csv' was not found.")
        print("Make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")