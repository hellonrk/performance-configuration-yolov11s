import pandas as pd
import subprocess
import os
import random
import time
import threading
import queue
import shutil
from datetime import datetime
import cv2
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"vimeo_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Global variables
DOWNLOAD_DIR = "temp_downloads"
OUTPUT_DIR = "final-dataset"
MAX_WORKERS = 4  # Number of parallel threads
RATE_LIMIT_DELAY = (3, 8)  # Random delay range in seconds between requests (min, max)
DOWNLOAD_TIMEOUT = 600  # 10 minutes timeout for downloads
LOCK = threading.Lock()  # Lock for thread-safe operations

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_video(url, video_number):
    """
    Download a Vimeo video in the best available format
    
    Returns:
    str: Path to downloaded file or None if failed
    """
    output_path = os.path.join(DOWNLOAD_DIR, f"{video_number}_temp.mp4")
    
    # Command to download the video in best quality
    command = [
        "yt-dlp",
        "--format", "bestvideo+bestaudio/best",  # Best quality available
        "--merge-output-format", "mp4",          # Merge to mp4 format
        "--output", output_path,                 # Output file path
        "--no-warnings",                         # Hide warnings
        "--no-check-certificate",                # Skip HTTPS certificate validation
        url                                      # Video URL
    ]
    
    try:
        logger.info(f"Downloading video {video_number} from {url}")
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=DOWNLOAD_TIMEOUT
        )
        
        if process.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Successfully downloaded video {video_number}")
            return output_path
        else:
            logger.error(f"Failed to download video {video_number}: {process.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while downloading video {video_number}")
        return None
    except Exception as e:
        logger.error(f"Exception while downloading video {video_number}: {str(e)}")
        return None

def capture_random_frames(video_path, video_number, num_frames=5):
    """
    Capture random frames from a video
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count <= 0:
            logger.error(f"Invalid frame count for video {video_number}: {frame_count}")
            cap.release()
            return False
        
        # Ensure the output directory exists
        frame_dir = os.path.join(OUTPUT_DIR, str(video_number))
        ensure_dir_exists(frame_dir)
        
        # Choose random timestamps
        if frame_count < num_frames:
            # If video has fewer frames than requested, use evenly spaced frames
            frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        else:
            # Otherwise, choose random frames
            frame_indices = sorted(random.sample(range(frame_count), num_frames))
        
        # Extract and save frames
        for i, frame_idx in enumerate(frame_indices, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(frame_dir, f"{video_number}_{i}.jpg")
                cv2.imwrite(output_path, frame)
                logger.info(f"Saved frame {i} for video {video_number}")
            else:
                logger.warning(f"Failed to extract frame {i} for video {video_number}")
        
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"Error capturing frames for video {video_number}: {str(e)}")
        return False

def secure_delete_file(file_path):
    """Securely delete a file and ensure it's removed from trash"""
    try:
        if os.path.exists(file_path):
            # First try normal removal
            os.remove(file_path)
            
            # For extra security, you can use the following on macOS (uncomment if needed)
            # subprocess.run(["rm", "-P", file_path], check=True)
            
            logger.info(f"Deleted file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        return False

def process_video(video_number, url, stats):
    """Process a single video: download, capture frames, and clean up"""
    try:
        # Respect rate limiting
        with LOCK:
            delay = random.uniform(RATE_LIMIT_DELAY[0], RATE_LIMIT_DELAY[1])
            time.sleep(delay)
        
        # Download the video
        video_path = download_video(url, video_number)
        if not video_path:
            with LOCK:
                stats['failed'] += 1
            return False
        
        # Capture random frames
        frames_success = capture_random_frames(video_path, video_number)
        
        # Clean up the downloaded video
        secure_delete_file(video_path)
        
        # Update stats
        with LOCK:
            if frames_success:
                stats['successful'] += 1
            else:
                stats['partial'] += 1
        
        return frames_success
    
    except Exception as e:
        logger.error(f"Unexpected error processing video {video_number}: {str(e)}")
        with LOCK:
            stats['failed'] += 1
        return False

def batch_process_videos(csv_file, limit=None):
    """
    Process videos from CSV with multithreading
    
    Parameters:
    csv_file (str): Path to the CSV file
    limit (int): Maximum number of videos to process (None for all)
    """
    start_time = time.time()
    
    # Create necessary directories
    ensure_dir_exists(DOWNLOAD_DIR)
    ensure_dir_exists(OUTPUT_DIR)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"CSV loaded successfully. Found {len(df)} videos.")
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return
    
    # Check if required columns exist
    if "video_number" not in df.columns or "vimeo_link" not in df.columns:
        logger.error(f"Error: CSV must contain 'video_number' and 'vimeo_link' columns")
        logger.error(f"Found columns: {', '.join(df.columns)}")
        return
    
    # Limit to specified number of videos if requested
    if limit:
        df = df.head(limit)
    
    # Stats tracking
    stats = {
        'total': len(df),
        'successful': 0,
        'partial': 0,
        'failed': 0
    }
    
    logger.info(f"Starting processing of {stats['total']} videos with {MAX_WORKERS} workers")
    
    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_video, row["video_number"], row["vimeo_link"], stats): row["video_number"]
            for _, row in df.iterrows() if not pd.isna(row["vimeo_link"]) and row["vimeo_link"].strip() != ""
        }
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_video):
            video_number = future_to_video[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video {video_number}: {str(e)}")
    
    # Clean up temporary directory
    try:
        if os.path.exists(DOWNLOAD_DIR) and len(os.listdir(DOWNLOAD_DIR)) == 0:
            os.rmdir(DOWNLOAD_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    logger.info("=" * 50)
    logger.info(f"Processing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Total videos: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Partial success: {stats['partial']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("=" * 50)

if __name__ == "__main__":
    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        logger.info("yt-dlp is not installed. Installing now...")
        subprocess.run(["pip", "install", "yt-dlp"], stdout=subprocess.PIPE)
        logger.info("yt-dlp installed successfully")
    
    # Check for OpenCV
    try:
        import cv2
    except ImportError:
        logger.info("OpenCV is not installed. Installing now...")
        subprocess.run(["pip", "install", "opencv-python"], stdout=subprocess.PIPE)
        import cv2
        logger.info("OpenCV installed successfully")
    
    # Replace with your CSV file path
    csv_file = "final-list.csv"
    
    # Process all videos (or specify a limit)
    batch_process_videos(csv_file)