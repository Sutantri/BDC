import pandas as pd
import os
import time
from yt_dlp import YoutubeDL
import re
from urllib.parse import urlparse
import concurrent.futures
from threading import Lock

# Configuration
MAX_WORKERS = 3  # Number of parallel downloads (default)
DOWNLOAD_DELAY = 3  # Delay between downloads (seconds)
MAX_RETRIES = 3  # Maximum retry attempts per video

# Global variables for thread safety
download_lock = Lock()
success_count = 0
failed_count = 0
failed_downloads = []

def sanitize_filename(filename):
    """Sanitize filename to be valid for filesystem"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s-]', '_', filename)
    filename = filename.strip()
    if len(filename) > 50:
        filename = filename[:50]
    return filename

def download_single_video(url, output_path, video_id, max_retries=MAX_RETRIES):
    """Download a single video with retry mechanism"""
    global success_count, failed_count, failed_downloads
    
    for attempt in range(max_retries):
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[height<=480]/best',  # Lower quality for faster download
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
                'writeinfojson': False,
                'writedescription': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'socket_timeout': 30,
                'retries': 2,
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            with download_lock:
                success_count += 1
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for video {video_id}: {error_msg}")
                time.sleep(DOWNLOAD_DELAY * (attempt + 1))
            else:
                with download_lock:
                    failed_count += 1
                    failed_downloads.append({
                        'id': video_id, 
                        'url': url, 
                        'error': error_msg
                    })
                return False, error_msg
    
    return False, "Max retries exceeded"

def download_video_worker(args):
    """Worker function for parallel downloading"""
    row, output_dir, dataset_type = args
    
    video_id = row['id']
    video_url = row['video']
    
    if pd.isna(video_url):
        print(f"Video {video_id}: No URL found")
        return False
    
    # Create filename based on emotion (for training) or just id (for test)
    if dataset_type == 'train' and 'emotion' in row:
        emotion = str(row['emotion']).replace(' ', '_')
        emotion_dir = os.path.join(output_dir, sanitize_filename(emotion))
        os.makedirs(emotion_dir, exist_ok=True)
        filename = f"{video_id}_{emotion}.%(ext)s"
        output_path = os.path.join(emotion_dir, filename)
    else:
        filename = f"{video_id}.%(ext)s"
        output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading {dataset_type} video {video_id}...")
    
    success, error = download_single_video(video_url, output_path, video_id)
    
    if success:
        print(f"✅ Downloaded: {video_id}")
    else:
        print(f"❌ Failed {video_id}: {error}")
    
    time.sleep(DOWNLOAD_DELAY)
    return success

def download_dataset(csv_path, output_dir, dataset_type, use_parallel=True, num_workers=MAX_WORKERS):
    """Download all videos from a dataset CSV"""
    global success_count, failed_count, failed_downloads
    
    # Reset counters
    success_count = 0
    failed_count = 0
    failed_downloads = []
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n{dataset_type.upper()} DATASET")
        print(f"Found {len(df)} videos to download")
        print(f"Columns: {df.columns.tolist()}")
        
        if 'emotion' in df.columns:
            emotions = df['emotion'].value_counts()
            print(f"Emotion distribution: {dict(emotions)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare worker arguments
    worker_args = [(row, output_dir, dataset_type) for _, row in df.iterrows()]
    
    start_time = time.time()
    
    if use_parallel and len(df) > 1:
        print(f"Starting parallel download with {num_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(download_video_worker, worker_args))
    else:
        print("Starting sequential download...")
        results = [download_video_worker(args) for args in worker_args]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{dataset_type.upper()} DOWNLOAD SUMMARY")
    print(f"{'='*40}")
    print(f"Total videos: {len(df)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed downloads: {failed_count}")
    print(f"Success rate: {(success_count/len(df)*100):.1f}%")
    print(f"Time taken: {total_time:.1f} seconds")
    
    # Save failed downloads log
    if failed_downloads:
        failed_df = pd.DataFrame(failed_downloads)
        failed_log = os.path.join(output_dir, f"failed_{dataset_type}_downloads.csv")
        failed_df.to_csv(failed_log, index=False)
        print(f"Failed downloads saved to: {failed_log}")
    
    return success_count, failed_count

def create_updated_csv(original_csv, download_dir, dataset_type, output_csv=None):
    """Create CSV with local file paths"""
    try:
        df = pd.read_csv(original_csv)
        local_paths = []
        
        for _, row in df.iterrows():
            video_id = row['id']
            
            # Look for downloaded file
            found_file = None
            extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']
            
            if dataset_type == 'train' and 'emotion' in row:
                emotion = str(row['emotion']).replace(' ', '_')
                emotion_dir = os.path.join(download_dir, sanitize_filename(emotion))
                
                for ext in extensions:
                    file_path = os.path.join(emotion_dir, f"{video_id}_{emotion}{ext}")
                    if os.path.exists(file_path):
                        found_file = file_path
                        break
            else:
                for ext in extensions:
                    file_path = os.path.join(download_dir, f"{video_id}{ext}")
                    if os.path.exists(file_path):
                        found_file = file_path
                        break
            
            local_paths.append(found_file)
        
        df['local_path'] = local_paths
        
        if output_csv is None:
            name, ext = os.path.splitext(original_csv)
            output_csv = f"{name}_with_paths{ext}"
        
        df.to_csv(output_csv, index=False)
        
        successful_paths = sum(1 for path in local_paths if path is not None)
        print(f"Updated CSV saved: {output_csv}")
        print(f"Successfully mapped {successful_paths}/{len(df)} videos")
        
        return output_csv
        
    except Exception as e:
        print(f"Error creating updated CSV: {e}")
        return None

def main():
    """Main execution function"""
    print("EMOTION CLASSIFICATION - Video Batch Downloader")
    print("="*50)
    
    # Configuration
    TRAIN_CSV = "D:/BDC/datatrain.csv"
    TEST_CSV = "D:/BDC/datatest.csv"
    BASE_OUTPUT_DIR = "D:/BDC/downloaded_videos"
    
    # Preview datasets
    print("DATASET PREVIEW:")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        
        print(f"\nTraining data: {len(train_df)} videos")
        print(f"Columns: {train_df.columns.tolist()}")
        if 'emotion' in train_df.columns:
            print(f"Emotions: {train_df['emotion'].value_counts().to_dict()}")
        print(f"Sample: {train_df.head(2).to_dict('records')}")
        
        print(f"\nTest data: {len(test_df)} videos") 
        print(f"Columns: {test_df.columns.tolist()}")
        print(f"Sample: {test_df.head(2).to_dict('records')}")
        
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    # User choices
    print("\nSelect download option:")
    print("1. Download training data only")
    print("2. Download test data only")
    print("3. Download both datasets")
    print("4. Preview only (no download)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "4":
        return
    
    # Download settings
    parallel = input("Use parallel downloads? (y/n) [default: y]: ").strip().lower()
    use_parallel = parallel != 'n'
    
    num_workers = MAX_WORKERS  # pakai default
    if use_parallel:
        workers = input(f"Number of workers [default: {MAX_WORKERS}]: ").strip()
        if workers.isdigit():
            num_workers = int(workers)
    
    # Execute downloads
    if choice in ["1", "3"]:
        train_output = os.path.join(BASE_OUTPUT_DIR, "train")
        print(f"\nDownloading training videos to: {train_output}")
        train_success, train_failed = download_dataset(
            TRAIN_CSV, train_output, "train", use_parallel, num_workers
        )
        
        # Create updated CSV
        create_updated_csv(
            TRAIN_CSV, train_output, "train", 
            "D:/BDC/datatrain_with_paths.csv"
        )
    
    if choice in ["2", "3"]:
        test_output = os.path.join(BASE_OUTPUT_DIR, "test")
        print(f"\nDownloading test videos to: {test_output}")
        test_success, test_failed = download_dataset(
            TEST_CSV, test_output, "test", use_parallel, num_workers
        )
        
        # Create updated CSV
        create_updated_csv(
            TEST_CSV, test_output, "test",
            "D:/BDC/datatest_with_paths.csv"
        )
    
    print("\n" + "="*50)
    print("DOWNLOAD COMPLETED!")
    print("You can now run the emotion classification training script.")
    print("The updated CSV files contain local file paths.")

if __name__ == "__main__":
    main()
