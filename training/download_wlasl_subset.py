import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import cv2
import requests
from tqdm import tqdm

# --- Configuration ---

# The 40-word vocabulary for the ASL translator project.
TARGET_VOCAB = {
    "hello", "thank you", "please", "sorry", "yes", "no", "help", "want", 
    "need", "like", "love", "more", "finish", "again", "learn", "understand", 
    "know", "name", "my", "you", "friend", "family", "mom", "dad", "home", 
    "school", "work", "eat", "drink", "water", "hungry", "happy", "sad", 
    "sick", "tired", "good", "bad", "where", "what", "who"
}

# Path to the WLASL metadata file.
METADATA_PATH = pathlib.Path("raw_data/WLASL_v0.3.json")

# Base directory where the final, extracted video clips will be saved.
OUTPUT_DIR = pathlib.Path("../data/raw_videos")


def download_video(url, temp_path):
    """
    Downloads a video from a given URL. Supports YouTube and direct links.
    Returns True on success, False on failure.
    """
    try:
        # Use yt-dlp for YouTube URLs
        if "youtube.com" in url or "youtu.be" in url:
            # Run yt-dlp as a subprocess to download the video.
            # -f 'best[ext=mp4]': Download the best quality MP4 format.
            # -o '{temp_path}': Specify the output file path.
            # --quiet: Suppress console output.
            # --no-warnings: Suppress warnings.
            subprocess.run(
                ["yt-dlp", "-f", "best[ext=mp4]", "-o", str(temp_path), url],
                check=True,
                capture_output=True,
                text=True
            )
        # Use requests for other direct video URLs
        else:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except (subprocess.CalledProcessError, requests.RequestException, Exception) as e:
        # Log errors for failed downloads (e.g., 404, private video)
        # print(f"   -> Failed to download {url}: {e}")
        return False


def extract_clip(source_path, dest_path, frame_start, frame_end, fps):
    """
    Extracts a clip from a source video file using OpenCV and saves it.
    Returns True on success, False on failure.
    """
    try:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            # print(f"   -> Error: Could not open video file {source_path}")
            return False

        # Get video properties for writing the output clip
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # Use original FPS if the metadata FPS is invalid
        output_fps = video_fps if fps <= 0 else fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(str(dest_path), fourcc, output_fps, (width, height))

        # If frame_end is -1, read until the end of the video
        is_full_video = (frame_end == -1)
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame if it's within the desired range
            if current_frame >= frame_start and (is_full_video or current_frame <= frame_end):
                writer.write(frame)

            if not is_full_video and current_frame > frame_end:
                break
                
            current_frame += 1

        cap.release()
        writer.release()
        return True
    except Exception as e:
        # print(f"   -> Failed to extract clip from {source_path}: {e}")
        if 'writer' in locals() and writer.isOpened():
            writer.release()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return False


def main():
    """
    Main function to filter, download, and process WLASL video clips.
    """
    print("Starting WLASL subset download and processing script.")

    # 1. Check for metadata file
    if not METADATA_PATH.exists():
        print(f"\nERROR: Metadata file not found at '{METADATA_PATH}'")
        print("Please download 'WLASL_v0.3.json' from the official WLASL GitHub repository:")
        print("https://github.com/dxli94/WLASL/blob/master/start_kit/WLASL_v0.3.json")
        print(f"And place it in the '{METADATA_PATH.parent}' directory.")
        return

    # 2. Load and filter metadata
    print(f"Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, "r") as f:
        all_data = json.load(f)

    # Create a flat list of all video instances that match our vocabulary
    target_instances = []
    for entry in all_data:
        gloss = entry["gloss"]
        if gloss in TARGET_VOCAB:
            for instance in entry["instances"]:
                # Add the gloss to each instance for easier access later
                instance["gloss"] = gloss
                target_instances.append(instance)

    print(f"Found {len(target_instances)} video instances for {len(TARGET_VOCAB)} target words.")

    # 3. Create output directories and process videos
    print(f"Ensuring output directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    successful_clips = 0
    failed_clips = 0
    gloss_counts = {gloss: 0 for gloss in TARGET_VOCAB}

    # Use a temporary directory for raw downloads to keep things clean
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        print(f"Using temporary directory for downloads: {temp_dir}")

        for instance in tqdm(target_instances, desc="Processing videos"):
            gloss = instance["gloss"]
            video_id = instance["video_id"]
            instance_id = instance["instance_id"]
            url = instance["url"]

            # Define paths
            gloss_dir = OUTPUT_DIR / gloss
            gloss_dir.mkdir(exist_ok=True)
            
            # Use a unique name for the temporary download to avoid conflicts
            temp_video_file = temp_path / f"{video_id}_{instance_id}.mp4"
            final_clip_path = gloss_dir / f"{video_id}_{instance_id}.mp4"

            # Skip if the clip already exists
            if final_clip_path.exists():
                # tqdm.write(f"Skipping existing clip: {final_clip_path}")
                successful_clips += 1
                gloss_counts[gloss] += 1
                continue

            # Download the full source video
            if not download_video(url, temp_video_file):
                failed_clips += 1
                continue

            # Extract the relevant clip
            frame_start = instance["frame_start"]
            frame_end = instance["frame_end"]
            fps = instance["fps"]

            if extract_clip(temp_video_file, final_clip_path, frame_start, frame_end, fps):
                successful_clips += 1
                gloss_counts[gloss] += 1
            else:
                failed_clips += 1
                # Clean up failed extraction artifact if it exists
                if final_clip_path.exists():
                    final_clip_path.unlink()

            # Clean up the large temporary file immediately
            if temp_video_file.exists():
                temp_video_file.unlink()

    # 4. Print final summary report
    print("\n--- Processing Complete ---")
    print(f"Total clips successfully downloaded/verified: {successful_clips}")
    print(f"Total clips failed or skipped: {failed_clips}")
    
    words_with_clips = sum(1 for count in gloss_counts.values() if count > 0)
    print(f"\n{words_with_clips} out of {len(TARGET_VOCAB)} words have at least one clip.")

    words_with_zero_clips = [gloss for gloss, count in gloss_counts.items() if count == 0]
    if words_with_zero_clips:
        print("\nWords with 0 successful clips (you may need to find alternative sources):")
        for gloss in sorted(words_with_zero_clips):
            print(f"- {gloss}")
    else:
        print("\nGreat! All words in the vocabulary have at least one video clip.")


if __name__ == "__main__":
    main()
