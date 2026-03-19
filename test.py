import subprocess
import os
import sys

def resize_video(input_path, output_path, width, height):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
        
    # Using ffmpeg to scale the video exactly to 886x1920.
    # Note: If you want to maintain the aspect ratio and pad with black bars instead of stretching,
    # you can change the filter to: f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    command = [
        'ffmpeg',
        '-y',               # Overwrite if output exists
        '-i', input_path,   # Input file
        '-vf', f'scale={width}:{height}', # Video filter for resizing
        '-c:a', 'copy',     # Copy audio without re-encoding
        output_path         # Output file
    ]
    
    print(f"Starting conversion of '{input_path}' to {width}x{height}...")
    try:
        subprocess.run(command, check=True)
        print(f"Conversion complete. Output saved to '{output_path}'.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with error code {e.returncode}.")
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in the system's PATH.")
        print("Please install it using 'brew install ffmpeg' (on macOS).")

if __name__ == "__main__":
    input_video = "/Users/apple/Documents/Cellz_demo.mp4"
    
    # Generate output filename based on input filename
    base, ext = os.path.splitext(input_video)
    output_video = f"{base}_886x1920{ext}"
    
    resize_video(input_video, output_video, 886, 1920)
