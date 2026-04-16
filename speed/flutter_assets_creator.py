import os
import shutil

# Source directory
source_dir = 'focus_clips'  # change if needed
# Destination directory
dest_dir = 'flutter_assets'

# Create destination folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Prepare list for YAML assets
assets_list = []

# Iterate through each folder in source_dir
for label in os.listdir(source_dir):
    src_folder = os.path.join(source_dir, label)
    if os.path.isdir(src_folder):
        dest_folder = os.path.join(dest_dir, label)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Find video files
        video_files = [f for f in os.listdir(src_folder) 
                       if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        if video_files:
            # Take the first video
            source_video = os.path.join(src_folder, video_files[0])
            dest_video = os.path.join(dest_folder, 'video_1.mp4')
            shutil.copy2(source_video, dest_video)
            print(f"Copied {source_video} -> {dest_video}")
        
        # Add to assets YAML list
        assets_list.append(f"    - assets/gestures/{label}/")

# Print YAML-ready assets list
print("\nassets:")
for item in assets_list:
    print(item)