from pathlib import Path
import pandas as pd
from body_tracker import BodyTracker

class MotionAnalyzer:
    def __init__(self):
        self.body_tracker = BodyTracker()

    def analyze_all_clips(self, source_folder, destination_csv):
        folder_path = Path(source_folder)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        collected_data = []
        clip_files = list(folder_path.glob('*.mp4'))

        if not clip_files:
            raise ValueError(f"No MP4 files found in: {folder_path}")

        for clip_file in clip_files:
            print(f"Analyzing clip: {clip_file.name}")
            
            # Extracting landmarks from the clip
            clip_dataframe = self.body_tracker.extract_landmarks_from_clip(clip_file)

            if clip_dataframe.empty:
                print(f"No landmarks detected in {clip_file.name}")
                continue

            # Parsing meta-data from the filename
            filename_segments = clip_file.stem.split('_')
            print(f"Parsing filename: {filename_segments}")

            # Determining movement type and speed
            if 'walk' in filename_segments[0]:
                movement_components = filename_segments[0].split('-')
                movement_type = '_'.join(movement_components)
                speed = filename_segments[-1]
            else:
                movement_type = filename_segments[0]
                speed = filename_segments[1] if len(filename_segments) > 1 else 'undefined'

            # Adding labels to the dataframe
            clip_dataframe['movement_type'] = movement_type
            clip_dataframe['speed'] = speed

            collected_data.append(clip_dataframe)
            print(f"✓ {clip_file.name} completed - {len(clip_dataframe)} frames analyzed")

        # Validating that data was processed
        if not collected_data:
            raise ValueError("Failed to process any video data")

        # Consolidating and saving
        consolidated_dataframe = pd.concat(collected_data, ignore_index=True)
        output_location = Path(destination_csv)
        output_location.parent.mkdir(parents=True, exist_ok=True)
        consolidated_dataframe.to_csv(destination_csv, index=False)

        print(f"\n✓ Dataset saved to: {destination_csv}")
        print(f"  Frames analyzed: {len(consolidated_dataframe)}")
        print(f"  Movement types detected: {consolidated_dataframe['movement_type'].unique()}")

        return consolidated_dataframe