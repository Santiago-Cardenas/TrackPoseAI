from motion_analyzer import MotionAnalyzer
from pathlib import Path

if __name__ == "__main__":
    clips_directory = Path("src/data/clips")
    csv_output_path = Path("src/data/output/motion_data.csv")
    
    analyzer = MotionAnalyzer()
    result_dataframe = analyzer.analyze_all_clips(clips_directory, csv_output_path)