import cv2
import mediapipe as mp
import pandas as pd

class BodyTracker:
    def __init__(self):
        # Configurar MediaPipe para rastreo corporal
        mp_pose_solution = mp.solutions.pose
        self.tracker = mp_pose_solution.Pose(
            static_image_mode=False, #Using tracking between frames
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks_from_clip(self, clip_path):
        """Take landmarks from a video clip and return them in a DataFrame."""
        video_capture = cv2.VideoCapture(str(clip_path))
        landmarks_records = []
        current_frame_number = 0

        while video_capture.isOpened():
            frame_available, current_frame = video_capture.read()
            
            if not frame_available:
                break
            
            # Convert BGR to RGB Without this colors would be inverted and pose detection would fail
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

            # Pose detection
            detection_result = self.tracker.process(frame_rgb)

            if detection_result.pose_landmarks:
                # Building frame record
                record = {
                    'frame_number': current_frame_number, 
                    'clip_name': clip_path.name
                }
                
                for landmark_index, landmark in enumerate(detection_result.pose_landmarks.landmark):
                    record[f'coord_x_{landmark_index}'] = landmark.x
                    record[f'coord_y_{landmark_index}'] = landmark.y
                    record[f'coord_z_{landmark_index}'] = landmark.z
                    record[f'confidence_{landmark_index}'] = landmark.visibility

                landmarks_records.append(record)

            current_frame_number += 1

        video_capture.release()
        return pd.DataFrame(landmarks_records)