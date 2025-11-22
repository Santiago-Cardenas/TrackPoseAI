# Entrega3/src/model/activity_predictor.py

from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


class ActivityPredictor:

    def __init__(self, models_dir: Path | None = None):
        """
        Carga el modelo reducido, el scaler y el label encoder.
        También inicializa MediaPipe Pose.
        """

        base_dir = Path(__file__).resolve().parent

        if models_dir is None:
            models_dir = base_dir
        self.models_dir = Path(models_dir)

        self.model = joblib.load(self.models_dir / "rf_core_rfe10.pkl")
        self.scaler = joblib.load(self.models_dir / "scaler_core.pkl")
        self.label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")

        self.core_features = [
            "nose_x", "nose_y", "nose_z", "nose_confidence",
            "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
            "left_shoulder_confidence",
            "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
            "right_shoulder_confidence",
            "left_hip_x", "left_hip_y", "left_hip_z", "left_hip_confidence",
            "right_hip_x", "right_hip_y", "right_hip_z", "right_hip_confidence",
        ]

        self.selected_features = [
            "nose_confidence",
            "left_shoulder_x",
            "right_shoulder_x",
            "right_shoulder_y",
            "right_shoulder_confidence",
            "left_hip_y",
            "left_hip_z",
            "left_hip_confidence",
            "right_hip_y",
            "right_hip_z",
        ]

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )


    def _extract_landmarks(self, video_path: str) -> pd.DataFrame:
        """
        Procesa un video con MediaPipe Pose y devuelve un DataFrame
        con columnas coord_x_i, coord_y_i, coord_z_i, confidence_i.
        """
        cap = cv2.VideoCapture(str(video_path))
        frames_data = []
        clip_name = Path(video_path).name
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if not results.pose_landmarks:
                frame_idx += 1
                continue

            row = {
                "frame_number": frame_idx,
                "clip_name": clip_name,
            }

            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f"coord_x_{i}"] = lm.x
                row[f"coord_y_{i}"] = lm.y
                row[f"coord_z_{i}"] = lm.z
                row[f"confidence_{i}"] = lm.visibility

            frames_data.append(row)
            frame_idx += 1

        cap.release()
        return pd.DataFrame(frames_data)

    def _build_core_view(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Toma el DF crudo de MediaPipe y arma la Vista B/Core
        con los mismos nombres de columnas que usaste para entrenar.
        """
        mapping = {
            "coord_x_0": "nose_x",
            "coord_y_0": "nose_y",
            "coord_z_0": "nose_z",
            "confidence_0": "nose_confidence",

            "coord_x_11": "left_shoulder_x",
            "coord_y_11": "left_shoulder_y",
            "coord_z_11": "left_shoulder_z",
            "confidence_11": "left_shoulder_confidence",

            "coord_x_12": "right_shoulder_x",
            "coord_y_12": "right_shoulder_y",
            "coord_z_12": "right_shoulder_z",
            "confidence_12": "right_shoulder_confidence",

            "coord_x_23": "left_hip_x",
            "coord_y_23": "left_hip_y",
            "coord_z_23": "left_hip_z",
            "confidence_23": "left_hip_confidence",

            "coord_x_24": "right_hip_x",
            "coord_y_24": "right_hip_y",
            "coord_z_24": "right_hip_z",
            "confidence_24": "right_hip_confidence",
        }

        core_df = df_raw[list(mapping.keys())].rename(columns=mapping)
        return core_df

    def _preprocess_features(self, core_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el scaler y toma solo las 10 features seleccionadas.
        """
        X = core_df[self.core_features]
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.core_features)
        X_opt = X_scaled_df[self.selected_features]
        return X_opt


    def predict(self, video_path: str):
        """
        Devuelve:
        - predictions: array de etiquetas por frame (ej: 'caminar_adelante')
        - probs:     matriz de probabilidades por frame (n_frames x n_clases)
        """
        df_raw = self._extract_landmarks(video_path)
        if df_raw.empty:
            raise ValueError("No se detectaron landmarks en el video.")

        core_df = self._build_core_view(df_raw)
        X_opt = self._preprocess_features(core_df)

        y_pred_int = self.model.predict(X_opt)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_int)
        probs = self.model.predict_proba(X_opt)

        return np.array(y_pred_labels), probs

    def get_summary(self, predictions: np.ndarray):
        """
        A partir de las predicciones frame a frame,
        devuelve actividad dominante y distribución porcentual.
        """
        counts = Counter(predictions)
        total = len(predictions)

        distribution = {
            activity: 100.0 * count / total
            for activity, count in counts.items()
        }

        dominant_activity = max(distribution, key=distribution.get)
        dominant_percentage = distribution[dominant_activity]

        return {
            "dominant_activity": dominant_activity,
            "dominant_percentage": dominant_percentage,
            "summary": distribution,
        }
